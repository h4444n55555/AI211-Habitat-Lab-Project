import argparse
import csv
import json
import os
import random
import sys
import math
from contextlib import nullcontext
from collections import defaultdict
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:  # pragma: no cover
    GradScaler = None
    autocast = nullcontext

# Enable imports from task2_vln package root when script is run from repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from data.dataset import JsonlVlnDataset, build_collate_fn  # noqa: E402
from models.vln_policy import CLIPCrossAttentionPolicy  # noqa: E402


ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_class_weights(dataset: JsonlVlnDataset) -> torch.Tensor:
    counts = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    for sample in dataset.samples:
        counts[int(sample.action)] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def eval_loader(model, loader, device, class_weights: torch.Tensor = None, label_smoothing: float = 0.0) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    cm = np.zeros((4, 4), dtype=np.int64)

    for batch in loader:
        x_img = batch["pixel_values"].to(device)
        x_ids = batch["input_ids"].to(device)
        x_mask = batch["attention_mask"].to(device)
        y = batch["actions"].to(device)

        logits = model(pixel_values=x_img, input_ids=x_ids, attention_mask=x_mask)["logits"]
        loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=label_smoothing)
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item() * y.numel()
        total += y.numel()
        correct += (preds == y).sum().item()

        y_np = y.cpu().numpy()
        p_np = preds.cpu().numpy()
        for t, p in zip(y_np, p_np):
            cm[t, p] += 1

    return total_loss / max(total, 1), correct / max(total, 1), cm


def train_one_run(
    run_name: str,
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    out_dir: str,
    class_weights: torch.Tensor,
    label_smoothing: float,
    num_workers: int,
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=max(lr * 0.05, 1e-6))
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if GradScaler is not None else None

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    patience = 3
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n = 0

        for batch in train_loader:
            x_img = batch["pixel_values"].to(device)
            x_ids = batch["input_ids"].to(device)
            x_mask = batch["attention_mask"].to(device)
            y = batch["actions"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(pixel_values=x_img, input_ids=x_ids, attention_mask=x_mask)["logits"]
                loss = F.cross_entropy(logits, y, weight=class_weights.to(device), label_smoothing=label_smoothing)

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss_sum += loss.item() * y.numel()
            n += y.numel()

        train_loss = train_loss_sum / max(n, 1)
        val_loss, val_acc, _ = eval_loader(model, val_loader, device, class_weights=class_weights.to(device), label_smoothing=label_smoothing)
        scheduler.step()

        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        print(f"[{run_name}] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        ckpt = {"model_state": model.state_dict(), "clip_model": "openai/clip-vit-base-patch16", "action_names": ACTION_NAMES}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"[{run_name}] early stopping at epoch={epoch}")
                break

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    # Learning curve plot
    plt.figure(figsize=(8, 4))
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Learning Curves - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close()

    return {
        "run_name": run_name,
        "best_val_acc": float(max(hist["val_acc"]) if hist["val_acc"] else 0.0),
        "final_val_acc": float(hist["val_acc"][-1] if hist["val_acc"] else 0.0),
        "final_val_loss": float(hist["val_loss"][-1] if hist["val_loss"] else 0.0),
    }


@torch.no_grad()
def rollout_sr_spl(model, processor, episodes_path: str, image_root: str, device, videos_dir: str, num_videos: int = 8):
    with open(episodes_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    sr_count = 0
    spl_values = []
    oracle_success_count = 0
    ne_values = []
    records = []

    os.makedirs(videos_dir, exist_ok=True)

    scene_cache: Dict[str, Tuple[object, object]] = {}

    def make_scene_sim(scene_path: str):
        if scene_path in scene_cache:
            return scene_cache[scene_path]
        from habitat_sim.utils.settings import default_sim_settings, make_cfg

        settings = default_sim_settings.copy()
        settings.update(
            {
                "scene": scene_path,
                "width": 256,
                "height": 256,
                "color_sensor": True,
                "depth_sensor": False,
                "semantic_sensor": False,
                "sensor_height": 1.5,
                "seed": 17,
                "silent": True,
                "enable_physics": False,
                "gpu_device_id": -1,
            }
        )
        cfg = make_cfg(settings)
        cfg.sim_cfg.gpu_device_id = -1
        sim = __import__("habitat_sim").Simulator(cfg)
        scene_cache[scene_path] = sim
        return sim

    try:
        for ep_i, ep in enumerate(episodes):
            instruction = ep["instruction"]
            expert = ep["expert_actions"]
            shortest = max(float(ep["shortest_path_length"]), 1e-6)
            image_paths = ep["image_paths"]
            scene_path = ep.get("scene_path")
            if not scene_path:
                continue

            sim = make_scene_sim(scene_path)
            agent = sim.initialize_agent(0)
            from habitat_sim import AgentState

            start_state = AgentState()
            start_state.position = np.array(ep["start_position"], dtype=np.float32)
            agent.set_state(start_state)

            pred_actions: List[int] = []
            path_length = 0.0
            visited_distances: List[float] = []
            frames = []
            previous_position = np.array(start_state.position, dtype=np.float32)
            goal_position = np.array(ep["goal_position"], dtype=np.float32)

            for step_idx, rel_path in enumerate(image_paths):
                abs_path = os.path.join(image_root, rel_path)
                image = Image.open(abs_path).convert("RGB")

                enc = processor(
                    text=[instruction],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                )
                logits = model(
                    pixel_values=enc["pixel_values"].to(device),
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )["logits"]
                pred = int(torch.argmax(logits, dim=-1).item())
                pred_actions.append(pred)

                frame = image.copy()
                d = ImageDraw.Draw(frame)
                d.rectangle([4, 4, 252, 72], fill=(0, 0, 0))
                d.text((10, 10), f"ep={ep['episode_id']} step={step_idx + 1}", fill=(255, 255, 255))
                d.text((10, 28), f"pred={ACTION_NAMES[pred]} | expert={ACTION_NAMES[int(expert[step_idx])]}", fill=(255, 255, 255))
                frames.append(np.array(frame))

                if pred == ACTION_TO_ID["stop"]:
                    break

                sim.step(ACTION_NAMES[pred])
                current_position = np.array(sim.get_agent(0).state.position, dtype=np.float32)
                path_length += float(np.linalg.norm(current_position - previous_position))
                previous_position = current_position
                visited_distances.append(float(np.linalg.norm(current_position - goal_position)))

            final_position = np.array(sim.get_agent(0).state.position, dtype=np.float32)
            final_distance = float(np.linalg.norm(final_position - goal_position))
            oracle_distance = min([float(np.linalg.norm(np.array(ep["start_position"], dtype=np.float32) - goal_position))] + visited_distances) if visited_distances else final_distance
            success = int(final_distance <= 1.5 and pred_actions and pred_actions[-1] == ACTION_TO_ID["stop"])
            oracle_success = int(oracle_distance <= 1.5)
            if success:
                sr_count += 1
            if oracle_success:
                oracle_success_count += 1
            ne_values.append(final_distance)

            spl = success * (shortest / max(shortest, path_length if path_length > 0 else shortest))
            spl_values.append(spl)

            records.append(
                {
                    "episode_id": ep["episode_id"],
                    "success": success,
                    "oracle_success": oracle_success,
                    "spl": float(spl),
                    "navigation_error": final_distance,
                    "shortest_path_length": shortest,
                    "path_length": path_length,
                    "num_pred_steps": len(pred_actions),
                }
            )

            if ep_i < num_videos and frames:
                gif_path = os.path.join(videos_dir, f"rollout_{ep['episode_id']}.gif")
                imageio.mimsave(gif_path, frames, fps=2)
    finally:
        for sim in scene_cache.values():
            sim.close()

    sr = sr_count / max(len(records), 1)
    osr = oracle_success_count / max(len(records), 1)
    spl = float(np.mean(spl_values)) if spl_values else 0.0
    ne = float(np.mean(ne_values)) if ne_values else 0.0
    return sr, spl, osr, ne, records


def plot_confusion(cm: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(4), ACTION_NAMES, rotation=25, ha="right")
    plt.yticks(range(4), ACTION_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 baseline training + hyperparameter tuning + SR/SPL")
    p.add_argument("--data-root", default="task 3/data")
    p.add_argument("--artifacts-root", default="task 3/artifacts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="auto", help="Device: 'auto' (GPU if available), 'cuda', 'cpu'")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--use-class-weights", action="store_true", default=True)
    p.add_argument("--sweep-batch-sizes", default="8,16", help="Comma-separated batch sizes for hyperparameter sweep")
    p.add_argument("--sweep-lrs", default="1e-4,2e-4,5e-4", help="Comma-separated learning rates for hyperparameter sweep")
    p.add_argument("--augment", action="store_true", default=True, help="Enable training data augmentation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifacts = os.path.abspath(args.artifacts_root)
    plots_dir = os.path.join(artifacts, "plots")
    videos_dir = os.path.join(artifacts, "videos")
    images_dir = os.path.join(artifacts, "images")
    logs_dir = os.path.join(artifacts, "logs")
    ckpt_dir = os.path.join(artifacts, "checkpoints")
    for d in [plots_dir, videos_dir, images_dir, logs_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    train_jsonl = os.path.join(args.data_root, "train", "samples.jsonl")
    val_jsonl = os.path.join(args.data_root, "val", "samples.jsonl")
    val_episodes = os.path.join(args.data_root, "val", "episodes.json")
    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val")

    processor = AutoProcessor.from_pretrained(args.clip_model)

    # Auto-detect GPU if requested but not explicitly specified.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cuda_available = torch.cuda.is_available()
    print(f"Task3 device={device} | CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_ds = JsonlVlnDataset(train_jsonl, image_root=train_root, augment=args.augment)
    val_ds = JsonlVlnDataset(val_jsonl, image_root=val_root, augment=False)
    class_weights = _compute_class_weights(train_ds) if args.use_class_weights else torch.ones(len(ACTION_NAMES), dtype=torch.float32)

    # Hyperparameter sweep (lr, batch_size)
    sweep_lrs = [float(item.strip()) for item in args.sweep_lrs.split(",") if item.strip()]
    sweep_batch_sizes = [int(item.strip()) for item in args.sweep_batch_sizes.split(",") if item.strip()]
    search_space = [(lr, batch_size) for lr in sweep_lrs for batch_size in sweep_batch_sizes]

    results = []

    for lr, batch_size in search_space:
        run_name = f"lr_{lr}_bs_{batch_size}".replace(".", "p")
        run_dir = os.path.join(ckpt_dir, run_name)

        collate = build_collate_fn(processor, max_text_length=64)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        )

        model = CLIPCrossAttentionPolicy(clip_model_name=args.clip_model)
        model.freeze_clip(unfreeze_last_text_layers=0, unfreeze_last_vision_layers=0)
        model.to(device)

        run_metrics = train_one_run(
            run_name,
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            lr,
            run_dir,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
        )
        run_metrics["lr"] = lr
        run_metrics["batch_size"] = batch_size

        # confusion matrix for each run final/best
        best_ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
        model.load_state_dict(best_ckpt["model_state"], strict=True)
        _, _, cm = eval_loader(model, val_loader, device, class_weights=class_weights.to(device), label_smoothing=args.label_smoothing)
        plot_confusion(cm, os.path.join(plots_dir, f"confusion_{run_name}.png"), f"Confusion Matrix {run_name}")

        results.append(run_metrics)

    # Save hyperparameter table
    results_sorted = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    with open(os.path.join(logs_dir, "hyperparam_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, indent=2)

    with open(os.path.join(logs_dir, "hyperparam_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "lr", "batch_size", "best_val_acc", "final_val_acc", "final_val_loss"])
        writer.writeheader()
        for row in results_sorted:
            writer.writerow(row)

    # Plot hyperparameter comparison
    labels = [r["run_name"] for r in results_sorted]
    vals = [r["best_val_acc"] for r in results_sorted]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("best_val_acc")
    plt.title("Hyperparameter Sweep Results")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hyperparam_comparison.png"), dpi=150)
    plt.close()

    # Use best run for SR/SPL rollout evaluation
    best = results_sorted[0]
    best_run_name = best["run_name"]
    best_ckpt_path = os.path.join(ckpt_dir, best_run_name, "best.pt")

    model = CLIPCrossAttentionPolicy(clip_model_name=args.clip_model)
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    sr, spl, osr, ne, ep_records = rollout_sr_spl(
        model=model,
        processor=processor,
        episodes_path=val_episodes,
        image_root=val_root,
        device=device,
        videos_dir=videos_dir,
        num_videos=10,
    )

    task3_metrics = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "N/A",
        "best_run": best,
        "SR": sr,
        "SPL": spl,
        "OSR": osr,
        "NE": ne,
        "num_eval_episodes": len(ep_records),
        "cuda_requested": True,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    with open(os.path.join(logs_dir, "task3_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(task3_metrics, f, indent=2)

    with open(os.path.join(logs_dir, "rollout_episode_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "success",
                "oracle_success",
                "spl",
                "navigation_error",
                "shortest_path_length",
                "path_length",
                "num_pred_steps",
            ],
        )
        writer.writeheader()
        for row in ep_records:
            writer.writerow(row)

    # Visualize SR/SPL summary image
    plt.figure(figsize=(5, 4))
    plt.bar(["SR", "SPL"], [sr, spl])
    plt.ylim(0, 1.0)
    plt.title("Task 3 Rollout Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "sr_spl_summary.png"), dpi=150)
    plt.close()

    print(json.dumps(task3_metrics, indent=2))


if __name__ == "__main__":
    main()
