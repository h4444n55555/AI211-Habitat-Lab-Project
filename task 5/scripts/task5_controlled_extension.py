#!/usr/bin/env python3
"""Task 5 controlled extension: compare baseline VLN policy with a lightweight variant.

The extension keeps the same frozen-CLIP training setup used in Task 3 and changes only
one small design choice: a narrower fusion block with slightly stronger dropout.
This makes the comparison controlled and easy to interpret.
"""

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
    autocast = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from task2_vln.data.dataset import JsonlVlnDataset, build_collate_fn  # noqa: E402
from task2_vln.models.vln_policy import CLIPCrossAttentionPolicy  # noqa: E402

ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]


@dataclass
class ExperimentResult:
    name: str
    fusion_dim: int
    num_heads: int
    dropout: float
    total_params: int
    trainable_params: int
    best_val_acc: float
    final_val_acc: float
    final_val_loss: float
    mean_epoch_time_sec: float
    history: Dict[str, List[float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: JsonlVlnDataset) -> torch.Tensor:
    counts = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    for sample in dataset.samples:
        counts[int(sample.action)] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    label_smoothing: float,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["actions"].to(device)

        logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["logits"]
        loss = F.cross_entropy(logits, labels, weight=class_weights.to(device), label_smoothing=label_smoothing)

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.numel()
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_run(
    name: str,
    fusion_dim: int,
    num_heads: int,
    dropout: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    clip_model: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: str,
    class_weights: torch.Tensor,
    label_smoothing: float,
    num_workers: int,
) -> ExperimentResult:
    os.makedirs(output_dir, exist_ok=True)

    model = CLIPCrossAttentionPolicy(
        clip_model_name=clip_model,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    model.freeze_clip(unfreeze_last_text_layers=0, unfreeze_last_vision_layers=0)
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=max(lr * 0.05, 1e-6))
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if GradScaler is not None else None

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
    }
    best_val_acc = -1.0
    epochs_without_improve = 0
    patience = 3

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        sample_count = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["actions"].to(device)

            optimizer.zero_grad(set_to_none=True)
            amp_context = autocast(enabled=use_amp) if autocast is not None else nullcontext()
            with amp_context:
                logits = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )["logits"]
                loss = F.cross_entropy(logits, labels, weight=class_weights.to(device), label_smoothing=label_smoothing)

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

            train_loss_sum += loss.item() * labels.size(0)
            sample_count += labels.numel()

        train_loss = train_loss_sum / max(sample_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, class_weights, label_smoothing)
        scheduler.step()
        epoch_time = time.perf_counter() - start_time

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["epoch_time_sec"].append(float(epoch_time))

        print(
            f"[{name}] epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "clip_model": clip_model,
            "fusion_dim": fusion_dim,
            "num_heads": num_heads,
            "dropout": dropout,
            "action_names": ACTION_NAMES,
        }
        torch.save(ckpt, os.path.join(output_dir, "last.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(output_dir, "best.pt"))
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"[{name}] early stopping at epoch {epoch}")
                break

    result = ExperimentResult(
        name=name,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        dropout=dropout,
        total_params=total_params,
        trainable_params=trainable_params,
        best_val_acc=float(best_val_acc),
        final_val_acc=float(history["val_acc"][-1]),
        final_val_loss=float(history["val_loss"][-1]),
        mean_epoch_time_sec=float(np.mean(history["epoch_time_sec"])),
        history=history,
    )

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def plot_comparison(results: List[ExperimentResult], output_dir: str) -> None:
    labels = [result.name for result in results]
    val_acc = [result.best_val_acc for result in results]
    val_loss = [result.final_val_loss for result in results]
    params = [result.trainable_params for result in results]
    times = [result.mean_epoch_time_sec for result in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].bar(labels, val_acc, color=["#2a9d8f", "#e76f51"])
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_title("Best Validation Accuracy")
    axes[0, 0].set_ylabel("accuracy")

    axes[0, 1].bar(labels, val_loss, color=["#457b9d", "#f4a261"])
    axes[0, 1].set_title("Final Validation Loss")
    axes[0, 1].set_ylabel("loss")

    axes[1, 0].bar(labels, params, color=["#264653", "#8d99ae"])
    axes[1, 0].set_title("Trainable Parameters")
    axes[1, 0].set_ylabel("parameters")

    axes[1, 1].bar(labels, times, color=["#1d3557", "#e9c46a"])
    axes[1, 1].set_title("Mean Epoch Time")
    axes[1, 1].set_ylabel("seconds")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Task 5 Controlled Extension: Baseline vs Lightweight VLN Policy", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "task5_comparison.png"), dpi=300)
    plt.close(fig)


def write_summary(results: List[ExperimentResult], output_dir: str) -> None:
    payload = [asdict(result) for result in results]
    with open(os.path.join(output_dir, "task5_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    baseline = results[0]
    improved = results[1]
    param_reduction = 100.0 * (1.0 - improved.trainable_params / max(baseline.trainable_params, 1))
    acc_delta = 100.0 * (improved.best_val_acc - baseline.best_val_acc)
    speedup = baseline.mean_epoch_time_sec / max(improved.mean_epoch_time_sec, 1e-9)

    lines = [
        "# Task 5 Controlled Extension",
        "",
        "## Baseline vs Lightweight Variant",
        "",
        f"- Baseline: fusion_dim={baseline.fusion_dim}, num_heads={baseline.num_heads}, dropout={baseline.dropout}",
        f"- Lightweight: fusion_dim={improved.fusion_dim}, num_heads={improved.num_heads}, dropout={improved.dropout}",
        f"- Trainable parameter reduction: {param_reduction:.2f}%",
        f"- Validation accuracy change: {acc_delta:+.2f} percentage points",
        f"- Mean epoch time speedup: {speedup:.2f}x",
        "",
        "## Interpretation",
        "",
        "The lightweight variant keeps the frozen-CLIP training regime intact while shrinking the",
        "fusion block and increasing dropout slightly. This tests whether the model is over-parameterized",
        "for the synthetic VLN task and whether a cheaper policy can preserve performance.",
    ]
    with open(os.path.join(output_dir, "task5_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5 controlled extension experiment")
    parser.add_argument("--data-root", default="task 3/data_large")
    parser.add_argument("--output-dir", default="task 5/artifacts")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(args.clip_model)
    train_jsonl = os.path.join(args.data_root, "train", "samples.jsonl")
    val_jsonl = os.path.join(args.data_root, "val", "samples.jsonl")
    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val")

    train_ds = JsonlVlnDataset(train_jsonl, image_root=train_root, augment=True)
    val_ds = JsonlVlnDataset(val_jsonl, image_root=val_root, augment=False)
    class_weights = compute_class_weights(train_ds)
    collate_fn = build_collate_fn(processor, max_text_length=64)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    baseline = train_run(
        name="baseline",
        fusion_dim=512,
        num_heads=8,
        dropout=0.1,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        clip_model=args.clip_model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=os.path.join(args.output_dir, "baseline"),
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
    )

    lightweight = train_run(
        name="lightweight",
        fusion_dim=256,
        num_heads=4,
        dropout=0.2,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        clip_model=args.clip_model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=os.path.join(args.output_dir, "lightweight"),
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
    )

    results = [baseline, lightweight]
    write_summary(results, args.output_dir)
    plot_comparison(results, args.output_dir)

    baseline_params = baseline.trainable_params
    improved_params = lightweight.trainable_params
    param_reduction = 100.0 * (1.0 - improved_params / max(baseline_params, 1))

    print(json.dumps({
        "baseline": asdict(baseline),
        "lightweight": asdict(lightweight),
        "trainable_param_reduction_pct": param_reduction,
    }, indent=2))


if __name__ == "__main__":
    main()
