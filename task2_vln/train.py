#!/usr/bin/env python3
"""Task 2: Baseline VLN Training with CLIP Cross-Attention Policy.

Trains a vision-language navigation policy using imitation learning on
RGB observations paired with natural-language instructions.
"""

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
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
except Exception:
    GradScaler = None
    autocast = nullcontext

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from data.dataset import JsonlVlnDataset, build_collate_fn
from models.vln_policy import CLIPCrossAttentionPolicy

ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
def evaluate(model, loader, device, class_weights=None, label_smoothing=0.0):
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
        loss = F.cross_entropy(
            logits, y,
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=label_smoothing,
        )
        preds = torch.argmax(logits, dim=-1)
        total_loss += loss.item() * y.numel()
        total += y.numel()
        correct += (preds == y).sum().item()
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1

    return total_loss / max(total, 1), correct / max(total, 1), cm


def plot_learning_curves(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history["epoch"]
    ax1.plot(epochs, history["train_loss"], "b-o", lw=2, ms=5, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-s", lw=2, ms=5, label="Val Loss")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_acc"], "g-^", lw=2, ms=6, label="Val Accuracy")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy", fontweight="bold")
    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=200)
    plt.close()


def plot_confusion(cm, output_dir):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Action Prediction Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted", fontweight="bold")
    plt.ylabel("True", fontweight="bold")
    plt.xticks(range(4), ACTION_NAMES, rotation=25, ha="right")
    plt.yticks(range(4), ACTION_NAMES)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontweight="bold")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


def train(model, train_loader, val_loader, device, epochs, lr, output_dir,
          class_weights, label_smoothing=0.1):
    os.makedirs(output_dir, exist_ok=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=max(lr * 0.05, 1e-6))
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if GradScaler is not None else None

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc = -1.0
    patience = 5
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0

        for batch in train_loader:
            x_img = batch["pixel_values"].to(device)
            x_ids = batch["input_ids"].to(device)
            x_mask = batch["attention_mask"].to(device)
            y = batch["actions"].to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = autocast(enabled=use_amp) if autocast is not nullcontext else nullcontext()
            with amp_ctx:
                logits = model(pixel_values=x_img, input_ids=x_ids, attention_mask=x_mask)["logits"]
                loss = F.cross_entropy(logits, y, weight=class_weights.to(device),
                                       label_smoothing=label_smoothing)
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
            loss_sum += loss.item() * y.numel()
            n += y.numel()

        train_loss = loss_sum / max(n, 1)
        val_loss, val_acc, _ = evaluate(model, val_loader, device,
                                         class_weights=class_weights, label_smoothing=label_smoothing)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
              f"lr={cur_lr:.2e} | {time.time()-t0:.1f}s")

        ckpt = {"model_state": model.state_dict(), "clip_model": "openai/clip-vit-base-patch16",
                "action_names": ACTION_NAMES, "epoch": epoch}
        torch.save(ckpt, os.path.join(output_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(output_dir, "best.pt"))
            no_improve = 0
            print(f"  ✓ New best: {best_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    return history, best_acc


def parse_args():
    p = argparse.ArgumentParser(description="Task 2: Baseline VLN Training")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--output-dir", default="results/task2")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--augment", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Task 2: Baseline VLN Training")
    print("=" * 60)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_root = os.path.dirname(os.path.abspath(args.train_jsonl))
    val_root = os.path.dirname(os.path.abspath(args.val_jsonl))
    train_ds = JsonlVlnDataset(args.train_jsonl, image_root=train_root, augment=args.augment)
    val_ds = JsonlVlnDataset(args.val_jsonl, image_root=val_root, augment=False)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    class_weights = compute_class_weights(train_ds)
    processor = AutoProcessor.from_pretrained(args.clip_model)
    collate = build_collate_fn(processor, max_text_length=64)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              persistent_workers=args.num_workers > 0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                            persistent_workers=args.num_workers > 0, collate_fn=collate)

    model = CLIPCrossAttentionPolicy(clip_model_name=args.clip_model)
    model.freeze_clip(unfreeze_last_text_layers=0, unfreeze_last_vision_layers=0)
    model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_p:,} total, {train_p:,} trainable")

    t_start = time.time()
    history, best_acc = train(model, train_loader, val_loader, device,
                              args.epochs, args.lr, args.output_dir,
                              class_weights, args.label_smoothing)
    t_total = time.time() - t_start

    # Final eval
    best_ckpt = torch.load(os.path.join(args.output_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    model.to(device)
    val_loss, val_acc, cm = evaluate(model, val_loader, device, class_weights=class_weights)
    per_class = {ACTION_NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(4)}

    plot_learning_curves(history, args.output_dir)
    plot_confusion(cm, args.output_dir)

    metrics = {
        "history": history, "best_val_acc": float(best_acc),
        "final_val_loss": float(val_loss), "final_val_acc": float(val_acc),
        "per_class_accuracy": per_class, "training_time_sec": t_total,
        "total_params": total_p, "trainable_params": train_p,
        "config": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                   "label_smoothing": args.label_smoothing, "clip_model": args.clip_model},
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Task 2 complete! Best val_acc={best_acc:.4f} in {t_total/60:.1f} min")


if __name__ == "__main__":
    main()
