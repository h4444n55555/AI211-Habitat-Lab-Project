#!/usr/bin/env python
"""
Task 4: Ablation Study
Compare performance with different encoder configurations:
- Frozen vision encoder
- Frozen text encoder
- Frozen both encoders
- Fine-tune all (baseline)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import AutoProcessor

# Setup paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(PKG_ROOT)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from task2_vln
sys.path.insert(0, os.path.join(REPO_ROOT, 'task2_vln'))
from data.dataset import JsonlVlnDataset, build_collate_fn
from models.vln_policy import CLIPCrossAttentionPolicy


ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]


def _compute_class_weights(dataset: JsonlVlnDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle action imbalance."""
    counts = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    for sample in dataset.samples:
        counts[int(sample.action)] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def eval_loader(
    model,
    loader,
    device,
    class_weights: torch.Tensor = None,
    label_smoothing: float = 0.0,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

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

    return total_loss / max(total, 1), correct / max(total, 1)


def freeze_vision_encoder(model):
    """Freeze CLIP vision encoder only, train text encoder + fusion + head."""
    for param in model.clip.vision_model.parameters():
        param.requires_grad = False
    return model


def freeze_text_encoder(model):
    """Freeze CLIP text encoder only, train vision encoder + fusion + head."""
    for param in model.clip.text_model.parameters():
        param.requires_grad = False
    return model


def freeze_both_encoders(model):
    """Freeze both CLIP encoders, only train fusion & policy head."""
    for param in model.clip.parameters():
        param.requires_grad = False
    return model


def train_run(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    class_weights: torch.Tensor = None,
    label_smoothing: float = 0.1,
) -> Dict:
    """Train model and return history."""
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=max(lr * 0.05, 1e-6))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_acc = -1.0
    patience = 4
    epochs_without_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * y.numel()
            train_count += y.numel()

        train_loss = train_loss_sum / max(train_count, 1)
        scheduler.step()

        # Eval
        val_loss, val_acc = eval_loader(model, val_loader, device, class_weights=class_weights, label_smoothing=label_smoothing)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return history


def run_ablation_study(args):
    """Run full ablation study with different configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(args.clip_model)

    results = {}

    # Load datasets
    print("Loading datasets...")
    train_dataset = JsonlVlnDataset(args.train_data, augment=True)
    val_dataset = JsonlVlnDataset(args.val_data, augment=False)
    class_weights = _compute_class_weights(train_dataset)
    print(f"Class weights: {dict(zip(ACTION_NAMES, class_weights.tolist()))}")

    collate_fn = build_collate_fn(processor, max_text_length=64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Ablation configurations
    configs = {
        "baseline": {"freeze_fn": None, "desc": "Fine-tune all (baseline)"},
        "frozen_vision": {"freeze_fn": freeze_vision_encoder, "desc": "Frozen vision encoder"},
        "frozen_text": {"freeze_fn": freeze_text_encoder, "desc": "Frozen text encoder"},
        "frozen_both": {"freeze_fn": freeze_both_encoders, "desc": "Frozen both encoders"},
    }

    print("\nRunning ablation study...")
    for config_name, config_info in configs.items():
        print(f"\n[{config_name}] {config_info['desc']}")

        # Create fresh model for each config
        model = CLIPCrossAttentionPolicy(clip_model_name=args.clip_model)

        # Apply freezing
        if config_info["freeze_fn"]:
            model = config_info["freeze_fn"](model)

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,}")

        # Train
        history = train_run(
            model, train_loader, val_loader, device, args.epochs, args.lr,
            class_weights=class_weights, label_smoothing=args.label_smoothing,
        )

        results[config_name] = {
            "config": config_info["desc"],
            "trainable_params": trainable,
            "total_params": total,
            "history": history,
            "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0.0,
            "final_val_acc": history["val_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="task 3/data_large/train/samples.jsonl")
    parser.add_argument("--val-data", default="task 3/data_large/val/samples.jsonl")
    parser.add_argument("--out-dir", default="task 4/artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Run ablation
    results = run_ablation_study(args)

    # Save results
    results_path = os.path.join(args.out_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        # Convert numpy arrays and tensors to serializable format
        results_serializable = {}
        for config, data in results.items():
            results_serializable[config] = {
                "config": data["config"],
                "trainable_params": int(data["trainable_params"]),
                "total_params": int(data["total_params"]),
                "history": {
                    "train_loss": [float(x) for x in data["history"]["train_loss"]],
                    "val_loss": [float(x) for x in data["history"]["val_loss"]],
                    "val_acc": [float(x) for x in data["history"]["val_acc"]],
                },
                "best_val_acc": float(data["best_val_acc"]),
                "final_val_acc": float(data["final_val_acc"]),
                "final_val_loss": float(data["final_val_loss"]),
            }
        json.dump(results_serializable, f, indent=2)

    print(f"\n✓ Ablation results saved to: {results_path}")

    # Generate ablation comparison plot
    plot_ablation_results(results, args.out_dir)


def plot_ablation_results(results: Dict, out_dir: str):
    """Generate ablation study comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Final Validation Accuracy
    configs = list(results.keys())
    best_accs = [results[c]["best_val_acc"] for c in configs]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    ax = axes[0]
    bars = ax.bar(configs, best_accs, color=colors, edgecolor="black", linewidth=1.5, alpha=0.8)
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{acc:.3f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Best Validation Accuracy", fontweight="bold")
    ax.set_title("Ablation Study: Best Accuracy", fontweight="bold", fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Training Curves
    ax = axes[1]
    for config, color in zip(configs, colors):
        val_accs = results[config]["history"]["val_acc"]
        epochs = list(range(1, len(val_accs) + 1))
        ax.plot(epochs, val_accs, marker="o", linewidth=2.5, markersize=7,
                label=config, color=color, alpha=0.8)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Validation Accuracy", fontweight="bold")
    ax.set_title("Ablation Study: Training Progress", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 3: Trainable parameter count
    ax = axes[2]
    param_counts = [results[c]["trainable_params"] / 1e6 for c in configs]
    bars = ax.bar(configs, param_counts, color=colors, edgecolor="black", linewidth=1.5, alpha=0.8)
    for bar, p in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{p:.1f}M",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Trainable Params (M)", fontweight="bold")
    ax.set_title("Trainable Parameters", fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "ablation_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Ablation plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
