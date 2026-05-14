import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.nav_datasets import PointNavDataset, pointnav_collate_fn, ObjectNavDataset, objectnav_collate_fn
from models.pointnav_policy import PointNavPolicy
from models.objectnav_policy import ObjectNavPolicy

def train_epoch(model, dataloader, optimizer, device, task_type):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        actions = batch["actions"].to(device)
        
        optimizer.zero_grad()
        
        if task_type == "pointnav":
            point_goal = batch["point_goal"].to(device)
            out = model(pixel_values, point_goal)
        elif task_type == "objectnav":
            object_id = batch["object_id"].to(device)
            out = model(pixel_values, object_id)
            
        logits = out["logits"]
        loss = F.cross_entropy(logits, actions)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * actions.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == actions).sum().item()
        total += actions.size(0)
        
    return total_loss / total, correct / total


def plot_metrics(history, task_name, out_dir):
    """Generate loss and accuracy curves."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{task_name.upper()} Training Curves", fontsize=16, fontweight="bold")

    # ── Loss ──
    ax1.plot(epochs, history["loss"], "o-", color="#e74c3c", linewidth=2, markersize=6, label="Training Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Cross-Entropy Loss", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # ── Accuracy ──
    acc_pct = [a * 100 for a in history["accuracy"]]
    ax2.plot(epochs, acc_pct, "o-", color="#2ecc71", linewidth=2, markersize=6, label="Training Accuracy")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Action Prediction Accuracy", fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(out_dir, f"{task_name}_training_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["pointnav", "objectnav"], required=True)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default="training_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = "openai/clip-vit-base-patch16"
    processor = AutoProcessor.from_pretrained(clip_model)
    
    if args.task == "pointnav":
        ds = PointNavDataset(args.train_jsonl, image_root=args.image_root)
        collate_fn = pointnav_collate_fn(processor)
        model = PointNavPolicy(clip_model_name=clip_model)
    else:
        ds = ObjectNavDataset(args.train_jsonl, image_root=args.image_root)
        collate_fn = objectnav_collate_fn(processor)
        model = ObjectNavPolicy(clip_model_name=clip_model)

    # Freeze vision backbone for faster training
    model.freeze_vision()
    model.to(device)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"loss": [], "accuracy": []}

    print(f"Training {args.task.upper()} policy...")
    for epoch in range(args.epochs):
        loss, acc = train_epoch(model, loader, optimizer, device, args.task)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}")

    # Save model weights
    save_path = f"{args.task}_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model weights to {save_path}")

    # Save history JSON
    os.makedirs(args.out_dir, exist_ok=True)
    hist_path = os.path.join(args.out_dir, f"{args.task}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history → {hist_path}")

    # Generate plots
    plot_metrics(history, args.task, args.out_dir)


if __name__ == "__main__":
    main()
