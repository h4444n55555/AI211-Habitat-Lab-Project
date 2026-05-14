import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from data.dataset import JsonlVlnDataset, build_collate_fn
from models.vln_policy import CLIPCrossAttentionPolicy


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["actions"].to(device)

        logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)["logits"]
        preds = torch.argmax(logits, dim=-1)

        total += labels.numel()
        correct += (preds == labels).sum().item()

        for y_true, y_pred in zip(labels.tolist(), preds.tolist()):
            per_class[y_true]["total"] += 1
            if y_true == y_pred:
                per_class[y_true]["correct"] += 1

    acc = correct / max(total, 1)
    per_class_acc = {
        str(k): (v["correct"] / max(v["total"], 1))
        for k, v in sorted(per_class.items(), key=lambda x: x[0])
    }
    return {
        "accuracy": acc,
        "per_class_accuracy": per_class_acc,
        "total_samples": total,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Task 2 VLN policy checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--image-root", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-text-length", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default="eval_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    clip_model = ckpt.get("clip_model", "openai/clip-vit-base-patch16")

    processor = AutoProcessor.from_pretrained(clip_model)
    ds = JsonlVlnDataset(args.eval_jsonl, image_root=args.image_root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=build_collate_fn(processor, max_text_length=args.max_text_length),
    )

    model = CLIPCrossAttentionPolicy(clip_model_name=clip_model)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    metrics = evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
