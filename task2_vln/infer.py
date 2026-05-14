import argparse
import json

from PIL import Image
import torch
from transformers import AutoProcessor

from models.vln_policy import CLIPCrossAttentionPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run single-sample inference for Task 2 policy")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--image", required=True, help="Path to RGB image")
    p.add_argument("--instruction", required=True, help="Navigation instruction text")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-text-length", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    clip_model = ckpt.get("clip_model", "openai/clip-vit-base-patch16")
    action_names = ckpt.get("action_names", ["move_forward", "turn_left", "turn_right", "stop"])

    processor = AutoProcessor.from_pretrained(clip_model)
    model = CLIPCrossAttentionPolicy(clip_model_name=clip_model)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    enc = processor(
        text=[args.instruction],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_text_length,
    )

    pixel_values = enc["pixel_values"].to(device)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["logits"]
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())

    output = {
        "image": args.image,
        "instruction": args.instruction,
        "predicted_action_id": pred_idx,
        "predicted_action": action_names[pred_idx],
        "action_probabilities": {
            action_names[i]: float(probs[i].item()) for i in range(len(action_names))
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
