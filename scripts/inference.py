#!/usr/bin/env python3
"""
Inference script for Vision-Language Navigation Model.
Usage:
    python scripts/inference.py --image path/to/image.jpg --instruction "Go to the door" --checkpoint results/task2/best.pt
"""
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor
import sys
import os

# Add task2_vln to path to import models
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "task2_vln"))

from models.vln_policy import CLIPCrossAttentionPolicy

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained VLN model")
    parser.add_argument("--image", required=True, help="Path to the input image (RGB)")
    parser.add_argument("--instruction", required=True, help="Natural language instruction")
    parser.add_argument("--checkpoint", default="results/task2/best.pt", help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--device", default="auto", help="Device to run inference on (auto, cuda, cpu)")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    clip_model_name = ckpt.get("clip_model", "openai/clip-vit-base-patch16")
    action_names = ckpt.get("action_names", ["move_forward", "turn_left", "turn_right", "stop"])

    # Load model
    print("Building model...")
    model = CLIPCrossAttentionPolicy(clip_model_name=clip_model_name)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    # Load processor
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    processor = AutoProcessor.from_pretrained(clip_model_name)

    # Load image
    print(f"Loading image: {args.image}")
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    print(f"Instruction: '{args.instruction}'")

    # Process inputs
    inputs = processor(
        text=[args.instruction],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )

    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        logits = output["logits"]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Get predicted action
    pred_idx = torch.argmax(probs, dim=-1).item()
    pred_action = action_names[pred_idx]
    confidence = probs[0, pred_idx].item()

    print("\n" + "="*40)
    print("🎯 INFERENCE RESULTS")
    print("="*40)
    print(f"Instruction : {args.instruction}")
    print(f"Image       : {os.path.basename(args.image)}")
    print(f"Action      : {pred_action}")
    print(f"Confidence  : {confidence*100:.1f}%")
    print("="*40)
    
    print("\nAll action probabilities:")
    for i, action in enumerate(action_names):
        print(f"  {action:15s} : {probs[0, i].item()*100:.1f}%")

if __name__ == "__main__":
    main()
