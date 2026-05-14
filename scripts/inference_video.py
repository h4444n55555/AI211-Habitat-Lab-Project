#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoProcessor

# Habitat-sim requires TMPDIR to be set before import if /tmp isn't writable
os.environ["TMPDIR"] = os.path.join(os.path.expanduser("~"), "habitat", ".tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
import habitat_sim

# Add task2_vln to path to import models
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "task2_vln"))

from models.vln_policy import CLIPCrossAttentionPolicy

ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]

def make_scene_sim(scene_path: str):
    from habitat_sim.utils.settings import default_sim_settings, make_cfg
    settings = default_sim_settings.copy()
    settings.update({
        "scene": scene_path,
        "width": 512,
        "height": 512,
        "color_sensor": True,
        "depth_sensor": False,
        "semantic_sensor": False,
        "sensor_height": 1.5,
        "seed": 42,
        "silent": True,
        "enable_physics": False,
        "gpu_device_id": -1,
    })
    cfg = make_cfg(settings)
    cfg.sim_cfg.gpu_device_id = -1
    sim = habitat_sim.Simulator(cfg)
    return sim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = os.path.join(REPO_DIR, "RUN 1")
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(REPO_DIR, "results", "task2", "best.pt")
    scene_path = os.path.join(REPO_DIR, "habitat-sim/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb")
    # Start position from the validation dataset to ensure navigability
    start_position = [-8.66596794128418, 0.07244700193405151, 1.3662927150726318]
    instruction = "move forward 2 steps, then turn right, then move forward, then turn left, then move forward, then turn right, then move forward 4 steps, then turn right 3 times, then move forward 6 steps, then turn left, then move forward, then stop at the goal."
    max_steps = 30
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    clip_model_name = ckpt.get("clip_model", "openai/clip-vit-base-patch16")
    
    print("Building model...")
    model = CLIPCrossAttentionPolicy(clip_model_name=clip_model_name)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    processor = AutoProcessor.from_pretrained(clip_model_name)
    
    print("Initializing Simulator...")
    sim = make_scene_sim(scene_path)
    agent = sim.initialize_agent(0)
    
    from habitat_sim import AgentState
    start_state = AgentState()
    start_state.position = np.array(start_position, dtype=np.float32)
    agent.set_state(start_state)
    
    frames = []
    probs_history = {act: [] for act in ACTION_NAMES}
    
    print(f"\nInstruction: '{instruction}'")
    print("Starting rollout...\n")
    
    for step in range(max_steps):
        obs = sim.get_sensor_observations()
        img_arr = obs["color_sensor"]
        image = Image.fromarray(img_arr, mode="RGBA").convert("RGB")
        
        inputs = processor(
            text=[instruction],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        
        with torch.no_grad():
            output = model(
                pixel_values=inputs["pixel_values"].to(device),
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device)
            )
            logits = output["logits"]
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
            
        pred_idx = int(np.argmax(probs))
        pred_action = ACTION_NAMES[pred_idx]
        
        for i, act in enumerate(ACTION_NAMES):
            probs_history[act].append(probs[i])
            
        print(f"Step {step+1:02d} | Action: {pred_action:15s} | Confidence: {probs[pred_idx]*100:5.1f}%")
        
        # Draw on image
        frame = image.copy()
        d = ImageDraw.Draw(frame)
        d.rectangle([5, 5, 300, 60], fill=(0, 0, 0))
        d.text((10, 10), f"Instruction: {instruction}", fill=(255, 255, 255))
        d.text((10, 25), f"Step: {step+1}", fill=(255, 255, 255))
        d.text((10, 40), f"Action: {pred_action} ({probs[pred_idx]*100:.1f}%)", fill=(0, 255, 0))
        frames.append(np.array(frame))
        
        if pred_action == "stop":
            if step < 3:
                print("  Model predicted STOP, but ignoring for the first 3 steps to ensure movement.")
                # Fallback to the second most likely action
                probs[pred_idx] = 0.0
                pred_idx = int(np.argmax(probs))
                pred_action = ACTION_NAMES[pred_idx]
                print(f"  Fallback Action: {pred_action:15s} | Confidence: {probs[pred_idx]*100:5.1f}%")
            else:
                print("\nModel predicted STOP. Ending rollout.")
                break
            
        sim.step(pred_action)

    # Save Video
    print("\nGenerating video and graphs...")
    video_path = os.path.join(output_dir, "rollout_video.mp4")
    imageio.mimwrite(video_path, frames, fps=3, macro_block_size=None)
    
    # Plot Probabilities
    plt.figure(figsize=(10, 6))
    for act in ACTION_NAMES:
        plt.plot(range(1, len(probs_history[act])+1), probs_history[act], marker='o', label=act, linewidth=2)
    plt.title(f"Action Probabilities over Time\nInstruction: '{instruction}'", fontsize=14, fontweight="bold")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim([0, 1.05])
    plt.xticks(range(1, len(probs_history["stop"])+1))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    graph_path = os.path.join(output_dir, "probabilities_graph.png")
    plt.savefig(graph_path, dpi=200)
    plt.close()
    
    sim.close()
    print("\n" + "="*40)
    print("✅ RUN 1 COMPLETE")
    print(f"Video: {video_path}")
    print(f"Graph: {graph_path}")
    print("="*40)

if __name__ == "__main__":
    main()
