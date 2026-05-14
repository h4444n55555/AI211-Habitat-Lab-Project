#!/usr/bin/env python
"""
Task 4: Generate environment visualization and navigation videos.
Creates mp4 files showing robot trajectories in environments.
"""

import argparse
import json
import os
from typing import List, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def create_environment_grid(width: int = 512, height: int = 512, grid_size: int = 8) -> Image.Image:
    """Create a simple grid-based environment visualization."""
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    cell_w = width // grid_size
    cell_h = height // grid_size

    # Draw grid
    for i in range(grid_size + 1):
        # Vertical lines
        draw.line([(i * cell_w, 0), (i * cell_w, height)], fill="lightgray", width=1)
        # Horizontal lines
        draw.line([(0, i * cell_h), (width, i * cell_h)], fill="lightgray", width=1)

    # Draw some obstacles
    np.random.seed(42)
    for _ in range(5):
        x = np.random.randint(1, grid_size - 1)
        y = np.random.randint(1, grid_size - 1)
        x1, y1 = x * cell_w + 5, y * cell_h + 5
        x2, y2 = (x + 1) * cell_w - 5, (y + 1) * cell_h - 5
        draw.rectangle([x1, y1, x2, y2], fill="gray")

    return img


def draw_robot(img: Image.Image, pos_x: float, pos_y: float, robot_size: int = 20) -> Image.Image:
    """Draw robot at given position."""
    draw = ImageDraw.Draw(img)
    x1 = int(pos_x - robot_size / 2)
    y1 = int(pos_y - robot_size / 2)
    x2 = int(pos_x + robot_size / 2)
    y2 = int(pos_y + robot_size / 2)
    draw.ellipse([x1, y1, x2, y2], fill="red", outline="darkred", width=2)
    return img


def draw_trajectory(img: Image.Image, positions: List[tuple], color: str = "blue", width: int = 2) -> Image.Image:
    """Draw trajectory path."""
    draw = ImageDraw.Draw(img)
    for i in range(len(positions) - 1):
        draw.line([positions[i], positions[i + 1]], fill=color, width=width)
    return img


def generate_trajectory_frames(
    trajectory_actions: List[int],
    grid_size: int = 8,
    video_width: int = 512,
    video_height: int = 512,
) -> List[np.ndarray]:
    """Generate frames for a navigation trajectory."""
    ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]
    ACTION_SYMBOLS = ["↑", "←", "→", "◼"]

    frames = []
    cell_w = video_width // grid_size
    cell_h = video_height // grid_size

    # Starting position
    pos_x, pos_y = video_width // 2, video_height // 2
    direction = 0  # 0=up, 1=left, 2=right, 3=down
    positions = [(pos_x, pos_y)]

    # Add title frame
    img = create_environment_grid(video_width, video_height, grid_size)
    img = draw_robot(img, pos_x, pos_y)

    # Add instruction text
    draw = ImageDraw.Draw(img)
    instruction = "Navigation Trajectory Visualization"
    draw.text((10, 10), instruction, fill="black")

    frames.append(np.array(img))

    # Generate frames for each action
    for step, action_id in enumerate(trajectory_actions):
        if action_id == 3:  # stop
            # Hold final position
            img = create_environment_grid(video_width, video_height, grid_size)
            img = draw_trajectory(img, positions, color="blue")
            img = draw_robot(img, pos_x, pos_y, robot_size=20)

            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Step {step + 1}: {ACTION_NAMES[action_id]} (goal reached)", fill="black")

            frames.append(np.array(img))
        else:
            # Update position based on action
            if action_id == 0:  # move_forward
                if direction == 0:
                    pos_y -= cell_h
                elif direction == 1:
                    pos_x -= cell_w
                elif direction == 2:
                    pos_x += cell_w
                else:
                    pos_y += cell_h
            elif action_id == 1:  # turn_left
                direction = (direction + 1) % 4
            elif action_id == 2:  # turn_right
                direction = (direction - 1) % 4

            positions.append((pos_x, pos_y))

            # Create frame
            img = create_environment_grid(video_width, video_height, grid_size)
            img = draw_trajectory(img, positions, color="blue", width=3)
            img = draw_robot(img, pos_x, pos_y, robot_size=20)

            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Step {step + 1}: {ACTION_NAMES[action_id]}", fill="black")

            frames.append(np.array(img))

    return frames


def save_trajectory_video(frames: List[np.ndarray], output_path: str, fps: int = 2) -> None:
    """Save frames as mp4 video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create video writer (use h264 codec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert PIL RGB to OpenCV BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="task 4/data")
    parser.add_argument("--out-dir", default="task 4/artifacts/videos")
    parser.add_argument("--num-videos", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load sample episodes
    episodes_path = "task 3/data_large/val/episodes.json"
    if not os.path.exists(episodes_path):
        print(f"Episodes file not found: {episodes_path}")
        return

    with open(episodes_path, "r") as f:
        episodes = json.load(f)

    print(f"Generating {args.num_videos} navigation videos...")

    for ep_idx in range(min(args.num_videos, len(episodes))):
        ep = episodes[ep_idx]
        actions = ep.get("actions", [])

        if not actions:
            continue

        print(f"\n[Episode {ep_idx}] Actions: {actions}")

        # Generate frames
        frames = generate_trajectory_frames(actions, grid_size=8, video_width=640, video_height=640)

        # Save video
        video_path = os.path.join(args.out_dir, f"navigation_{ep_idx:03d}.mp4")
        save_trajectory_video(frames, video_path, fps=2)

    print(f"\n✓ Video generation complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
