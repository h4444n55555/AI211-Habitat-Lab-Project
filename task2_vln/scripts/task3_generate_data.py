import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw


ACTION_TO_ID = {
    "move_forward": 0,
    "turn_left": 1,
    "turn_right": 2,
    "stop": 3,
}


@dataclass
class EpisodeSpec:
    episode_id: str
    instruction: str
    actions: List[int]
    shortest_path_length: int


def _draw_frame(path: str, action_id: int, step_idx: int, total_steps: int, episode_id: str) -> None:
    size = (256, 256)
    color_map = {
        0: (60, 140, 240),
        1: (245, 166, 35),
        2: (214, 88, 88),
        3: (95, 184, 128),
    }
    img = Image.new("RGB", size, color=color_map[action_id])
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, 244, 244], outline=(255, 255, 255), width=3)
    d.text((18, 18), f"ep={episode_id}", fill=(255, 255, 255))
    d.text((18, 42), f"step={step_idx+1}/{total_steps}", fill=(255, 255, 255))
    d.text((18, 66), f"target_action={action_id}", fill=(255, 255, 255))

    # Draw a simple directional marker to encode action semantics.
    if action_id == 0:  # forward
        d.polygon([(128, 100), (112, 136), (144, 136)], fill=(255, 255, 255))
    elif action_id == 1:  # left
        d.polygon([(96, 128), (132, 112), (132, 144)], fill=(255, 255, 255))
    elif action_id == 2:  # right
        d.polygon([(160, 128), (124, 112), (124, 144)], fill=(255, 255, 255))
    else:  # stop
        d.rectangle([112, 112, 144, 144], fill=(255, 255, 255))

    img.save(path)


def _build_episode(ep_idx: int, split: str) -> EpisodeSpec:
    # Construct small navigation-like plans: optional turn -> several forwards -> stop.
    turn_choice = random.choice([None, "left", "right"])
    forward_steps = random.randint(2, 8)

    actions: List[int] = []
    instruction_parts: List[str] = []

    if turn_choice == "left":
        actions.append(ACTION_TO_ID["turn_left"])
        instruction_parts.append("Turn left")
    elif turn_choice == "right":
        actions.append(ACTION_TO_ID["turn_right"])
        instruction_parts.append("Turn right")

    actions.extend([ACTION_TO_ID["move_forward"]] * forward_steps)
    actions.append(ACTION_TO_ID["stop"])

    if instruction_parts:
        instruction = f"{instruction_parts[0]} and move forward {forward_steps} steps, then stop."
    else:
        instruction = f"Move forward {forward_steps} steps and then stop."

    shortest_path_length = len(actions) - 1  # excludes terminal STOP

    return EpisodeSpec(
        episode_id=f"{split}_ep_{ep_idx:05d}",
        instruction=instruction,
        actions=actions,
        shortest_path_length=shortest_path_length,
    )


def generate_dataset(out_dir: str, split: str, num_episodes: int) -> Tuple[str, str]:
    split_dir = os.path.join(out_dir, split)
    image_dir = os.path.join(split_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    jsonl_path = os.path.join(split_dir, "samples.jsonl")
    episodes_path = os.path.join(split_dir, "episodes.json")

    episodes_meta = []

    with open(jsonl_path, "w", encoding="utf-8") as samples_f:
        for ep_idx in range(num_episodes):
            ep = _build_episode(ep_idx=ep_idx, split=split)

            image_rel_paths = []
            for step_idx, action_id in enumerate(ep.actions):
                image_name = f"{ep.episode_id}_step_{step_idx:03d}.png"
                image_path = os.path.join(image_dir, image_name)
                _draw_frame(image_path, action_id, step_idx, len(ep.actions), ep.episode_id)

                image_rel = os.path.join("images", image_name)
                image_rel_paths.append(image_rel)

                row = {
                    "episode_id": ep.episode_id,
                    "step_idx": step_idx,
                    "image": image_rel,
                    "instruction": ep.instruction,
                    "action": action_id,
                }
                samples_f.write(json.dumps(row) + "\n")

            episodes_meta.append(
                {
                    "episode_id": ep.episode_id,
                    "instruction": ep.instruction,
                    "expert_actions": ep.actions,
                    "image_paths": image_rel_paths,
                    "shortest_path_length": ep.shortest_path_length,
                }
            )

    with open(episodes_path, "w", encoding="utf-8") as f:
        json.dump(episodes_meta, f, indent=2)

    return jsonl_path, episodes_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Task 3 synthetic IL dataset with episode metadata")
    p.add_argument("--out-dir", default="task 3/data")
    p.add_argument("--train-episodes", type=int, default=220)
    p.add_argument("--val-episodes", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    train_jsonl, train_eps = generate_dataset(args.out_dir, "train", args.train_episodes)
    val_jsonl, val_eps = generate_dataset(args.out_dir, "val", args.val_episodes)

    print("Generated Task 3 dataset")
    print(f"train_jsonl={train_jsonl}")
    print(f"train_episodes={train_eps}")
    print(f"val_jsonl={val_jsonl}")
    print(f"val_episodes={val_eps}")


if __name__ == "__main__":
    main()
