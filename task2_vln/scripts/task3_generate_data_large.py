#!/usr/bin/env python3
"""Generate Task 3 imitation-learning data from Habitat-Sim MP3D scenes."""
import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from PIL import Image


ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}
TURN_DEGREE = 30.0
MAX_EPISODE_STEPS = 24


@dataclass
class EpisodeSpec:
    episode_id: str
    scene_id: str
    scene_path: str
    instruction: str
    start_position: List[float]
    goal_position: List[float]
    expert_actions: List[int]
    shortest_path_length: float
    path_length: float
    image_paths: List[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_scan_ids(mp3d_download_root: Optional[Path]) -> List[str]:
    if mp3d_download_root is None:
        return []
    scans_txt = mp3d_download_root / "scans.txt"
    if not scans_txt.exists():
        return []
    return [line.strip() for line in scans_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def _discover_scene_paths(scene_roots: Sequence[Path], preferred_scan_ids: Sequence[str]) -> List[Tuple[str, str]]:
    discovered: List[Tuple[str, str]] = []
    seen_paths = set()

    for root in scene_roots:
        if not root.exists():
            continue
        for glb_path in root.rglob("*.glb"):
            if glb_path in seen_paths:
                continue
            seen_paths.add(glb_path)
            scene_id = glb_path.stem
            if preferred_scan_ids and scene_id not in preferred_scan_ids:
                continue
            discovered.append((scene_id, str(glb_path)))

    if not discovered:
        fallback = _repo_root() / "habitat-sim" / "data" / "scene_datasets" / "mp3d_example" / "17DRP5sb8fy" / "17DRP5sb8fy.glb"
        if fallback.exists():
            discovered.append(("17DRP5sb8fy", str(fallback)))

    return sorted(discovered, key=lambda item: item[0])


def _make_simulator(scene_path: str, width: int, height: int, seed: int) -> habitat_sim.Simulator:
    settings = default_sim_settings.copy()
    settings.update(
        {
            "scene": scene_path,
            "width": width,
            "height": height,
            "color_sensor": True,
            "depth_sensor": False,
            "semantic_sensor": False,
            "sensor_height": 1.5,
            "seed": seed,
            "silent": True,
            "enable_physics": False,
            "frustum_culling": True,
            "gpu_device_id": -1,
        }
    )
    cfg = make_cfg(settings)
    cfg.sim_cfg.gpu_device_id = -1
    return habitat_sim.Simulator(cfg)


def _action_name(action_id: int) -> str:
    return ACTION_NAMES[action_id]


def _describe_actions(actions: Sequence[int]) -> str:
    if not actions:
        return "Move toward the destination and stop."

    pieces: List[str] = []
    run_action = actions[0]
    run_length = 0

    def flush(action_id: int, length: int) -> None:
        if action_id == ACTION_TO_ID["move_forward"]:
            if length == 1:
                pieces.append("move forward")
            else:
                pieces.append(f"move forward {length} steps")
        elif action_id == ACTION_TO_ID["turn_left"]:
            pieces.append("turn left" if length == 1 else f"turn left {length} times")
        elif action_id == ACTION_TO_ID["turn_right"]:
            pieces.append("turn right" if length == 1 else f"turn right {length} times")

    for action_id in actions:
        if action_id == run_action:
            run_length += 1
        else:
            flush(run_action, run_length)
            run_action = action_id
            run_length = 1

    flush(run_action, run_length)

    if pieces and pieces[-1].startswith("move forward"):
        return f"{', then '.join(pieces)}, then stop at the goal."
    return f"{', then '.join(pieces)}, then stop."


def _sample_episode(
    sim: habitat_sim.Simulator,
    scene_id: str,
    scene_path: str,
    episode_id: str,
    split_dir: Path,
) -> Optional[Tuple[EpisodeSpec, List[Dict[str, object]]]]:
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        return None

    agent = sim.initialize_agent(0)
    max_attempts = 120

    for _ in range(max_attempts):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = start
        shortest.requested_end = goal
        if not pathfinder.find_path(shortest):
            continue
        if not math.isfinite(shortest.geodesic_distance) or shortest.geodesic_distance < 2.5:
            continue

        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(start, dtype=np.float32)
        agent_state.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
        agent.set_state(agent_state)

        follower = sim.make_greedy_follower(agent_id=0)
        try:
            action_plan = follower.find_path(goal)
        except habitat_sim.errors.GreedyFollowerError:
            continue

        expert_actions = [ACTION_TO_ID[action] for action in action_plan if action in ACTION_TO_ID]
        if not expert_actions or expert_actions[-1] != ACTION_TO_ID["stop"]:
            expert_actions.append(ACTION_TO_ID["stop"])

        if len(expert_actions) > MAX_EPISODE_STEPS:
            continue

        image_dir = split_dir / episode_id
        image_dir.mkdir(parents=True, exist_ok=True)

        image_paths: List[str] = []
        samples: List[Dict[str, object]] = []
        path_length = 0.0
        previous_position = np.array(start, dtype=np.float32)

        for step_idx, action_id in enumerate(expert_actions):
            observations = sim.get_sensor_observations()
            rgb_obs = observations["color_sensor"]
            frame_path = image_dir / f"frame_{step_idx:03d}.png"
            Image.fromarray(rgb_obs).save(frame_path)

            image_paths.append(f"{episode_id}/frame_{step_idx:03d}.png")
            samples.append(
                {
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "image": image_paths[-1],
                    "instruction": _describe_actions(expert_actions),
                    "action": action_id,
                    "scene_id": scene_id,
                }
            )

            if action_id == ACTION_TO_ID["stop"]:
                break

            action_name = _action_name(action_id)
            sim.step(action_name)
            current_position = np.array(sim.get_agent(0).state.position, dtype=np.float32)
            path_length += float(np.linalg.norm(current_position - previous_position))
            previous_position = current_position

        episode = EpisodeSpec(
            episode_id=episode_id,
            scene_id=scene_id,
            scene_path=scene_path,
            instruction=_describe_actions(expert_actions),
            start_position=np.array(start, dtype=float).tolist(),
            goal_position=np.array(goal, dtype=float).tolist(),
            expert_actions=expert_actions,
            shortest_path_length=float(shortest.geodesic_distance),
            path_length=float(path_length),
            image_paths=image_paths,
        )
        return episode, samples

    return None


def generate_dataset(out_dir: str, train_episodes: int = 500, val_episodes: int = 100, seed: int = 42, width: int = 256, height: int = 256) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    repo_root = _repo_root()
    mp3d_download_root = repo_root / "mp3d_download"
    preferred_scan_ids = _read_scan_ids(mp3d_download_root)
    scene_roots = [
        repo_root / "habitat-sim" / "data" / "scene_datasets" / "mp3d_example",
        repo_root / "habitat-sim" / "data" / "scene_datasets" / "mp3d",
    ]
    scenes = _discover_scene_paths(scene_roots, preferred_scan_ids)
    if not scenes:
        raise RuntimeError("No Habitat-MP3D scenes were found locally. Expected a GLB under habitat-sim/data/scene_datasets/.")

    print("Discovered scenes:")
    for scene_id, scene_path in scenes:
        print(f"  {scene_id}: {scene_path}")

    def generate_split(num_episodes: int, split_name: str) -> Tuple[List[EpisodeSpec], List[Dict[str, object]]]:
        split_dir = Path(out_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        episodes: List[EpisodeSpec] = []
        samples: List[Dict[str, object]] = []
        with _make_simulator(scenes[0][1], width=width, height=height, seed=seed) as sim:
            for ep_idx in range(num_episodes):
                episode_id = f"{split_name}_{ep_idx:06d}"
                sampled = _sample_episode(sim, scenes[0][0], scenes[0][1], episode_id, split_dir)
                if sampled is None:
                    continue
                episode, episode_samples = sampled
                episodes.append(episode)
                samples.extend(episode_samples)

        episodes_file = split_dir / "episodes.json"
        with open(episodes_file, "w", encoding="utf-8") as f:
            json.dump([asdict(ep) for ep in episodes], f, indent=2)

        jsonl_file = split_dir / "samples.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"Generated {split_name}: {len(episodes)} episodes, {len(samples)} samples")
        return episodes, samples

    train_eps, train_samples = generate_split(train_episodes, "train")
    val_eps, val_samples = generate_split(val_episodes, "val")

    print("\nDataset generation complete:")
    print(f"  Train: {len(train_eps)} episodes, {len(train_samples)} samples")
    print(f"  Val: {len(val_eps)} episodes, {len(val_samples)} samples")
    print(f"  Output: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 3 IL dataset from Habitat-Sim MP3D scenes")
    parser.add_argument("--out-dir", default="task 3/data_large")
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    generate_dataset(args.out_dir, args.train_episodes, args.val_episodes, args.seed, args.width, args.height)


if __name__ == "__main__":
    main()
