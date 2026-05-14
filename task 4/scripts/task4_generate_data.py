#!/usr/bin/env python3
"""Task 4 data generation for generalization and ablation studies."""

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from PIL import Image


ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}


@dataclass
class ReducedEpisode:
    episode_id: str
    instruction: str
    expert_actions: List[int]
    image_paths: List[str]
    shortest_path_length: float
    scene_path: str
    start_position: List[float]
    goal_position: List[float]


INSTRUCTION_VARIANTS = {
    "move_forward": ["move forward", "walk ahead", "go straight", "advance forward"],
    "turn_left": ["turn left", "rotate left", "pivot left"],
    "turn_right": ["turn right", "rotate right", "pivot right"],
    "stop": ["stop", "halt", "come to a stop"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_scenes() -> List[Tuple[str, str]]:
    roots = [
        _repo_root() / "habitat-sim" / "data" / "scene_datasets" / "mp3d_example",
        _repo_root() / "habitat-sim" / "data" / "scene_datasets" / "mp3d",
    ]
    scenes: List[Tuple[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for glb_path in sorted(root.rglob("*.glb")):
            scenes.append((glb_path.stem, str(glb_path)))
    return scenes


def _make_simulator(scene_path: str) -> habitat_sim.Simulator:
    settings = default_sim_settings.copy()
    settings.update(
        {
            "scene": scene_path,
            "width": 256,
            "height": 256,
            "color_sensor": True,
            "depth_sensor": False,
            "semantic_sensor": False,
            "sensor_height": 1.5,
            "seed": 13,
            "silent": True,
            "enable_physics": False,
            "gpu_device_id": -1,
        }
    )
    cfg = make_cfg(settings)
    cfg.sim_cfg.gpu_device_id = -1
    return habitat_sim.Simulator(cfg)


def _describe_actions(actions: Sequence[int]) -> str:
    tokens: List[str] = []
    run_action = None
    run_length = 0

    def flush(action_id: int, length: int) -> None:
        if action_id == ACTION_TO_ID["move_forward"]:
            tokens.append("move forward" if length == 1 else f"move forward {length} steps")
        elif action_id == ACTION_TO_ID["turn_left"]:
            tokens.append("turn left" if length == 1 else f"turn left {length} times")
        elif action_id == ACTION_TO_ID["turn_right"]:
            tokens.append("turn right" if length == 1 else f"turn right {length} times")

    for action_id in actions:
        if run_action is None:
            run_action = action_id
            run_length = 1
        elif action_id == run_action:
            run_length += 1
        else:
            flush(run_action, run_length)
            run_action = action_id
            run_length = 1

    if run_action is not None:
        flush(run_action, run_length)

    if tokens:
        return ", then ".join(tokens) + ", then stop at the goal."
    return "Move to the destination and stop."


def _sample_episode(sim: habitat_sim.Simulator, episode_id: str, scene_id: str, scene_path: str) -> Optional[ReducedEpisode]:
    pathfinder = sim.pathfinder
    agent = sim.initialize_agent(0)

    for _ in range(120):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = start
        shortest.requested_end = goal
        if not pathfinder.find_path(shortest) or not np.isfinite(shortest.geodesic_distance) or shortest.geodesic_distance < 2.5:
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

        if len(expert_actions) > 24:
            continue

        instruction = _describe_actions(expert_actions)
        return ReducedEpisode(
            episode_id=episode_id,
            instruction=instruction,
            expert_actions=expert_actions,
            image_paths=[],
            shortest_path_length=float(shortest.geodesic_distance),
            scene_path=scene_path,
            start_position=np.array(start, dtype=float).tolist(),
            goal_position=np.array(goal, dtype=float).tolist(),
        )

    return None


def _generate_unseen_environment(num_episodes: int, out_dir: Path, scene_path: str, scene_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: List[Dict[str, object]] = []
    samples: List[Dict[str, object]] = []

    with _make_simulator(scene_path) as sim:
        for idx in range(num_episodes):
            episode_id = f"unseen_{idx:06d}"
            episode = _sample_episode(sim, episode_id, scene_id, scene_path)
            if episode is None:
                continue

            image_dir = out_dir / episode_id
            image_dir.mkdir(parents=True, exist_ok=True)
            image_paths: List[str] = []
            agent = sim.get_agent(0)
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array(episode.start_position, dtype=np.float32)
            agent_state.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
            agent.set_state(agent_state)

            for step_idx, action_id in enumerate(episode.expert_actions):
                obs = sim.get_sensor_observations()["color_sensor"]
                frame_path = image_dir / f"frame_{step_idx:03d}.png"
                Image.fromarray(obs).save(frame_path)
                rel_path = f"{episode_id}/frame_{step_idx:03d}.png"
                image_paths.append(rel_path)
                samples.append(
                    {
                        "episode_id": episode_id,
                        "step_idx": step_idx,
                        "image": rel_path,
                        "instruction": episode.instruction,
                        "action": action_id,
                        "scene_id": scene_id,
                    }
                )
                if action_id == ACTION_TO_ID["stop"]:
                    break
                sim.step(ACTION_NAMES[action_id])

            episodes.append(
                {
                    **asdict(episode),
                    "image_paths": image_paths,
                }
            )

    with open(out_dir / "episodes.json", "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2)
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def _load_episodes(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _paraphrase_instruction(base_instruction: str) -> str:
    words = base_instruction.replace(", then ", " | ").split(" |")
    replacements = []
    for word in words:
        lowered = word.lower()
        if "turn left" in lowered:
            replacements.append(random.choice(INSTRUCTION_VARIANTS["turn_left"]))
        elif "turn right" in lowered:
            replacements.append(random.choice(INSTRUCTION_VARIANTS["turn_right"]))
        elif "move forward" in lowered:
            replacements.append(random.choice(INSTRUCTION_VARIANTS["move_forward"]))
        elif "stop" in lowered:
            replacements.append(random.choice(INSTRUCTION_VARIANTS["stop"]))
        else:
            replacements.append(word)
    return ", then ".join(replacements)


def generate_paraphrased_instructions(base_episodes_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    base_episodes = _load_episodes(base_episodes_path)
    base_root = Path(base_episodes_path).parent
    paraphrased_episodes: List[Dict[str, object]] = []
    samples: List[Dict[str, object]] = []

    for ep in base_episodes:
        paraphrased_instruction = _paraphrase_instruction(str(ep["instruction"]))
        paraphrased_episodes.append({**ep, "instruction": paraphrased_instruction})
        for step_idx, action in enumerate(ep["expert_actions"]):
            samples.append(
                {
                    "episode_id": ep["episode_id"],
                    "step_idx": step_idx,
                    "image": str((base_root / str(ep["image_paths"][step_idx])).resolve()),
                    "instruction": paraphrased_instruction,
                    "action": int(action),
                    "scene_id": ep.get("scene_id", "unknown"),
                }
            )

    with open(os.path.join(out_dir, "episodes.json"), "w", encoding="utf-8") as f:
        json.dump(paraphrased_episodes, f, indent=2)
    with open(os.path.join(out_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Generated paraphrased instructions: {len(paraphrased_episodes)} episodes, {len(samples)} samples")
    return len(samples)


def generate_reduced_data(base_train_path: str, out_dir: str, percentage: int) -> Tuple[int, int]:
    os.makedirs(out_dir, exist_ok=True)
    base_episodes = _load_episodes(base_train_path)
    base_root = Path(base_train_path).parent
    base_samples_path = os.path.join(os.path.dirname(base_train_path), "samples.jsonl")
    with open(base_samples_path, "r", encoding="utf-8") as f:
        base_samples = [json.loads(line) for line in f if line.strip()]

    num_episodes = max(1, int(len(base_episodes) * percentage / 100))
    sampled_episodes = random.sample(base_episodes, num_episodes)
    sampled_episode_ids = {ep["episode_id"] for ep in sampled_episodes}
    sampled_samples = []
    for sample in base_samples:
        if sample["episode_id"] not in sampled_episode_ids:
            continue
        rel_image = str(sample["image"])
        sampled_samples.append(
            {
                **sample,
                "image": str((base_root / rel_image).resolve()) if not os.path.isabs(rel_image) else rel_image,
            }
        )

    with open(os.path.join(out_dir, "episodes.json"), "w", encoding="utf-8") as f:
        json.dump(sampled_episodes, f, indent=2)
    with open(os.path.join(out_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for sample in sampled_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {percentage}% reduced data: {len(sampled_episodes)} episodes, {len(sampled_samples)} samples")
    return len(sampled_episodes), len(sampled_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", default="task 3/data_large/train/episodes.json")
    parser.add_argument("--out-dir", default="task 4/data")
    parser.add_argument("--unseen-episodes", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scenes = _discover_scenes()
    if len(scenes) < 2:
        raise RuntimeError("Need at least two local MP3D scenes to create an unseen-environment split.")

    unseen_scene_id, unseen_scene_path = scenes[1]
    unseen_dir = os.path.join(args.out_dir, "unseen_env")
    _generate_unseen_environment(args.unseen_episodes, Path(unseen_dir), unseen_scene_path, unseen_scene_id)

    paraphrased_dir = os.path.join(args.out_dir, "paraphrased")
    generate_paraphrased_instructions(args.base_train, paraphrased_dir)

    for pct in [10, 25, 50]:
        reduced_dir = os.path.join(args.out_dir, f"reduced_{pct}pct")
        generate_reduced_data(args.base_train, reduced_dir, pct)

    print(f"\nTask 4 data generation complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
