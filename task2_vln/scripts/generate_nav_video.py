"""
Generate professional navigation demo videos (PointNav, ObjectNav, Combined).
Uses nav_utils for shared utilities.
"""
import argparse
import numpy as np
import habitat_sim
import torch
import imageio
import cv2
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pointnav_policy import PointNavPolicy
from models.objectnav_policy import ObjectNavPolicy
from nav_utils import (
    ACTION_NAMES, OBJECT_CATEGORIES,
    load_font, make_simulator, get_relative_goal,
    get_topdown_map, world_to_map, draw_path_on_map, draw_agent_marker,
    compose_frame, plan_expert_path, find_farthest_goal,
    scan_scene_objects, set_agent_position,
)
from transformers import AutoProcessor


# ───────────────── PointNav video ──────────────────

def generate_pointnav_video(scene_path, out_file, max_steps=120):
    print("Generating PointNav video...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = "openai/clip-vit-base-patch16"
    processor = AutoProcessor.from_pretrained(clip_model)
    model = PointNavPolicy(clip_model_name=clip_model).to(device)
    if os.path.exists("pointnav_model.pt"):
        model.load_state_dict(torch.load("pointnav_model.pt", map_location=device))
        print("  Loaded pointnav_model.pt")
    model.eval()

    sim = make_simulator(scene_path)
    agent = sim.initialize_agent(0)
    pf = sim.pathfinder

    start = pf.get_random_navigable_point()
    goal, sp, dist = find_farthest_goal(pf, start)
    print(f"  PointNav goal distance: {dist:.2f}m")

    set_agent_position(agent, start)
    expert_actions = plan_expert_path(sim, goal)
    print(f"  Expert plan length: {len(expert_actions)} actions")

    # Reset to start after planning
    set_agent_position(agent, start)

    # Map setup
    base_map, bounds, mpp = get_topdown_map(sim)
    map_with_sp = draw_path_on_map(base_map.copy(), sp.points, bounds, mpp, (0,200,0), 2)
    sx, sy = world_to_map(start, bounds, mpp)
    gx, gy = world_to_map(goal, bounds, mpp)
    cv2.circle(map_with_sp, (sx, sy), 6, (0,255,0), -1, cv2.LINE_AA)
    cv2.circle(map_with_sp, (gx, gy), 6, (0,0,255), -1, cv2.LINE_AA)

    frames = []
    agent_positions = [np.array(start)]

    with torch.no_grad():
        for step, aid in enumerate(expert_actions):
            if step >= max_steps:
                break
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][..., :3]
            state = agent.get_state()
            rel_goal = get_relative_goal(state.position, state.rotation, goal)
            dist_now = float(np.linalg.norm(goal - state.position))

            # Model inference (reference)
            image_pt = processor(images=rgb, return_tensors="pt")["pixel_values"].to(device)
            goal_pt = torch.tensor(rel_goal, dtype=torch.float32).unsqueeze(0).to(device)
            model(image_pt, goal_pt)

            expert_name = ACTION_NAMES[aid]
            cur_map = map_with_sp.copy()
            draw_path_on_map(cur_map, agent_positions, bounds, mpp, (30,30,255), 2)
            draw_agent_marker(cur_map, state.position, state.rotation, bounds, mpp)

            instruction = (
                f"PointNav: Navigate to goal ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})  "
                f"|  Relative: ({rel_goal[0]:+.2f}, {rel_goal[1]:+.2f})  "
                f"|  Distance: {dist_now:.2f}m"
            )
            frame = compose_frame(rgb, cur_map, instruction, step, expert_name)
            frames.append(frame)

            if expert_name == "stop":
                frames.extend([frame] * 12)
                break
            sim.step(expert_name)
            agent_positions.append(np.array(agent.get_state().position))

    sim.close()
    imageio.mimsave(out_file, frames, fps=8)
    print(f"  Saved {out_file}  ({len(frames)} frames)")


# ───────────────── ObjectNav video ─────────────────

def generate_objectnav_video(scene_path, out_file, max_steps=80):
    print("Generating ObjectNav video...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = "openai/clip-vit-base-patch16"
    processor = AutoProcessor.from_pretrained(clip_model)
    model = ObjectNavPolicy(clip_model_name=clip_model).to(device)
    if os.path.exists("objectnav_model.pt"):
        model.load_state_dict(torch.load("objectnav_model.pt", map_location=device))
        print("  Loaded objectnav_model.pt")
    model.eval()

    sim = make_simulator(scene_path)
    agent = sim.initialize_agent(0)
    start = sim.pathfinder.get_random_navigable_point()
    set_agent_position(agent, start)

    base_map, bounds, mpp = get_topdown_map(sim)
    sx, sy = world_to_map(start, bounds, mpp)
    cv2.circle(base_map, (sx, sy), 6, (0,255,0), -1, cv2.LINE_AA)

    frames = []
    agent_positions = [np.array(start)]
    object_id = 5
    target_name = OBJECT_CATEGORIES.get(object_id, f"category_{object_id}")

    with torch.no_grad():
        for step in range(max_steps):
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][..., :3]

            image_pt = processor(images=rgb, return_tensors="pt")["pixel_values"].to(device)
            obj_pt = torch.tensor([object_id], dtype=torch.long).to(device)
            out = model(image_pt, obj_pt)
            action_id = torch.argmax(out["logits"], dim=-1).item()
            action_name = ACTION_NAMES[action_id]

            state = agent.get_state()
            cur_map = base_map.copy()
            draw_path_on_map(cur_map, agent_positions, bounds, mpp, (30,30,255), 2)
            draw_agent_marker(cur_map, state.position, state.rotation, bounds, mpp)

            instruction = (
                f"ObjectNav: Go to the {target_name}.  "
                f"Find and navigate to the farthest {target_name} in the environment."
            )
            frame = compose_frame(rgb, cur_map, instruction, step, action_name)
            frames.append(frame)

            if action_name == "stop":
                frames.extend([frame] * 8)
                break
            sim.step(action_name)
            agent_positions.append(np.array(agent.get_state().position))

    sim.close()
    imageio.mimsave(out_file, frames, fps=8)
    print(f"  Saved {out_file}  ({len(frames)} frames)")


# ──────────── Combined ObjectNav + Pick + PointNav ─────────────

def generate_combined_video(scene_path, out_file, max_steps_per_phase=80):
    print("Generating Combined (ObjectNav -> Pick -> PointNav) video...")
    sim = make_simulator(scene_path, semantic=True)
    agent = sim.initialize_agent(0)
    pf = sim.pathfinder

    # Find nearest chair
    categories = scan_scene_objects(sim)
    start = pf.get_random_navigable_point()

    chair_positions = categories.get("chair", [])
    if not chair_positions:
        print("  WARNING: No chair found, using random point")
        chair_positions = [pf.get_random_navigable_point()]

    dists = [float(np.linalg.norm(np.array(cp) - np.array(start))) for cp in chair_positions]
    chair_pos = chair_positions[int(np.argmin(dists))]
    print(f"  Nearest chair at distance {min(dists):.2f}m")

    # Far drop-off
    drop_pos, _, drop_dist = find_farthest_goal(pf, chair_pos)
    print(f"  Drop-off at distance {drop_dist:.2f}m from chair")

    # Map
    set_agent_position(agent, start)
    base_map, bounds, mpp = get_topdown_map(sim)
    for pt, col in [(start,(0,255,0)), (chair_pos,(0,255,255)), (drop_pos,(0,0,255))]:
        px, py = world_to_map(pt, bounds, mpp)
        cv2.circle(base_map, (px, py), 7, col, -1, cv2.LINE_AA)

    frames = []
    all_pos = [np.array(start)]

    # ── Phase 1: to chair ──
    set_agent_position(agent, start)
    actions1 = plan_expert_path(sim, chair_pos)
    set_agent_position(agent, start)  # reset after planning

    sp1 = habitat_sim.ShortestPath()
    sp1.requested_start = start; sp1.requested_end = chair_pos
    pf.find_path(sp1)
    p1_map = draw_path_on_map(base_map.copy(), sp1.points, bounds, mpp, (255,255,0), 2)

    print(f"  Phase 1 plan: {len(actions1)} actions")
    for step, aid in enumerate(actions1):
        if step >= max_steps_per_phase: break
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][..., :3]
        state = agent.get_state()
        name = ACTION_NAMES[aid]

        cur = p1_map.copy()
        draw_path_on_map(cur, all_pos, bounds, mpp, (30,30,255), 2)
        draw_agent_marker(cur, state.position, state.rotation, bounds, mpp)

        d = float(np.linalg.norm(np.array(chair_pos) - state.position))
        frame = compose_frame(rgb, cur,
            f"Phase 1 - ObjectNav: Go to the nearest chair  |  Distance: {d:.2f}m",
            step, name)
        frames.append(frame)
        if name == "stop": break
        sim.step(name)
        all_pos.append(np.array(agent.get_state().position))

    # ── Phase 2: pick ──
    print("  Phase 2: Picking up chair...")
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][..., :3]
    state = agent.get_state()
    cur = p1_map.copy()
    draw_path_on_map(cur, all_pos, bounds, mpp, (30,30,255), 2)
    draw_agent_marker(cur, state.position, state.rotation, bounds, mpp)
    for i in range(16):
        frames.append(compose_frame(rgb, cur, "Phase 2 - Picking up the chair...", i, "PICK"))

    # ── Phase 3: to drop-off ──
    actions2 = plan_expert_path(sim, drop_pos)
    sp2 = habitat_sim.ShortestPath()
    sp2.requested_start = np.array(agent.get_state().position)
    sp2.requested_end = drop_pos
    pf.find_path(sp2)
    p3_map = draw_path_on_map(p1_map.copy(), sp2.points, bounds, mpp, (200,0,200), 2)
    p3_map = draw_path_on_map(p3_map, all_pos, bounds, mpp, (30,30,255), 2)

    print(f"  Phase 3 plan: {len(actions2)} actions")
    for step, aid in enumerate(actions2):
        if step >= max_steps_per_phase: break
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][..., :3]
        state = agent.get_state()
        name = ACTION_NAMES[aid]
        d = float(np.linalg.norm(np.array(drop_pos) - state.position))

        cur = p3_map.copy()
        draw_path_on_map(cur, all_pos, bounds, mpp, (30,30,255), 2)
        draw_agent_marker(cur, state.position, state.rotation, bounds, mpp)

        frame = compose_frame(rgb, cur,
            f"Phase 3 - PointNav: Carry chair to ({drop_pos[0]:.1f}, {drop_pos[1]:.1f}, {drop_pos[2]:.1f})  |  Distance: {d:.2f}m",
            step, name)
        frames.append(frame)
        if name == "stop":
            frames.extend([frame] * 12)
            break
        sim.step(name)
        all_pos.append(np.array(agent.get_state().position))

    sim.close()
    imageio.mimsave(out_file, frames, fps=8)
    print(f"  Saved {out_file}  ({len(frames)} frames, ~{len(frames)/8:.1f}s)")


# ───────────────────── main ────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", default="videos")
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generate_pointnav_video(args.scene, os.path.join(args.out_dir, "pointnav_demo.mp4"), args.max_steps)
    generate_objectnav_video(args.scene, os.path.join(args.out_dir, "objectnav_demo.mp4"), args.max_steps)
    generate_combined_video(args.scene, os.path.join(args.out_dir, "combined_objectnav_pointnav_demo.mp4"), args.max_steps)
