import os
import json
import numpy as np
import habitat_sim
import quaternion
import random
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg

ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}

def make_simulator(scene_path):
    settings = default_sim_settings.copy()
    settings.update({
        "scene": scene_path,
        "width": 256,
        "height": 256,
        "color_sensor": True,
        "depth_sensor": False,
        "semantic_sensor": False,
        "sensor_height": 1.5,
        "seed": 42,
        "silent": True,
        "enable_physics": False,
    })
    cfg = make_cfg(settings)
    cfg.sim_cfg.gpu_device_id = -1
    return habitat_sim.Simulator(cfg)

def get_relative_goal(agent_pos, agent_rot, goal_pos):
    diff = goal_pos - agent_pos
    inv_rot = np.conjugate(agent_rot)
    local_diff = quaternion.rotate_vectors(inv_rot, diff)
    return [float(local_diff[0]), float(local_diff[2])]

def generate_real_data(scene_path, out_dir, num_episodes=20):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    sim = make_simulator(scene_path)
    pathfinder = sim.pathfinder
    agent = sim.initialize_agent(0)
    
    pointnav_samples = []
    objectnav_samples = []
    
    ep_count = 0
    img_idx = 0
    
    while ep_count < num_episodes:
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = start
        shortest.requested_end = goal
        if not pathfinder.find_path(shortest) or shortest.geodesic_distance < 2.0:
            continue
            
        agent_state = habitat_sim.AgentState()
        agent_state.position = start
        agent_state.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
        agent.set_state(agent_state)
        
        follower = sim.make_greedy_follower(agent_id=0)
        try:
            action_plan = follower.find_path(goal)
        except habitat_sim.errors.GreedyFollowerError:
            continue
            
        expert_actions = [ACTION_TO_ID[a] for a in action_plan if a in ACTION_TO_ID]
        if not expert_actions or expert_actions[-1] != ACTION_TO_ID["stop"]:
            expert_actions.append(ACTION_TO_ID["stop"])
            
        if len(expert_actions) > 24:
            continue
            
        object_id = random.randint(0, 20)
        
        for action_id in expert_actions:
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][..., :3]
            img_name = f"frame_{img_idx:05d}.png"
            img_path = os.path.join(img_dir, img_name)
            Image.fromarray(rgb).save(img_path)
            
            state = agent.get_state()
            rel_goal = get_relative_goal(state.position, state.rotation, goal)
            
            pointnav_samples.append({
                "image": f"images/{img_name}",
                "point_goal": rel_goal,
                "action": action_id
            })
            
            objectnav_samples.append({
                "image": f"images/{img_name}",
                "object_id": object_id,
                "action": action_id
            })
            
            if action_id == ACTION_TO_ID["stop"]:
                break
                
            sim.step(ACTION_NAMES[action_id])
            img_idx += 1
            
        ep_count += 1
        print(f"Generated episode {ep_count}/{num_episodes} - Total samples: {img_idx}")
        
    sim.close()
    
    with open(os.path.join(out_dir, "train_pointnav.jsonl"), "w") as f:
        for s in pointnav_samples:
            f.write(json.dumps(s) + "\n")
            
    with open(os.path.join(out_dir, "train_objectnav.jsonl"), "w") as f:
        for s in objectnav_samples:
            f.write(json.dumps(s) + "\n")

if __name__ == "__main__":
    generate_real_data(
        scene_path="/run/media/rishi/New Volume/CPP/AI211/Matterport_Dataset/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
        out_dir="real_nav_data",
        num_episodes=20
    )
