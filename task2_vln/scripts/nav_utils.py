"""
nav_utils.py — Shared utilities for navigation scripts.

Provides:
  - Simulator creation
  - Top-down map rendering
  - Coordinate conversions (world ↔ map)
  - Path / agent drawing
  - Frame composition
  - Font loading
  - Scene object scanning
  - Expert path planning
"""
import math
import os
import numpy as np
import habitat_sim
import quaternion
import cv2
from PIL import Image, ImageDraw, ImageFont
from habitat_sim.utils.settings import default_sim_settings, make_cfg

# ── Constants ──

ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]
ACTION_TO_ID = {n: i for i, n in enumerate(ACTION_NAMES)}

OBJECT_CATEGORIES = {
    0: "chair", 1: "table", 2: "sofa", 3: "bed", 4: "toilet",
    5: "book", 6: "tv_monitor", 7: "plant", 8: "sink", 9: "bathtub",
    10: "refrigerator", 11: "microwave", 12: "oven", 13: "counter",
    14: "fireplace", 15: "gym_equipment", 16: "seating", 17: "clothes",
    18: "picture", 19: "cabinet", 20: "shelf"
}

INSTANCE_COLORS = [
    (0,200,200),(200,100,50),(50,200,100),(200,50,150),(100,150,250),
    (250,200,50),(50,250,200),(200,50,50),(150,100,200),(100,200,150),
    (250,150,100),(50,150,150),(200,200,50),(150,50,200),(50,100,200),
]

PANEL_H = 480
PANEL_W = 480
BAR_H = 80

# Skip these structural/generic categories when listing objects
_SKIP_CATEGORIES = {"void", "misc", "objects", "wall", "floor", "ceiling", ""}


# ── Font ──

def load_font(size=16):
    """Try to load a good TTF font, fall back to PIL default."""
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


# ── Simulator ──

def make_simulator(scene_path, width=480, height=480, semantic=False):
    """Create and return a habitat_sim.Simulator."""
    settings = default_sim_settings.copy()
    settings.update({
        "scene": scene_path,
        "width": width,
        "height": height,
        "color_sensor": True,
        "depth_sensor": False,
        "semantic_sensor": semantic,
        "sensor_height": 1.5,
        "seed": 42,
        "silent": True,
        "enable_physics": False,
    })
    cfg = make_cfg(settings)
    cfg.sim_cfg.gpu_device_id = -1
    return habitat_sim.Simulator(cfg)


# ── Coordinate helpers ──

def get_relative_goal(agent_pos, agent_rot, goal_pos):
    """Return goal position relative to agent frame as (x, z)."""
    diff = goal_pos - agent_pos
    inv_rot = np.conjugate(agent_rot)
    local_diff = quaternion.rotate_vectors(inv_rot, diff)
    return np.array([local_diff[0], local_diff[2]], dtype=np.float32)


def world_to_map(pos, bounds, mpp):
    """Convert world (x, z) → map pixel (col, row)."""
    col = int((pos[0] - bounds[0][0]) / mpp)
    row = int((pos[2] - bounds[0][2]) / mpp)
    return col, row


def map_to_world(col, row, bounds, mpp, y_height):
    """Convert map pixel (col, row) → world (x, y, z)."""
    x = bounds[0][0] + col * mpp
    z = bounds[0][2] + row * mpp
    return np.array([x, y_height, z], dtype=np.float32)


# ── Map rendering ──

def get_topdown_map(sim, meters_per_pixel=0.05):
    """Get a top-down occupancy map from the pathfinder.
    Returns (map_img, bounds, meters_per_pixel).
    """
    pf = sim.pathfinder
    bounds = pf.get_bounds()
    height = sim.get_agent(0).state.position[1]
    grid = pf.get_topdown_view(meters_per_pixel, height)
    map_img = np.zeros((*grid.shape, 3), dtype=np.uint8)
    map_img[grid]  = [200, 200, 200]
    map_img[~grid] = [60, 60, 60]
    return map_img, bounds, meters_per_pixel


def draw_path_on_map(map_img, positions, bounds, mpp, color, thickness=2):
    """Draw a polyline of world positions on the map."""
    pts = [world_to_map(p, bounds, mpp) for p in positions]
    for i in range(1, len(pts)):
        cv2.line(map_img, pts[i-1], pts[i], color, thickness, cv2.LINE_AA)
    return map_img


def draw_agent_marker(map_img, pos, rot, bounds, mpp, color=(255, 255, 255)):
    """Draw a small triangle showing agent position + heading."""
    cx, cy = world_to_map(pos, bounds, mpp)
    fwd = quaternion.rotate_vectors(rot, np.array([0, 0, -1.0]))
    angle = math.atan2(fwd[0], fwd[2])
    r = 8
    pts = []
    for da in [0, 2.4, -2.4]:
        a = angle + da
        pts.append((int(cx + r * math.sin(a)), int(cy + r * math.cos(a))))
    cv2.fillConvexPoly(map_img, np.array(pts), color, cv2.LINE_AA)


def resize_map_to_panel(map_img, panel_w=PANEL_W, panel_h=PANEL_H):
    """Resize a map image into a fixed-size panel, preserving aspect ratio."""
    mh, mw = map_img.shape[:2]
    scale = min(panel_w / mw, panel_h / mh)
    new_w, new_h = int(mw * scale), int(mh * scale)
    resized = cv2.resize(map_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    y_off = (panel_h - new_h) // 2
    x_off = (panel_w - new_w) // 2
    panel[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return panel, scale, x_off, y_off, new_w, new_h


# ── Frame composition ──

def compose_frame(rgb_frame, map_img, instruction_text, step, action_name,
                  panel_h=PANEL_H, extra_panel=None):
    """
    Compose the final video frame:
      [FPV | Map | (optional extra panel)]  +  [status bar]
    rgb_frame: (H, W, 3) uint8 RGB
    map_img:   raw top-down map (any size, will be resized)
    extra_panel: optional (panel_h, W, 3) uint8 to append on the right
    """
    panel_w = panel_h
    fpv = cv2.resize(rgb_frame, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4)
    map_panel, *_ = resize_map_to_panel(map_img, panel_w, panel_h)

    parts = [fpv, map_panel]
    if extra_panel is not None:
        parts.append(extra_panel)
    top = np.hstack(parts)

    # Bottom bar
    total_w = top.shape[1]
    bar = np.zeros((BAR_H, total_w, 3), dtype=np.uint8)
    bar_pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(bar_pil)
    draw.text((16, 10), instruction_text, fill=(255, 255, 255), font=load_font(18))
    draw.text((16, 48), f"Step {step}  |  Action: {action_name}",
              fill=(0, 255, 128), font=load_font(14))
    bar = np.array(bar_pil)

    return np.vstack([top, bar])


# ── Expert path planning ──

def plan_expert_path(sim, goal):
    """Use greedy follower to plan a shortest-path action sequence to goal.
    Returns list of action IDs.
    """
    follower = sim.make_greedy_follower(agent_id=0)
    try:
        plan = follower.find_path(goal)
    except habitat_sim.errors.GreedyFollowerError:
        plan = []
    actions = [ACTION_TO_ID[a] for a in plan if a in ACTION_TO_ID]
    if not actions or actions[-1] != ACTION_TO_ID["stop"]:
        actions.append(ACTION_TO_ID["stop"])
    return actions


def find_farthest_goal(pathfinder, start, num_samples=200):
    """Sample random navigable points and return the farthest reachable one.
    Returns (goal_pos, shortest_path, geodesic_distance).
    """
    best_dist, best_goal, best_sp = 0.0, None, None
    for _ in range(num_samples):
        candidate = pathfinder.get_random_navigable_point()
        sp = habitat_sim.ShortestPath()
        sp.requested_start = start
        sp.requested_end = candidate
        if pathfinder.find_path(sp) and sp.geodesic_distance > best_dist:
            best_dist = sp.geodesic_distance
            best_goal = candidate
            best_sp = sp
    return best_goal, best_sp, best_dist


# ── Scene object scanning ──

def scan_scene_objects(sim):
    """Scan semantic scene and return navigable object positions by category.
    Returns dict: { category_name: [nav_point, ...] }
    """
    pf = sim.pathfinder
    categories = {}
    for obj in sim.semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        name = obj.category.name()
        if name in _SKIP_CATEGORIES:
            continue
        center = obj.aabb.center
        if callable(center):
            center = center()
        center = np.array(center, dtype=np.float32)
        nav_pt = pf.snap_point(center)
        if pf.is_navigable(nav_pt):
            if name not in categories:
                categories[name] = []
            categories[name].append(nav_pt)
    return categories


def set_agent_position(agent, position):
    """Set agent to a specific position."""
    state = habitat_sim.AgentState()
    state.position = position
    agent.set_state(state)
