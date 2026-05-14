#!/usr/bin/env python3
"""
Interactive Navigation Demo
============================
Unified GUI with clickable object list sidebar + map click navigation.
Layout: [FPV | Map | Object List]  +  [Status Bar]

Usage:
  python scripts/interactive_nav.py --scene <path_to.glb>
"""
import argparse
import math
import os
import sys
import numpy as np
import habitat_sim
import torch
import cv2
import imageio
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.objectnav_policy import ObjectNavPolicy

from nav_utils import (
    ACTION_NAMES, ACTION_TO_ID, INSTANCE_COLORS, PANEL_H, PANEL_W, BAR_H,
    load_font, make_simulator, get_topdown_map,
    world_to_map, map_to_world,
    draw_path_on_map, draw_agent_marker, resize_map_to_panel,
    plan_expert_path, scan_scene_objects, set_agent_position,
)

SIDEBAR_W = 280
ITEM_H = 28


class InteractiveNav:
    """Unified interactive ObjectNav + PointNav with GUI."""

    def __init__(self, scene_path, out_dir):
        self.out_dir = out_dir

        # Simulator
        self.sim = make_simulator(scene_path, semantic=True)
        self.agent = self.sim.initialize_agent(0)
        self.pf = self.sim.pathfinder

        # Scan objects
        self.categories = scan_scene_objects(self.sim)
        self.cat_list = sorted(self.categories.keys(),
                               key=lambda n: -len(self.categories[n]))

        # State
        self.scroll_offset = 0
        self.selected_cat = None
        self.selected_instance = None
        self.navigating = False
        self.expert_actions = []
        self.step_idx = 0
        self.frames = []

        # Agent start
        self.start = self.pf.get_random_navigable_point()
        set_agent_position(self.agent, self.start)
        self.y_height = self.start[1]

        # Map
        self.base_map, self.bounds, self.mpp = get_topdown_map(self.sim)
        self.all_positions = [np.array(self.start)]
        self.status = "Select an object from the list, or click map for PointNav"

        # Precompute map display transform
        _, self.map_scale, self.map_xoff, self.map_yoff, \
            self.map_nw, self.map_nh = resize_map_to_panel(self.base_map)

    # ── Click handler ──

    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        sidebar_x = PANEL_W * 2

        # Sidebar click → select category
        if x >= sidebar_x and x < sidebar_x + SIDEBAR_W:
            local_y = y - 40  # header offset
            if local_y >= 0:
                idx = local_y // ITEM_H + self.scroll_offset
                if 0 <= idx < len(self.cat_list):
                    cat = self.cat_list[idx]
                    if self.selected_cat == cat:
                        self.selected_cat = None
                    else:
                        self.selected_cat = cat
                        self.selected_instance = None
                        self.navigating = False
                        instances = self.categories[cat]
                        if len(instances) == 1:
                            self._start_nav(instances[0], cat)
                        else:
                            self.status = (f"Click a {cat} marker on the map "
                                           f"({len(instances)} instances)")
            return

        # Map click
        if PANEL_W <= x < PANEL_W * 2 and 0 <= y < PANEL_H:
            lx = x - PANEL_W - self.map_xoff
            ly = y - self.map_yoff
            if 0 <= lx < self.map_nw and 0 <= ly < self.map_nh:
                map_col = lx / self.map_scale
                map_row = ly / self.map_scale

                # If category selected → pick instance
                if self.selected_cat and not self.navigating:
                    best_d, best_pt = 999, None
                    for pt in self.categories[self.selected_cat]:
                        mc, mr = world_to_map(pt, self.bounds, self.mpp)
                        d = math.sqrt((mc - map_col)**2 + (mr - map_row)**2)
                        if d < best_d:
                            best_d, best_pt = d, pt
                    if best_pt is not None and best_d < 30:
                        self._start_nav(best_pt, self.selected_cat)
                    return

                # No category → PointNav
                if not self.navigating:
                    world_pt = map_to_world(map_col, map_row,
                                            self.bounds, self.mpp, self.y_height)
                    nav_pt = self.pf.snap_point(world_pt)
                    if self.pf.is_navigable(nav_pt):
                        self._start_nav(nav_pt, "point")

    def _start_nav(self, target, label):
        """Plan and begin expert navigation to target."""
        self.selected_instance = target
        self.navigating = True
        self.step_idx = 0
        self.expert_actions = plan_expert_path(self.sim, target)
        # Reset agent to current position (planning moves it)
        set_agent_position(self.agent, self.all_positions[-1])
        self.expert_actions = plan_expert_path(self.sim, target)
        d = float(np.linalg.norm(target - self.agent.get_state().position))
        self.status = (f"Navigating to {label} ({d:.1f}m, "
                       f"{len(self.expert_actions)} steps)")
        print(f"  >> Nav to {label}: {d:.1f}m, {len(self.expert_actions)} actions")

    # ── Rendering ──

    def _render_sidebar(self):
        sidebar = np.zeros((PANEL_H, SIDEBAR_W, 3), dtype=np.uint8)
        sidebar[:] = [30, 30, 40]

        # Header
        cv2.rectangle(sidebar, (0, 0), (SIDEBAR_W, 36), (50, 50, 70), -1)
        cv2.putText(sidebar, "OBJECTS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        visible = self.cat_list[self.scroll_offset:]
        for i, name in enumerate(visible):
            if i * ITEM_H + 40 > PANEL_H - 10:
                break
            y_top = 40 + i * ITEM_H
            count = len(self.categories[name])
            color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
            is_sel = (name == self.selected_cat)

            if is_sel:
                cv2.rectangle(sidebar, (0, y_top),
                              (SIDEBAR_W, y_top + ITEM_H - 2), (80, 80, 100), -1)
                cv2.rectangle(sidebar, (0, y_top),
                              (4, y_top + ITEM_H - 2), color, -1)

            cv2.circle(sidebar, (15, y_top + ITEM_H // 2), 5, color, -1, cv2.LINE_AA)
            cv2.putText(sidebar, f"{name} ({count})", (28, y_top + ITEM_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv2.LINE_AA)
        return sidebar

    def _render_frame(self):
        obs = self.sim.get_sensor_observations()
        rgb = obs["color_sensor"][..., :3]
        state = self.agent.get_state()

        # FPV
        fpv = cv2.resize(rgb, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LANCZOS4)

        # Map
        cur_map = self.base_map.copy()
        sx, sy = world_to_map(self.start, self.bounds, self.mpp)
        cv2.circle(cur_map, (sx, sy), 5, (0,255,0), -1, cv2.LINE_AA)

        # Instance markers
        if self.selected_cat:
            ci = self.cat_list.index(self.selected_cat)
            col = INSTANCE_COLORS[ci % len(INSTANCE_COLORS)]
            for idx, pt in enumerate(self.categories[self.selected_cat]):
                px, py = world_to_map(pt, self.bounds, self.mpp)
                cv2.circle(cur_map, (px, py), 7, col, -1, cv2.LINE_AA)
                cv2.circle(cur_map, (px, py), 7, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(cur_map, str(idx+1), (px-4, py+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)

        # Goal + shortest path
        if self.selected_instance is not None:
            gx, gy = world_to_map(self.selected_instance, self.bounds, self.mpp)
            cv2.circle(cur_map, (gx, gy), 8, (0,0,255), -1, cv2.LINE_AA)
            sp = habitat_sim.ShortestPath()
            sp.requested_start = np.array(state.position)
            sp.requested_end = self.selected_instance
            if self.pf.find_path(sp):
                draw_path_on_map(cur_map, sp.points, self.bounds, self.mpp, (0,200,0), 2)

        # Agent path + marker
        draw_path_on_map(cur_map, self.all_positions, self.bounds, self.mpp, (30,30,255), 2)
        draw_agent_marker(cur_map, state.position, state.rotation, self.bounds, self.mpp)

        # Resize map
        map_panel, *_ = resize_map_to_panel(cur_map)

        # Sidebar
        sidebar = self._render_sidebar()

        # Compose
        top = np.hstack([fpv, map_panel, sidebar])

        # Bottom bar
        total_w = top.shape[1]
        bar = np.zeros((BAR_H, total_w, 3), dtype=np.uint8)
        bar_pil = Image.fromarray(bar)
        draw = ImageDraw.Draw(bar_pil)
        draw.text((16, 10), self.status, fill=(255,255,255), font=load_font(16))
        info = f"Step {self.step_idx}" if self.navigating else "Click object or map"
        draw.text((16, 45), info, fill=(0,255,128), font=load_font(13))
        draw.text((total_w-300, 10), "Click object list -> select category",
                  fill=(180,180,180), font=load_font(11))
        draw.text((total_w-300, 28), "Click map -> navigate to point/instance",
                  fill=(180,180,180), font=load_font(11))
        draw.text((total_w-300, 46), "R = reset | Q = quit & save",
                  fill=(180,180,180), font=load_font(11))
        bar = np.array(bar_pil)

        return np.vstack([top, bar])

    # ── Main loop ──

    def step(self):
        """Execute one expert action if navigating."""
        if self.navigating and self.step_idx < len(self.expert_actions):
            aid = self.expert_actions[self.step_idx]
            self.step_idx += 1
            name = ACTION_NAMES[aid]
            if name == "stop":
                self.navigating = False
                self.status = "Reached target! Select another object or click map."
                print("  >> Reached target!")
            else:
                self.sim.step(name)
                self.all_positions.append(np.array(self.agent.get_state().position))

    def run(self):
        win = "Interactive Navigation"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.on_click)

        print("\n" + "="*60)
        print("  INTERACTIVE NAVIGATION")
        print("  Click object names on right panel")
        print("  Click instance markers or empty map to navigate")
        print("  R = reset | Q = quit & save video")
        print("="*60 + "\n")

        while True:
            self.step()
            frame = self._render_frame()
            self.frames.append(frame.copy())
            cv2.imshow(win, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(80)
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.start = self.pf.get_random_navigable_point()
                set_agent_position(self.agent, self.start)
                self.all_positions = [np.array(self.start)]
                self.selected_cat = None
                self.selected_instance = None
                self.navigating = False
                self.status = "Reset! Select an object or click map."
                print("  >> Reset position")

        cv2.destroyAllWindows()
        self.sim.close()

        if self.frames:
            os.makedirs(self.out_dir, exist_ok=True)
            out = os.path.join(self.out_dir, "interactive_nav_session.mp4")
            imageio.mimsave(out, self.frames, fps=8)
            print(f"\n  Saved: {out} ({len(self.frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Navigation Demo")
    parser.add_argument("--scene", required=True, help="Path to .glb scene file")
    parser.add_argument("--out-dir", default="videos_interactive")
    args = parser.parse_args()

    app = InteractiveNav(args.scene, args.out_dir)
    app.run()
