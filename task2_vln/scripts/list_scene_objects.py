import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from collections import Counter

scene = "/run/media/rishi/New Volume/CPP/AI211/Matterport_Dataset/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
settings = default_sim_settings.copy()
settings.update({
    "scene": scene,
    "width": 64, "height": 64,
    "color_sensor": True, "depth_sensor": False,
    "semantic_sensor": True,
    "sensor_height": 1.5, "seed": 42,
    "silent": True, "enable_physics": False,
})
cfg = make_cfg(settings)
cfg.sim_cfg.gpu_device_id = -1
sim = habitat_sim.Simulator(cfg)

scene_obj = sim.semantic_scene
print("Scene: 17DRP5sb8fy")
print(f"Total objects: {len(scene_obj.objects)}")
print(f"Total categories: {len(scene_obj.categories)}")
print()

print("=== All Categories ===")
cats = {}
for cat in scene_obj.categories:
    cats[cat.index()] = cat.name()
    print(f"  ID {cat.index():3d}: {cat.name()}")

print()
print("=== Objects per Category ===")
cat_counts = Counter()
for obj in scene_obj.objects:
    if obj is not None and obj.category is not None:
        cat_counts[obj.category.name()] += 1

for name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:3d}x  {name}")

sim.close()
