import json
import os
import random
from PIL import Image
import argparse

def generate_dummy_data(task_type, num_samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, f"dummy_{task_type}.jsonl")
    
    # Create a dummy blank image
    dummy_img_path = os.path.join(img_dir, "dummy.png")
    if not os.path.exists(dummy_img_path):
        img = Image.new('RGB', (256, 256), color = (73, 109, 137))
        img.save(dummy_img_path)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            action = random.randint(0, 3)
            sample = {
                "image": "images/dummy.png",
                "action": action
            }
            if task_type == "pointnav":
                sample["point_goal"] = [random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)]
            elif task_type == "objectnav":
                sample["object_id"] = random.randint(0, 20)
                
            f.write(json.dumps(sample) + "\n")
            
    print(f"Generated {num_samples} dummy {task_type} samples in {jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="dummy_nav_data")
    args = parser.parse_args()
    
    generate_dummy_data("pointnav", 100, args.out_dir)
    generate_dummy_data("objectnav", 100, args.out_dir)
