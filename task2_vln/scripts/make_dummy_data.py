import argparse
import json
import os
import random
from typing import List, Tuple

from PIL import Image, ImageDraw


ACTIONS = [0, 1, 2, 3]
INSTRUCTION_TEMPLATES = {
    0: [
        "Go forward to the next hallway.",
        "Move straight ahead.",
        "Continue forward.",
    ],
    1: [
        "Turn left at the corridor.",
        "Rotate left and proceed.",
        "Take a left turn.",
    ],
    2: [
        "Turn right near the wall.",
        "Rotate right and continue.",
        "Take a right turn.",
    ],
    3: [
        "Stop near the target.",
        "Halt at the destination.",
        "Stop now.",
    ],
}


def _draw_image(path: str, action: int, size: Tuple[int, int] = (224, 224)) -> None:
    # The synthetic color cue intentionally makes overfit checks easy.
    base_color = {
        0: (70, 150, 70),
        1: (180, 70, 70),
        2: (70, 70, 180),
        3: (170, 170, 70),
    }[action]
    image = Image.new("RGB", size, color=base_color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=(255, 255, 255), width=4)
    draw.text((40, size[1] // 2 - 10), f"action={action}", fill=(255, 255, 255))
    image.save(path)


def _sample_record(image_rel_path: str, action: int) -> dict:
    instruction = random.choice(INSTRUCTION_TEMPLATES[action])
    return {
        "image": image_rel_path,
        "instruction": instruction,
        "action": action,
    }


def _write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny dummy dataset for Task 2 smoke tests")
    parser.add_argument("--out-dir", default="dummy_data")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    train_records: List[dict] = []
    val_records: List[dict] = []

    for split, count, bucket in [
        ("train", args.train_size, train_records),
        ("val", args.val_size, val_records),
    ]:
        for idx in range(count):
            action = random.choice(ACTIONS)
            image_name = f"{split}_{idx:05d}.png"
            image_path = os.path.join(images_dir, image_name)
            _draw_image(image_path, action)
            bucket.append(_sample_record(os.path.join("images", image_name), action))

    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_records)
    _write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_records)

    print(f"Created dummy dataset in: {args.out_dir}")
    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")


if __name__ == "__main__":
    main()
