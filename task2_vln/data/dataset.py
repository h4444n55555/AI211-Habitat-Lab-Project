import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


@dataclass
class VlnSample:
    image_path: str
    instruction: str
    action: int


class JsonlVlnDataset(Dataset):
    """Simple JSONL dataset for imitation-learning style VLN.

    Expected fields per JSON line:
    - image: path to RGB image file (required)
    - instruction: natural-language instruction (required)
    - action: int in [0, 3] mapped as 0=fwd,1=left,2=right,3=stop (required)
    """

    def __init__(self, jsonl_path: str, image_root: Optional[str] = None, augment: bool = False) -> None:
        self.jsonl_path = jsonl_path
        self.base_dir = os.path.dirname(os.path.abspath(jsonl_path))
        self.image_root = image_root
        self.samples: List[VlnSample] = []
        self.augment = augment

        # Training augmentations for better generalization
        if augment:
            self.transform = T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomHorizontalFlip(p=0.1),  # Low prob — flipping changes navigation semantics
            ])
        else:
            self.transform = None

        self._load()

    def _resolve_image_path(self, image_path: str) -> str:
        if os.path.isabs(image_path):
            return image_path
        if self.image_root is not None:
            return os.path.join(self.image_root, image_path)
        return os.path.join(self.base_dir, image_path)

    def _load(self) -> None:
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                if "image" not in record or "instruction" not in record or "action" not in record:
                    raise ValueError(
                        f"{self.jsonl_path}:{line_idx} missing one of required keys: image, instruction, action"
                    )

                image_path = self._resolve_image_path(str(record["image"]))
                instruction = str(record["instruction"])
                action = int(record["action"])
                if action < 0 or action > 3:
                    raise ValueError(f"{self.jsonl_path}:{line_idx} has invalid action={action}, expected 0..3")

                self.samples.append(VlnSample(image_path=image_path, instruction=instruction, action=action))

        if not self.samples:
            raise ValueError(f"No samples found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")

        # Apply training augmentations if enabled
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "instruction": sample.instruction,
            "action": sample.action,
            "image_path": sample.image_path,
        }


def build_collate_fn(processor: Any, max_text_length: int = 64) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [row["image"] for row in batch]
        instructions = [row["instruction"] for row in batch]
        actions = torch.tensor([int(row["action"]) for row in batch], dtype=torch.long)
        image_paths = [row["image_path"] for row in batch]

        enc = processor(
            text=instructions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_length,
        )

        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "actions": actions,
            "image_paths": image_paths,
            "instructions": instructions,
        }

    return collate_fn
