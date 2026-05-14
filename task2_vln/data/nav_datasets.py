import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class PointNavDataset(Dataset):
    """
    Dataset loader for PointNav imitation learning.
    Expects jsonl format where each line has:
    - image: path to image frame
    - point_goal: [x, y] or [r, theta] goal vector
    - action: integer action ID
    """
    def __init__(self, jsonl_path, image_root=None, transform=None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image"]
        if self.image_root:
            img_path = os.path.join(self.image_root, img_path)
            
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        point_goal = torch.tensor(sample["point_goal"], dtype=torch.float32)
        action = torch.tensor(sample["action"], dtype=torch.long)
        
        return {
            "pixel_values": image,
            "point_goal": point_goal,
            "actions": action
        }

def pointnav_collate_fn(processor_image_func):
    """
    Collate function to process images via CLIP processor and stack tensors.
    """
    def collate(batch):
        images = [item["pixel_values"] for item in batch]
        point_goals = torch.stack([item["point_goal"] for item in batch])
        actions = torch.stack([item["actions"] for item in batch])
        
        # We only use the image processing part of the CLIP processor
        pixel_values = processor_image_func(images=images, return_tensors="pt")["pixel_values"]
        
        return {
            "pixel_values": pixel_values,
            "point_goal": point_goals,
            "actions": actions
        }
    return collate


class ObjectNavDataset(Dataset):
    """
    Dataset loader for ObjectNav imitation learning.
    Expects jsonl format where each line has:
    - image: path to image frame
    - object_id: integer representing the target category
    - action: integer action ID
    """
    def __init__(self, jsonl_path, image_root=None, transform=None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image"]
        if self.image_root:
            img_path = os.path.join(self.image_root, img_path)
            
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        object_id = torch.tensor(sample["object_id"], dtype=torch.long)
        action = torch.tensor(sample["action"], dtype=torch.long)
        
        return {
            "pixel_values": image,
            "object_id": object_id,
            "actions": action
        }

def objectnav_collate_fn(processor_image_func):
    """
    Collate function for ObjectNav data.
    """
    def collate(batch):
        images = [item["pixel_values"] for item in batch]
        object_ids = torch.stack([item["object_id"] for item in batch])
        actions = torch.stack([item["actions"] for item in batch])
        
        pixel_values = processor_image_func(images=images, return_tensors="pt")["pixel_values"]
        
        return {
            "pixel_values": pixel_values,
            "object_id": object_ids,
            "actions": actions
        }
    return collate
