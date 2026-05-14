import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class PointNavPolicy(nn.Module):
    """
    Policy for Point-Goal Navigation (PointNav).
    - Vision encoder: CLIP vision transformer
    - Goal encoder: MLP for relative 2D coordinates (dx, dy) or (r, theta)
    - Fusion: Concatenation of vision and goal features
    - Policy head: Logits over 4 discrete actions
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        fusion_dim: int = 512,
        num_actions: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        
        vision_hidden = self.vision_encoder.config.hidden_size
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.goal_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_actions)
        )
        
    def freeze_vision(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, point_goal: torch.Tensor):
        """
        pixel_values: [B, C, H, W]
        point_goal: [B, 2] (relative coordinates to goal)
        """
        vision_out = self.vision_encoder(pixel_values=pixel_values)
        # Use the CLS token pooler output
        vision_feat = self.vision_proj(vision_out.pooler_output)
        
        goal_feat = self.goal_proj(point_goal)
        
        # Fuse via concatenation
        fused = torch.cat([vision_feat, goal_feat], dim=-1)
        
        logits = self.policy_head(fused)
        
        return {
            "logits": logits,
            "pooled": fused,
        }
