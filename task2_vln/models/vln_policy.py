from typing import Dict

import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPCrossAttentionPolicy(nn.Module):
    """CLIP-based VLN policy.

    - Vision encoder: CLIP vision transformer
    - Text encoder: CLIP text transformer
    - Fusion: text queries attending to image tokens
    - Policy head: logits over 4 discrete actions
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_fusion_layers: int = 2,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        text_hidden = self.clip.text_model.config.hidden_size
        vision_hidden = self.clip.vision_model.config.hidden_size

        self.text_proj = nn.Linear(text_hidden, fusion_dim)
        self.vision_proj = nn.Linear(vision_hidden, fusion_dim)

        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(max(1, num_fusion_layers))
            ]
        )
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in range(max(1, num_fusion_layers))])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim * 2, fusion_dim),
                )
                for _ in range(max(1, num_fusion_layers))
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in range(max(1, num_fusion_layers))])
        self.dropout = nn.Dropout(dropout)

        self.policy_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_actions),
        )

    def freeze_clip(
        self,
        unfreeze_last_text_layers: int = 0,
        unfreeze_last_vision_layers: int = 0,
        unfreeze_projection: bool = False,
    ) -> None:
        for p in self.clip.parameters():
            p.requires_grad = False

        if unfreeze_last_text_layers > 0:
            text_layers = self.clip.text_model.encoder.layers
            for layer in text_layers[-unfreeze_last_text_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        if unfreeze_last_vision_layers > 0:
            vision_layers = self.clip.vision_model.encoder.layers
            for layer in vision_layers[-unfreeze_last_vision_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        if unfreeze_projection:
            for p in self.clip.text_projection.parameters():
                p.requires_grad = True
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = True

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T]
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vision_out = self.clip.vision_model(pixel_values=pixel_values)

        text_seq = self.text_proj(text_out.last_hidden_state)
        vision_seq = self.vision_proj(vision_out.last_hidden_state)

        fused_text = text_seq
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.cross_attn_layers,
            self.cross_attn_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            attn_out, _ = attn(query=fused_text, key=vision_seq, value=vision_seq, need_weights=False)
            fused_text = attn_norm(fused_text + self.dropout(attn_out))
            fused_text = ffn_norm(fused_text + self.dropout(ffn(fused_text)))
        pooled = self._masked_mean(fused_text, attention_mask)

        logits = self.policy_head(pooled)
        return {
            "logits": logits,
            "pooled": pooled,
        }
