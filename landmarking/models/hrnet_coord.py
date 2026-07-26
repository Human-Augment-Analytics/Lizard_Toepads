"""HRNetLandmarkModel — cross-attention coordinate regression.

Uses HRNet multi-scale features with learned landmark queries,
cross-attention, and self-attention for direct coordinate prediction.
"""

import torch
from torch import nn
import timm

from .registry import register_model


@register_model("hrnet_coord")
class HRNetLandmarkModel(nn.Module):
    def __init__(
        self,
        num_landmarks: int,
        pretrained: bool = True,
        num_heads: int = 8,
        input_size: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True,
        )

        stage_channels = self.backbone.feature_info.channels()
        fuse_channels = stage_channels[1] + stage_channels[2] + stage_channels[3]

        self.feature_dim = 256

        self.feature_proj = nn.Sequential(
            nn.Linear(fuse_channels, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )

        self.landmark_queries = nn.Parameter(
            torch.randn(num_landmarks, self.feature_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.feature_dim)
        self.cross_attn_ff_norm = nn.LayerNorm(self.feature_dim)
        self.cross_attn_ff = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(self.feature_dim)
        self.self_attn_ff_norm = nn.LayerNorm(self.feature_dim)
        self.self_attn_ff = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )

        self.coord_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        all_feats = self.backbone(x)

        f1 = all_feats[1]
        f2 = all_feats[2]
        f3 = all_feats[3]

        H1, W1 = f1.shape[2], f1.shape[3]
        f2_up = nn.functional.interpolate(f2, size=(H1, W1), mode="bilinear", align_corners=False)
        f3_up = nn.functional.interpolate(f3, size=(H1, W1), mode="bilinear", align_corners=False)

        fused = torch.cat([f1, f2_up, f3_up], dim=1)

        B, C, H, W = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = self.feature_proj(tokens)

        queries = self.landmark_queries.unsqueeze(0).repeat(B, 1, 1)

        # Cross-attention block
        q_norm = self.cross_attn_norm(queries)
        attn_out, _ = self.cross_attn(q_norm, tokens, tokens)
        queries = queries + attn_out
        queries = queries + self.cross_attn_ff(self.cross_attn_ff_norm(queries))

        # Self-attention block
        q_norm = self.self_attn_norm(queries)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + attn_out
        queries = queries + self.self_attn_ff(self.self_attn_ff_norm(queries))

        coords = torch.sigmoid(self.coord_head(queries))
        return coords
