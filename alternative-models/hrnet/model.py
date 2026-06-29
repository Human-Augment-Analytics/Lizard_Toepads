import torch
from torch import nn
import timm

NUM_LANDMARKS = 9
INPUT_SIZE = 512


class HRNetLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=NUM_LANDMARKS, pretrained=True, num_heads=8):
        super().__init__()

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True
        )

        # Use stages 1,2,3 (36, 72, 144 channels) — richer spatial detail than
        # stage 4 alone. Stage 0 (18ch) is too low-level.
        stage_channels = self.backbone.feature_info.channels()  # [18, 36, 72, 144]
        fuse_channels = stage_channels[1] + stage_channels[2] + stage_channels[3]  # 252

        # Project fused multi-scale features to a common dim divisible by num_heads
        # 256 is cleanly divisible by 8
        self.feature_dim = 256
        self.num_landmarks = num_landmarks

        self.feature_proj = nn.Sequential(
            nn.Linear(fuse_channels, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )

        # Initialize queries at normal scale so LayerNorm preserves
        # meaningful variance between landmark queries
        self.landmark_queries = nn.Parameter(
            torch.randn(num_landmarks, self.feature_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
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
            batch_first=True
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
            nn.Linear(128, 2)
        )

    def forward(self, x):
        all_feats = self.backbone(x)  # list of [B, C_i, H_i, W_i]

        # Use stages 1, 2, 3 — upsample all to stage 1 spatial size (H1, W1)
        f1 = all_feats[1]  # [B, 36,  H1, W1]
        f2 = all_feats[2]  # [B, 72,  H2, W2]
        f3 = all_feats[3]  # [B, 144, H3, W3]

        H1, W1 = f1.shape[2], f1.shape[3]

        f2_up = nn.functional.interpolate(f2, size=(H1, W1), mode='bilinear', align_corners=False)
        f3_up = nn.functional.interpolate(f3, size=(H1, W1), mode='bilinear', align_corners=False)

        # Concatenate along channel dim → [B, 252, H1, W1]
        fused = torch.cat([f1, f2_up, f3_up], dim=1)

        # Flatten spatial → tokens [B, H1*W1, 252]
        B, C, H, W = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)  # [B, H*W, 252]

        # Project to feature_dim [B, H*W, 256]
        tokens = self.feature_proj(tokens)

        # Landmark queries [B, 9, 256]
        queries = self.landmark_queries.unsqueeze(0).repeat(B, 1, 1)

        # Cross-attention block (pre-norm, residual, feedforward)
        q_norm = self.cross_attn_norm(queries)
        attn_out, _ = self.cross_attn(q_norm, tokens, tokens)
        queries = queries + attn_out
        queries = queries + self.cross_attn_ff(self.cross_attn_ff_norm(queries))

        # Self-attention block (pre-norm, residual, feedforward)
        q_norm = self.self_attn_norm(queries)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + attn_out
        queries = queries + self.self_attn_ff(self.self_attn_ff_norm(queries))

        coords = torch.sigmoid(self.coord_head(queries))

        return coords
