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

        self.feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_landmarks = num_landmarks

        self.landmark_queries = nn.Parameter(
            torch.randn(num_landmarks, self.feature_dim) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.coord_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        B, C, H, W = feats.shape

        tokens = feats.flatten(2).transpose(1, 2)

        queries = self.landmark_queries.unsqueeze(0).expand(B, -1, -1)

        queries, _ = self.cross_attn(
            queries,
            tokens,
            tokens
        )

        queries, _ = self.self_attn(
            queries,
            queries,
            queries
        )

        coords = torch.sigmoid(self.coord_head(queries))

        return coords
