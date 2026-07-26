"""ViT Landmark Model — Vision Transformer with MLP coordinate head."""

import torch
from torch import nn
import timm

from .registry import register_model


@register_model("vit")
class ViTLandmark(nn.Module):
    """ViT-based landmark detection via direct coordinate regression.

    Args:
        num_landmarks: Number of landmarks (required).
        pretrained: Whether to use pretrained ViT weights.
        input_size: Expected input image size (224 for ViT).
    """

    def __init__(self, num_landmarks: int, pretrained: bool = True,
                 input_size: int = 224, **kwargs):
        super().__init__()
        self.num_landmarks = num_landmarks

        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=pretrained
        )
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, num_landmarks * 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        coords = torch.sigmoid(self.head(features))
        return coords.view(x.shape[0], self.num_landmarks, 2)
