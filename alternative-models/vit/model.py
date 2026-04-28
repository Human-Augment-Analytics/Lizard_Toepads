import torch
from torch import nn
import timm

NUM_LANDMARKS = 9

class ViTLandmark(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, NUM_LANDMARKS * 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        coords = torch.sigmoid(self.head(features))
        return coords
