"""HRNet Heatmap Regression — paper-faithful implementation.

Uses all 4 HRNet resolution branches fused at the head, matching the
official HRNetV2 face alignment architecture.
"""

import torch
from torch import nn
import torch.nn.functional as F
import timm

from .registry import register_model


@register_model("heatmap")
class HRNetHeatmap(nn.Module):
    """Paper-faithful HRNet heatmap regression.

    Args:
        num_landmarks: Number of landmarks (required).
        pretrained: Whether to use pretrained backbone weights.
        heatmap_size: Output heatmap spatial resolution.
    """

    def __init__(
        self,
        num_landmarks: int,
        pretrained: bool = True,
        heatmap_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True,
        )

        # Compute fused channel count from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feats = self.backbone(dummy)
            H0, W0 = feats[0].shape[2], feats[0].shape[3]
            fused_dummy = torch.cat([
                feats[0],
                F.interpolate(feats[1], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[2], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[3], size=(H0, W0), mode="bilinear", align_corners=False),
            ], dim=1)
            all_channels = fused_dummy.shape[1]

        self.head = nn.Sequential(
            nn.Conv2d(all_channels, all_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(all_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(all_channels, num_landmarks, kernel_size=1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, x):
        feat_maps = self.backbone(x)
        H, W = feat_maps[0].shape[2], feat_maps[0].shape[3]
        fused = torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H, W), mode="bilinear", align_corners=False),
        ], dim=1)

        heatmaps = self.head(fused)

        if self.heatmap_size is not None and H != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )

        coords = soft_argmax(heatmaps)
        return heatmaps, coords


def hard_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Hard argmax with sub-pixel refinement.

    Args:
        heatmaps: (B, K, H, W) raw logits.

    Returns:
        (B, K, 2) peak locations in [0, 1].
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)
    py = idx // W
    px = idx % W

    px_c = px.clamp(1, W - 2)
    py_c = py.clamp(1, H - 2)

    b_idx = torch.arange(B, device=heatmaps.device).unsqueeze(1).expand(B, K)
    k_idx = torch.arange(K, device=heatmaps.device).unsqueeze(0).expand(B, K)

    def g(r, c):
        return heatmaps[b_idx, k_idx, r.clamp(0, H - 1), c.clamp(0, W - 1)]

    dx = (g(py_c - 1, px_c) - g(py_c - 1, px_c - 2)).sign() * 0.25
    dy = (g(py_c, px_c - 1) - g(py_c - 2, px_c - 1)).sign() * 0.25

    x_refined = (px.float() + dx + 0.5) / W
    y_refined = (py.float() + dy + 0.5) / H

    return torch.stack([x_refined, y_refined], dim=-1)


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Differentiable spatial expectation (soft-argmax).

    Args:
        heatmaps: (B, K, H, W) raw logits.

    Returns:
        (B, K, 2) expected coordinates in [0, 1].
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    weights = F.softmax(flat, dim=-1).view(B, K, H, W)

    device = heatmaps.device
    xs = torch.linspace(0, 1, W, device=device)
    ys = torch.linspace(0, 1, H, device=device)

    x_coords = (weights.sum(dim=2) * xs.view(1, 1, W)).sum(dim=-1)
    y_coords = (weights.sum(dim=3) * ys.view(1, 1, H)).sum(dim=-1)

    return torch.stack([x_coords, y_coords], dim=-1)
