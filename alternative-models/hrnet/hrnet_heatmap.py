"""
HRNet Heatmap Regression — paper-faithful implementation.

Reference: Wang et al., "Deep High-Resolution Representation Learning for
Visual Recognition", CVPR 2019.  Section 3.3 / Appendix A (face alignment).

Architecture:
  - HRNet-W18 backbone (ImageNet pretrained via timm)
  - Uses ONLY the highest-resolution output branch (1/4 input stride, 128x128
    for a 512px input, 18 channels for W18)
  - 1x1 conv head: 18 → num_landmarks heatmap channels
  - Soft-argmax extracts (x, y) coordinates from each heatmap channel

Loss:
  MSE between predicted heatmaps and Gaussian target heatmaps.
  Target Gaussian sigma is typically 2px at heatmap resolution.

Coordinate extraction:
  Soft-argmax (differentiable spatial expectation) over the heatmap.
  Returns coordinates normalised to [0, 1].

Why this is paper-faithful:
  The original HRNet paper does NOT use multi-scale fusion for the prediction
  head — it simply takes the highest-resolution parallel branch output and
  applies a conv head directly. Multi-scale fusion is used internally within
  HRNet's repeated multi-resolution fusion modules, but the output head is
  single-scale.

  The current HRNetLandmarkModel in model.py uses cross-attention with
  landmark queries, which is a different architecture (closer to DETR).
  This file provides the correct paper baseline.
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm


class HRNetHeatmap(nn.Module):
    """Paper-faithful HRNet heatmap regression for landmark detection.

    Args:
        num_landmarks: Number of landmarks to predict (one heatmap per landmark).
        pretrained:    Load ImageNet pretrained weights for the HRNet backbone.
        heatmap_size:  Spatial size of the output heatmap (H = W = heatmap_size).
                       At input_size=512, HRNet-W18's highest-res branch is 128px.
                       Set to None to use the backbone's native output size.
    """

    def __init__(self, num_landmarks: int = 9, pretrained: bool = True,
                 heatmap_size: int = 128):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size

        # HRNet backbone — features_only=True returns all 4 branch outputs.
        # We use index 0: the highest-resolution branch (stride 4, 18ch for W18).
        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True,
        )
        high_res_channels = self.backbone.feature_info[0]["num_chs"]  # 18 for W18

        # Paper head: single 1x1 conv mapping backbone channels → num_landmarks
        self.head = nn.Conv2d(high_res_channels, num_landmarks, kernel_size=1)

        # Initialise head: small weights, negative bias so sigmoid(bias) ≈ 0.01
        # matching the near-zero Gaussian targets at initialisation.
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, -4.6)  # sigmoid(-4.6) ≈ 0.01

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) input images, ImageNet-normalised.

        Returns:
            heatmaps: (B, num_landmarks, heatmap_size, heatmap_size)
            coords:   (B, num_landmarks, 2) soft-argmax coordinates in [0, 1]
        """
        feat_maps = self.backbone(x)
        feat = feat_maps[0]  # highest-resolution branch: (B, 18, H/4, W/4)

        # Optionally resize to target heatmap_size
        if self.heatmap_size is not None and feat.shape[-1] != self.heatmap_size:
            feat = F.interpolate(
                feat, size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False
            )

        heatmaps = self.head(feat)  # (B, num_landmarks, Hm, Wm)

        # Apply sigmoid so predicted heatmap values are in [0, 1], matching the
        # Gaussian targets (peak = 1.0).  Without this, MSE drives logits to -inf
        # (all-zero output), soft-argmax collapses to 0.5, and loss flatlines.
        heatmaps_sigmoid = torch.sigmoid(heatmaps)

        coords = soft_argmax(heatmaps_sigmoid)  # (B, num_landmarks, 2) in [0, 1]

        return heatmaps_sigmoid, coords


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Differentiable spatial expectation (soft-argmax).

    Converts (B, K, H, W) heatmaps to (B, K, 2) normalised [0,1] coordinates
    by computing the expected (x, y) position under the softmax distribution.

    Args:
        heatmaps: (B, K, H, W) raw logits or scores.

    Returns:
        coords: (B, K, 2) with coords[..., 0] = x (col) and coords[..., 1] = y (row),
                both in [0, 1].
    """
    B, K, H, W = heatmaps.shape

    # Softmax over spatial dimensions
    flat = heatmaps.view(B, K, -1)                     # (B, K, H*W)
    weights = F.softmax(flat, dim=-1)                  # (B, K, H*W)
    weights = weights.view(B, K, H, W)

    # Build coordinate grids normalised to [0, 1]
    # x varies along columns (dim=3), y along rows (dim=2)
    device = heatmaps.device
    xs = torch.linspace(0, 1, W, device=device)        # (W,)
    ys = torch.linspace(0, 1, H, device=device)        # (H,)

    # Expected x: sum over rows first, then dot with xs
    x_coords = (weights.sum(dim=2) * xs.view(1, 1, W)).sum(dim=-1)   # (B, K)
    y_coords = (weights.sum(dim=3) * ys.view(1, 1, H)).sum(dim=-1)   # (B, K)

    return torch.stack([x_coords, y_coords], dim=-1)   # (B, K, 2)


def make_gaussian_heatmaps(
    coords_norm: torch.Tensor,
    heatmap_size: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Generate Gaussian target heatmaps from normalised landmark coordinates.

    Args:
        coords_norm: (B, K, 2) landmark coordinates normalised to [0, 1].
        heatmap_size: Spatial size of the output heatmap (square).
        sigma:        Standard deviation of the Gaussian in heatmap pixels.

    Returns:
        heatmaps: (B, K, heatmap_size, heatmap_size) float32 Gaussians,
                  each with peak value 1.0 at the landmark position.
    """
    B, K, _ = coords_norm.shape
    H = W = heatmap_size
    device = coords_norm.device

    # Pixel-space coordinates of landmarks on the heatmap grid
    px = coords_norm[:, :, 0] * (W - 1)   # (B, K)
    py = coords_norm[:, :, 1] * (H - 1)   # (B, K)

    # Build meshgrid
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each

    # Broadcast: (B, K, H, W)
    dx = grid_x.unsqueeze(0).unsqueeze(0) - px.unsqueeze(-1).unsqueeze(-1)
    dy = grid_y.unsqueeze(0).unsqueeze(0) - py.unsqueeze(-1).unsqueeze(-1)

    heatmaps = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    return heatmaps
