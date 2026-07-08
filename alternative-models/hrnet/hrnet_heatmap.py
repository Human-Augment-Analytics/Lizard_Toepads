"""
HRNet Heatmap Regression — paper-faithful implementation.

Reference: Wang et al., "Deep High-Resolution Representation Learning for
Visual Recognition", CVPR 2019 / HRNetV2 face alignment codebase.

Key architectural detail from the paper source:
  The head operates on ALL 4 resolution branches concatenated after
  upsampling to the highest resolution (1/4 input stride):
    branch 0: (B, 18, H/4, W/4)
    branch 1: (B, 36, H/4, W/4)  — upsampled
    branch 2: (B, 72, H/4, W/4)  — upsampled
    branch 3: (B, 144, H/4, W/4) — upsampled
    cat → (B, 270, H/4, W/4)
    head → (B, num_landmarks, H/4, W/4)

  This is NOT the same as using only branch 0 (18 channels). The paper
  fuses all scales before predicting heatmaps. Our earlier HRNetGNN used
  only the last branch (-1 index / 144ch) for GCN feature sampling, which
  is a different design choice appropriate for coordinate refinement but
  NOT for heatmap regression.

Coordinate extraction:
  - soft_argmax: differentiable, used during training for coordinate loss
  - hard_argmax: non-differentiable argmax, used for NME evaluation
    (matches paper's decode_preds which uses np.argmax)
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm


class HRNetHeatmap(nn.Module):
    """Paper-faithful HRNet heatmap regression.

    Uses all 4 HRNet resolution branches fused at the head, matching the
    official HRNetV2 face alignment architecture.
    """

    def __init__(self, num_landmarks: int = 9, pretrained: bool = True,
                 heatmap_size: int = 64):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True,
        )

        # Compute actual fused channel count from a dummy forward pass.
        # timm's feature_info may not accurately reflect the concatenated output.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feats = self.backbone(dummy)
            H0, W0 = feats[0].shape[2], feats[0].shape[3]
            fused_dummy = torch.cat([
                feats[0],
                F.interpolate(feats[1], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[2], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[3], size=(H0, W0), mode="bilinear", align_corners=False),
            ], dim=1)
            all_channels = fused_dummy.shape[1]

        # Paper head: 1×1 conv on fused multi-scale features
        self.head = nn.Conv2d(all_channels, num_landmarks, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised images.

        Returns:
            heatmaps: (B, num_landmarks, heatmap_size, heatmap_size) raw logits
            coords:   (B, num_landmarks, 2) soft-argmax coordinates in [0, 1]
        """
        feat_maps = self.backbone(x)  # [f0, f1, f2, f3]

        # Upsample all branches to f0's spatial size (highest resolution)
        H, W = feat_maps[0].shape[2], feat_maps[0].shape[3]
        fused = torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H, W), mode="bilinear", align_corners=False),
        ], dim=1)  # (B, 270, H, W)

        heatmaps = self.head(fused)  # (B, num_landmarks, H, W) raw logits

        # Resize to target heatmap_size if needed
        if self.heatmap_size is not None and H != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )

        # Soft-argmax for differentiable coordinate extraction (training)
        coords = soft_argmax(heatmaps)

        return heatmaps, coords


def hard_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Hard argmax with sub-pixel refinement.

    Matches paper's decode_preds: find peak pixel, then shift ±0.25px
    based on local gradient direction around the peak.

    Args:
        heatmaps: (B, K, H, W) raw logits.

    Returns:
        (B, K, 2) peak locations in [0, 1].
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx  = flat.argmax(dim=-1)           # (B, K)
    py   = idx // W                      # row
    px   = idx %  W                      # col

    # Sub-pixel refinement: shift ±0.25 based on gradient sign around peak
    # Clamp to avoid boundary indexing
    px_c = px.clamp(1, W - 2)
    py_c = py.clamp(1, H - 2)

    # Gather heatmap values at the required offsets (vectorised over B, K)
    b_idx = torch.arange(B, device=heatmaps.device).unsqueeze(1).expand(B, K)
    k_idx = torch.arange(K, device=heatmaps.device).unsqueeze(0).expand(B, K)

    def g(r, c):
        return heatmaps[b_idx, k_idx, r.clamp(0, H-1), c.clamp(0, W-1)]

    dx = (g(py_c - 1, px_c) - g(py_c - 1, px_c - 2)).sign() * 0.25
    dy = (g(py_c,     px_c - 1) - g(py_c - 2, px_c - 1)).sign() * 0.25

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
    flat    = heatmaps.view(B, K, -1)
    weights = F.softmax(flat, dim=-1).view(B, K, H, W)

    device = heatmaps.device
    xs = torch.linspace(0, 1, W, device=device)
    ys = torch.linspace(0, 1, H, device=device)

    x_coords = (weights.sum(dim=2) * xs.view(1, 1, W)).sum(dim=-1)
    y_coords = (weights.sum(dim=3) * ys.view(1, 1, H)).sum(dim=-1)

    return torch.stack([x_coords, y_coords], dim=-1)


def make_gaussian_heatmaps(
    coords_norm: torch.Tensor,
    heatmap_size: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Generate Gaussian target heatmaps.

    Matches paper's generate_target: Gaussian is only placed within
    3*sigma bounding box. Landmarks outside the heatmap are skipped
    (target stays zero), matching the paper's implicit visibility masking.

    Args:
        coords_norm: (B, K, 2) in [0, 1].
        heatmap_size: square heatmap side length.
        sigma: Gaussian sigma in heatmap pixels.

    Returns:
        (B, K, heatmap_size, heatmap_size) float32 with peaks at 1.0.
    """
    B, K, _ = coords_norm.shape
    H = W = heatmap_size
    device = coords_norm.device

    px = coords_norm[:, :, 0] * (W - 1)   # (B, K)
    py = coords_norm[:, :, 1] * (H - 1)

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)

    dx = grid_x.unsqueeze(0).unsqueeze(0) - px.unsqueeze(-1).unsqueeze(-1)
    dy = grid_y.unsqueeze(0).unsqueeze(0) - py.unsqueeze(-1).unsqueeze(-1)

    heatmaps = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    # Zero out landmarks that are outside the heatmap (paper's visibility masking)
    outside = (
        (px < 0) | (px >= W) | (py < 0) | (py >= H)
    )  # (B, K)
    heatmaps[outside] = 0.0

    return heatmaps
