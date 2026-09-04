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
        decode_mode: str = "windowed",
        decode_radius: int = 5,
        bn_momentum: float = 0.01,
        use_star: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        self.decode_mode = decode_mode
        self.decode_radius = decode_radius
        self.use_star = use_star

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

        # bn_momentum default 0.01 is the paper-faithful HRNet value and is kept so
        # the working WFLW baseline is unchanged. On SMALL datasets it is harmful:
        # a freshly initialized BN with momentum 0.01 at batch_size 4 needs ~300+
        # steps for its running stats to converge, so eval() normalizes with wrong
        # statistics. Measured 15.5x train-vs-eval logit-scale discrepancy on
        # identical weights after 150 steps. Raise it (0.1) for small-data configs.
        self.head = nn.Sequential(
            nn.Conv2d(all_channels, all_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(all_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(all_channels, num_landmarks, kernel_size=1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.constant_(self.head[-1].bias, 0)

        # --- STAR uncertainty head (Option A) ---
        # Per-landmark, per-cell Cholesky params [log_L11, L21, log_L22], read at
        # the decoded (argmax) cell to form a per-landmark covariance for the STAR
        # coordinate term. Zero-init so the model starts isotropic (Sigma = I),
        # i.e. STAR begins as plain L2 and must earn any anisotropy. Only created
        # when use_star, so the paper-faithful heatmap model is unchanged.
        self.sigma_head = None
        if use_star:
            self.sigma_head = nn.Conv2d(all_channels, 3 * num_landmarks, kernel_size=1)
            nn.init.constant_(self.sigma_head.weight, 0.0)
            nn.init.constant_(self.sigma_head.bias, 0.0)

    def _fuse(self, x):
        """Backbone + 4-branch fusion -> (fused_map, H, W)."""
        feat_maps = self.backbone(x)
        H, W = feat_maps[0].shape[2], feat_maps[0].shape[3]
        fused = torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H, W), mode="bilinear", align_corners=False),
        ], dim=1)
        return fused, H, W

    def forward_star(self, x):
        """Like forward, but also returns the STAR sigma map.

        Returns (heatmaps, coords, sigma) where sigma is (B, 3*N, Hs, Ws) at the
        heatmap resolution. Raises if the model was not built with use_star=True.
        """
        if self.sigma_head is None:
            raise RuntimeError(
                "forward_star requires the model built with use_star=True."
            )
        fused, H, W = self._fuse(x)
        heatmaps = self.head(fused)
        sigma = self.sigma_head(fused)
        if self.heatmap_size is not None and H != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps, size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )
            sigma = F.interpolate(
                sigma, size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )
        coords = decode_coords(
            heatmaps, mode=self.decode_mode, radius=self.decode_radius
        )
        return heatmaps, coords, sigma

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

        coords = decode_coords(
            heatmaps, mode=self.decode_mode, radius=self.decode_radius
        )
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
    """Differentiable spatial expectation (soft-argmax) over the WHOLE map.

    WARNING — severe centre bias at realistic logit scales. softmax over H*W cells
    (16384 at heatmap_size=128) is nearly uniform unless the logit range is large,
    and the expectation of a near-uniform distribution over [0,1] is 0.5. Measured
    on PERFECT Gaussian target maps at heatmap_size=128, mean error in canvas px
    (1024): logit scale 1 -> 265px, scale 5 -> 257px, scale 10 -> 85px,
    scale 15 -> 1.4px. Predicting the image centre for everything scores ~274px.

    So this decoder returns the image centre for any map that is not already very
    sharply peaked, and its accuracy is near-discontinuous in the logit scale.
    Prefer `windowed_soft_argmax` for a differentiable readout, or `hard_argmax`
    where gradients are not needed. Kept for backward compatibility and ablations.
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


def windowed_soft_argmax(heatmaps: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """Soft-argmax restricted to a window around each landmark's peak.

    Locates the peak with argmax (the index is DETACHED, so no gradient flows
    through the discrete selection), then takes the spatial expectation only over
    a (2*radius+1)^2 window around it. This keeps the readout differentiable with
    respect to heatmap VALUES while removing the centre bias caused by thousands
    of irrelevant cells dragging the global expectation toward 0.5.

    Measured on perfect Gaussian targets at heatmap_size=128 (canvas px @1024):
    6.7px at logit scale 0.1 and 0.5px at scale 10, versus 265px and 85px for the
    global version. Gradient flow verified.

    Args:
        heatmaps: (B, K, H, W) raw logits.
        radius: Half-width of the window in heatmap cells.

    Returns:
        (B, K, 2) expected coordinates in [0, 1].
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1).detach()
    cy = (idx // W).float().view(B, K, 1, 1)
    cx = (idx % W).float().view(B, K, 1, 1)

    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    inside = ((yy - cy).abs() <= radius) & ((xx - cx).abs() <= radius)

    # The peak cell is always inside the window, so no row is fully masked and the
    # softmax is well defined.
    masked = heatmaps.masked_fill(~inside, float("-inf"))
    weights = F.softmax(masked.view(B, K, -1), dim=-1).view(B, K, H, W)

    xs = torch.linspace(0, 1, W, device=device)
    ys = torch.linspace(0, 1, H, device=device)
    x_coords = (weights.sum(dim=2) * xs.view(1, 1, W)).sum(dim=-1)
    y_coords = (weights.sum(dim=3) * ys.view(1, 1, H)).sum(dim=-1)

    return torch.stack([x_coords, y_coords], dim=-1)


def decode_coords(
    heatmaps: torch.Tensor, mode: str = "windowed", radius: int = 5
) -> torch.Tensor:
    """Decode coordinates from heatmap logits.

    Args:
        heatmaps: (B, K, H, W) raw logits.
        mode: "windowed" (default, differentiable + unbiased), "global"
            (legacy soft-argmax, centre-biased), or "hard" (argmax with sub-pixel
            refinement, not differentiable).
        radius: Window half-width for "windowed".

    Returns:
        (B, K, 2) coordinates in [0, 1].
    """
    if mode == "windowed":
        return windowed_soft_argmax(heatmaps, radius=radius)
    if mode == "global":
        return soft_argmax(heatmaps)
    if mode == "hard":
        return hard_argmax(heatmaps)
    raise ValueError(
        f"Unknown decode mode {mode!r}; expected 'windowed', 'global', or 'hard'."
    )
