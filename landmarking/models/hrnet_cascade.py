"""HRNetCascade — cascaded heatmap refinement with intermediate supervision.

Reuses the `heatmap` variant's HRNet-W18 backbone + 4-branch fusion, the
framework `decode_coords`, and (in training) `heatmap_loss`, and adds a small
fixed cascade of refinement stages between the fused features and the final
decode:

    image -> HRNet-w18 -> fused map -> feat_proj -> stream x0
      for k in 1..K:
        feature_k, heatmap_k = RefineStage(x_{k-1})
        x_k = x_{k-1} + merge_pred(heatmap_k) + merge_feat(feature_k)   # additive
      coords = decode_coords(heatmap_K)

Design intent: isolate iterative heatmap refinement (per-landmark, local; needs
no relational structure) from the graph/attention/coordinate-iteration it is
usually bundled with. Every stage is supervised (intermediate supervision) and
the feedback merge is ADDITIVE (residual), so a poor refinement stage degrades
gracefully rather than overriding a good early peak.

Imports only torch/timm — never torch_geometric — and builds the timm HRNet
backbone, NOT the stacked-hourglass from-scratch stem.
"""

from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import timm

from .registry import register_model
from .hrnet_heatmap import decode_coords


def _gn_groups(width: int) -> int:
    """Largest group count in {32,16,8,4,2,1} that divides `width` (for GroupNorm)."""
    for g in (32, 16, 8, 4, 2, 1):
        if width % g == 0:
            return g
    return 1


class _RefineStage(nn.Module):
    """One refinement stage: refine the feature stream, emit a heatmap.

    Returns (feature, heatmap) where feature is at the stream width (for the
    feedback merge) and heatmap is per-landmark at the fused resolution.
    """

    def __init__(self, width: int, num_landmarks: int, bn_momentum: float):
        super().__init__()
        # GroupNorm, NOT BatchNorm: these refinement convs are trained from
        # scratch at small batch size, where BN's running stats do not converge
        # and eval() then normalizes with the wrong statistics — producing stable
        # train loss but violently oscillating val error (heatmaps mis-scaled ->
        # argmax jumps). GroupNorm has no running stats, so train and eval behave
        # identically. bn_momentum is accepted for API compatibility but unused.
        groups = _gn_groups(width)
        self.block = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.ReLU(inplace=True),
        )
        self.out_head = nn.Conv2d(width, num_landmarks, kernel_size=1)
        nn.init.normal_(self.out_head.weight, std=0.001)
        nn.init.constant_(self.out_head.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feature = self.block(x)
        heatmap = self.out_head(feature)
        return feature, heatmap


class _MergePair(nn.Module):
    """Feedback merge projections: heatmap -> stream, feature -> stream."""

    def __init__(self, width: int, num_landmarks: int):
        super().__init__()
        self.merge_pred = nn.Conv2d(num_landmarks, width, kernel_size=1)
        self.merge_feat = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: Tensor, heatmap: Tensor, feature: Tensor) -> Tensor:
        return x + self.merge_pred(heatmap) + self.merge_feat(feature)


@register_model("hrnet_cascade")
class HRNetCascade(nn.Module):
    """Cascaded heatmap refinement on HRNet features.

    Args:
        num_landmarks: Number of landmarks (N). Required.
        num_stages: Number of refinement stages (K >= 1).
        shared_weights: When True, one RefineStage + one MergePair are reused for
            every stage (recurrent refinement). When False, each stage has its
            own parameters.
        heatmap_size: Output heatmap spatial resolution.
        decode_mode: Passed to decode_coords ("windowed" default).
        decode_radius: Windowed soft-argmax radius.
        bn_momentum: BatchNorm momentum for the from-scratch refinement convs.
        pretrained: Load pretrained HRNet backbone weights.
        cascade_width: Refinement feature-stream width (D).
    """

    def __init__(
        self,
        num_landmarks: int,
        num_stages: int = 3,
        shared_weights: bool = True,
        heatmap_size: int = 128,
        decode_mode: str = "windowed",
        decode_radius: int = 5,
        bn_momentum: float = 0.1,
        pretrained: bool = True,
        cascade_width: int = 256,
        **kwargs,
    ):
        super().__init__()
        if num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")

        self.num_landmarks = num_landmarks
        self.num_stages = num_stages
        self.shared_weights = shared_weights
        self.heatmap_size = heatmap_size
        self.decode_mode = decode_mode
        self.decode_radius = decode_radius
        self.cascade_width = cascade_width

        # --- Backbone + fusion (reused from HRNetHeatmap) ---
        self.backbone = timm.create_model(
            "hrnet_w18", pretrained=pretrained, features_only=True,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feats = self.backbone(dummy)
            h0, w0 = feats[0].shape[2], feats[0].shape[3]
            fused_dummy = torch.cat([
                feats[0],
                F.interpolate(feats[1], size=(h0, w0), mode="bilinear", align_corners=False),
                F.interpolate(feats[2], size=(h0, w0), mode="bilinear", align_corners=False),
                F.interpolate(feats[3], size=(h0, w0), mode="bilinear", align_corners=False),
            ], dim=1)
            fused_channels = fused_dummy.shape[1]

        # --- Feature projection to the stream width ---
        self.feat_proj = nn.Conv2d(fused_channels, cascade_width, kernel_size=1)

        # --- Refinement stages (shared or independent) ---
        if shared_weights:
            self.stage = _RefineStage(cascade_width, num_landmarks, bn_momentum)
            # One merge pair reused between stages (only needed when K > 1).
            self.merge = _MergePair(cascade_width, num_landmarks) if num_stages > 1 else None
            self.stages = None
            self.merges = None
        else:
            self.stages = nn.ModuleList([
                _RefineStage(cascade_width, num_landmarks, bn_momentum)
                for _ in range(num_stages)
            ])
            self.merges = nn.ModuleList([
                _MergePair(cascade_width, num_landmarks)
                for _ in range(num_stages - 1)
            ])
            self.stage = None
            self.merge = None

    def _fuse(self, x: Tensor) -> Tensor:
        feats = self.backbone(x)
        h, w = feats[0].shape[2], feats[0].shape[3]
        return torch.cat([
            feats[0],
            F.interpolate(feats[1], size=(h, w), mode="bilinear", align_corners=False),
            F.interpolate(feats[2], size=(h, w), mode="bilinear", align_corners=False),
            F.interpolate(feats[3], size=(h, w), mode="bilinear", align_corners=False),
        ], dim=1)

    def _stage_module(self, k: int) -> _RefineStage:
        return self.stage if self.shared_weights else self.stages[k]

    def _merge_module(self, k: int):
        # Merge applied AFTER stage k to prepare stage k+1 (k in [0, K-2]).
        return self.merge if self.shared_weights else self.merges[k]

    def _resize(self, hm: Tensor, h: int, w: int) -> Tensor:
        if self.heatmap_size is not None and h != self.heatmap_size:
            return F.interpolate(
                hm, size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )
        return hm

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """Run the cascade.

        Returns:
            stage_heatmaps: list of K tensors, each (B, N, heatmap_size, heatmap_size).
            coords: (B, N, 2) decoded from the final stage, in [0, 1].
        """
        fused = self._fuse(x)
        h, w = fused.shape[2], fused.shape[3]
        stream = self.feat_proj(fused)

        stage_heatmaps: List[Tensor] = []
        for k in range(self.num_stages):
            feature, heatmap_fused = self._stage_module(k)(stream)  # (B,D,h,w), (B,N,h,w)
            stage_heatmaps.append(self._resize(heatmap_fused, h, w))
            # Additive feedback merge to prepare the next stage (not after last).
            if k < self.num_stages - 1:
                stream = self._merge_module(k)(stream, heatmap_fused, feature)

        coords = decode_coords(
            stage_heatmaps[-1], mode=self.decode_mode, radius=self.decode_radius
        )
        return stage_heatmaps, coords
