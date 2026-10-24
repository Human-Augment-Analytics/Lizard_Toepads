"""LiteShape — single-resolution, shape-constrained landmark regressor.

Purpose (lizard-specific, evidence-driven)
------------------------------------------
This model is the surviving conclusion of a long diagnostic. On the lizard
toepad data we established, with measurements rather than assumptions:

  - Input resolution is NOT the bottleneck. The median preprocessed crop long
    side is ~497px fit into a 512 canvas (0.97x), so 512 is already ~native;
    raising input_size mostly upsamples. (lizard_resolution_report)
  - Decode resolution is NOT the bottleneck. SimCC at higher bin counts did not
    improve and did not vary monotonically with resolution.
  - Multi-scale fusion HURT. Adding HRNet-style cross-scale fusion made the
    GCN family worse on lizard, i.e. the coarse branches carry blur, not signal,
    for a near-single-scale contour crop.
  - Occlusion is absent and pose shift is minimal, so the invariance machinery a
    multi-branch backbone pays for is largely unused here.

Together these say: a heavy multi-resolution backbone spends compute on an
inductive bias mismatched to this dataset, for no accuracy gain over ~6-7px.

So LiteShape deliberately does the opposite of HRNet-GCN:

  1. A SINGLE-resolution backbone (one timm feature stage, no 4-branch fusion),
     fed at 512. Far fewer FLOPs than hrnet_w18's parallel multi-res streams.
  2. A landmark-query attention head (cross-attn into the feature map + self-attn
     across the 9 queries). This is the GCN-free relational mechanism; it reuses
     the pattern already proven in the `hrnet_coord` variant.
  3. An OPTIONAL differentiable shape projection (the existing ShapeProjection
     ASM/PCA module) as the output constraint. The measured oracle bound says a
     k=6 projection floors at ~2.6px (finger) / ~2.1px (toe) vs ~6.5px
     unconstrained, and — structurally — a globally-coupled shape reconstruction
     cannot place a single landmark in the wrong region, which attacks the
     catastrophic-misprediction tail directly.

Two-step experiment this enables
---------------------------------
  (a) projection OFF: does the lightweight single-scale backbone MATCH
      HRNet-GCN's 6-7px at a fraction of the compute? (the efficiency claim)
  (b) projection ON:  does the shape constraint then push BELOW 6-7px toward
      the oracle bound, and suppress catastrophic failures? (the accuracy claim)

The projection is optional on purpose: no shape-basis .npz is required to run
(a), so the efficiency result is never blocked on generating a basis.

Forward contract
----------------
    forward(x) -> (coords_raw, coords_out)

Both (B, N, 2) in [0, 1]. `coords_raw` is the head's direct prediction;
`coords_out` is the shape-projected prediction when a basis is configured, else
identical to `coords_raw`. The engine supervises `coords_out` and (with a small
weight) `coords_raw`, so the encoder still gets a direct coordinate gradient even
when the projection is active.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import timm

from .registry import register_model
from .shape_projection import ShapeProjection, load_shape_basis


@register_model("lite_shape")
class LiteShape(nn.Module):
    """Single-resolution backbone + query-attention head + optional shape projection.

    Args:
        num_landmarks: Number of landmarks (required).
        backbone: timm backbone name. Default efficientnet_b0 (~3.6M params,
            ~11x smaller than hrnet_w18's 10.95M) — the single-resolution,
            genuinely-lighter extractor the efficiency claim depends on. (Note
            resnet34 is ~21M and would DEFEAT the compute goal; measured, not
            assumed.) Other sane lightweight options: convnext_femto (~4.8M),
            mobilenetv3_large_100 (~3.0M).
        pretrained: Load ImageNet weights for the backbone.
        feature_index: Which backbone feature stage to use as the single
            resolution. Higher index = coarser/stronger-semantics; lower =
            finer/more-local. Default -2 is a mid/high-res stage, a good
            precision/semantics tradeoff for a contour crop.
        feature_dim: Working width of the attention head.
        num_heads: Attention heads.
        input_size: Expected square input size (for reference only).
        shape_basis_path: Optional .npz from shape_bound_report --save-basis.
            If empty, the shape projection is disabled (pure coord regressor).
        shape_basis_group / shape_basis_variant / shape_components /
        shape_n_sigma / shape_blend / shape_fit_iters: forwarded to
            ShapeProjection when a basis is given.
    """

    def __init__(
        self,
        num_landmarks: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        feature_index: int = -2,
        feature_dim: int = 256,
        num_heads: int = 8,
        input_size: int = 512,
        shape_basis_path: str = "",
        shape_basis_group: str = "all",
        shape_basis_variant: str = "similarity",
        shape_components: int = 6,
        shape_n_sigma: float = 3.0,
        shape_blend: float = 1.0,
        shape_fit_iters: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.feature_index = feature_index
        self.feature_dim = feature_dim

        # Single-resolution backbone. features_only gives a list of stage maps;
        # we take exactly ONE of them — no cross-scale fusion. This is the whole
        # point: one resolution, chosen by feature_index.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        feat_channels = self.backbone.feature_info.channels()
        # Normalize a possibly-negative index to an absolute one for clarity.
        self._feat_idx = feature_index if feature_index >= 0 else len(feat_channels) + feature_index
        in_ch = feat_channels[self._feat_idx]

        self.feature_proj = nn.Sequential(
            nn.Linear(in_ch, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        # Learned per-landmark queries (identity embeddings).
        self.landmark_queries = nn.Parameter(torch.randn(num_landmarks, feature_dim))

        # Cross-attention: queries read the single-scale feature map.
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.cross_ff_norm = nn.LayerNorm(feature_dim)
        self.cross_ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # Self-attention: the GCN-free relational prior across the 9 landmarks.
        self.self_attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.self_norm = nn.LayerNorm(feature_dim)
        self.self_ff_norm = nn.LayerNorm(feature_dim)
        self.self_ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        self.coord_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128), nn.GELU(),
            nn.Linear(128, 2),
        )

        # Optional shape projection (reuses the existing ASM/PCA module).
        self.projection = None
        if shape_basis_path:
            basis = load_shape_basis(
                shape_basis_path, group=shape_basis_group, variant=shape_basis_variant
            )
            if basis["mean"].shape[0] != num_landmarks:
                raise ValueError(
                    f"shape basis has {basis['mean'].shape[0]} landmarks, model "
                    f"expects {num_landmarks}"
                )
            self.projection = ShapeProjection(
                mean=basis["mean"],
                components=basis["components"],
                explained_variance=basis["explained_variance"],
                num_components=shape_components,
                n_sigma=shape_n_sigma,
                allow_rotation=shape_basis_variant == "similarity",
                blend=shape_blend,
                n_fit_iter=shape_fit_iters,
            )

    def forward(self, x):
        feats = self.backbone(x)
        fmap = feats[self._feat_idx]  # (B, C, H, W) — one resolution only

        B, C, H, W = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.feature_proj(tokens)

        q = self.landmark_queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention block (pre-norm residual).
        qn = self.cross_norm(q)
        attn, _ = self.cross_attn(qn, tokens, tokens)
        q = q + attn
        q = q + self.cross_ff(self.cross_ff_norm(q))

        # Self-attention block across landmarks (pre-norm residual).
        qn = self.self_norm(q)
        attn, _ = self.self_attn(qn, qn, qn)
        q = q + attn
        q = q + self.self_ff(self.self_ff_norm(q))

        coords_raw = torch.sigmoid(self.coord_head(q))  # (B, N, 2) in [0,1]

        if self.projection is not None:
            coords_out = self.projection(coords_raw)
        else:
            coords_out = coords_raw
        return coords_raw, coords_out
