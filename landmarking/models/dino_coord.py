"""DinoCoord — frozen DINOv2 (ViT) backbone + landmark-query attention head.

Why DINOv2 over HRNet for lizard toepads (the justification)
------------------------------------------------------------
The lizard toepad localization problem, characterized from measurements this
project accumulated, is:

  - LOW local detail: the toepad edge is smooth and texture-poor. Fine
    high-resolution local features (HRNet's whole reason to exist) carry little
    signal here, and empirically adding HRNet multi-scale fusion made accuracy
    WORSE. Resolution is not the bottleneck (crops are ~0.97x native at 512;
    SimCC at higher readout resolution did not help).
  - HIGH global detail: correct placement depends on the overall toepad shape.
  - NEIGHBOR-AWARE: a point's location depends on where the others are.
  - NEAR-CONTOUR / ON-TOE: points must respect the toe foreground boundary.

HRNet is matched to the opposite profile (local-detail-rich, e.g. faces). The
backbone that fits THIS profile is a self-supervised ViT:

  - DINOv2 features have global receptive field from layer one (high global
    detail) and strong emergent PART / FOREGROUND structure without labels,
    which maps onto "which pixels are on the toe" (near-contour) and "this is
    the valley vs the tip" (neighbor-aware semantics).
  - Strong dense correspondence across instances suits a fixed 9-landmark
    configuration.
  - Frozen backbone + light head fits the small-annotation regime (~1.2k crops).

Caveats (kept honest)
---------------------
  - DINOv2 patch stride is 14px -> a coarse feature grid. That is a precision
    liability in general, but precision/resolution is NOT this task's bottleneck
    (established), and the query-attention head reads features at continuous
    positions via attention rather than a quantized argmax, so sub-patch
    placement is still recoverable.
  - Natural-image pretraining has a domain gap to toepads; start FROZEN, and
    only unfreeze if frozen underperforms.
  - This is a mechanism-level bet, not a guarantee.

Forward contract
----------------
    forward(x) -> (coords_raw, coords_out)

matching the `lite_shape` contract so the engine/eval dispatch is reused. With
no shape basis, coords_out == coords_raw.
"""

from __future__ import annotations

import torch
from torch import nn
import timm

from .registry import register_model
from .shape_projection import ShapeProjection, load_shape_basis


@register_model("dino_coord")
class DinoCoord(nn.Module):
    """Frozen DINOv2 backbone + query-attention coordinate head.

    Args:
        num_landmarks: Number of landmarks (required).
        backbone: timm DINOv2 model name. Default is the ViT-S/14 register-token
            variant (reg4 reduces feature-map artifacts). ViT-S is the small,
            cheap starting point; bump to _base_ only if S shows promise.
        pretrained: Load DINOv2 self-supervised weights (the whole point).
        freeze_backbone: Keep DINOv2 frozen (default True) — the small-data,
            clean-ablation choice. The input is resized to dino_input inside
            forward, so the stored 512 crops need no preprocessing change.
        dino_input: Square size fed to DINOv2; must be a multiple of patch 14.
            518 (=37*14) is DINOv2's canonical eval size.
        feature_dim: Working width of the attention head.
        num_heads: Attention heads.
        shape_basis_path / shape_*: optional shape projection (see lite_shape).
    """

    def __init__(
        self,
        num_landmarks: int,
        backbone: str = "vit_small_patch14_reg4_dinov2",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dino_input: int = 518,
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
        self.freeze_backbone = freeze_backbone
        if dino_input % 14 != 0:
            raise ValueError(
                f"dino_input must be a multiple of patch size 14, got {dino_input}"
            )
        self.dino_input = dino_input

        # DINOv2 ViT. num_classes=0 drops the classifier; we use forward_features.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        self.embed_dim = self.backbone.embed_dim
        # Prefix tokens = CLS + register tokens; they are NOT spatial and must be
        # stripped before reshaping to a patch grid. Read from the model so this
        # stays correct across reg0 / reg4 variants.
        self.num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self.landmark_queries = nn.Parameter(torch.randn(num_landmarks, feature_dim))

        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.cross_ff_norm = nn.LayerNorm(feature_dim)
        self.cross_ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

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

    def _dino_tokens(self, x):
        """Run DINOv2 and return spatial patch tokens (B, P, embed_dim)."""
        # Resize to DINOv2's patch-aligned input. The stored crops are 512; DINOv2
        # wants a multiple of 14. bilinear is fine — we are not chasing sub-pixel
        # detail here (established as not the bottleneck).
        if x.shape[-1] != self.dino_input or x.shape[-2] != self.dino_input:
            x = nn.functional.interpolate(
                x, size=(self.dino_input, self.dino_input),
                mode="bilinear", align_corners=False,
            )
        feats = self.backbone.forward_features(x)  # (B, num_prefix + P, C)
        return feats[:, self.num_prefix:, :]        # drop CLS + register tokens

    def forward(self, x):
        B = x.shape[0]
        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self._dino_tokens(x)
            tokens = tokens.detach()
        else:
            tokens = self._dino_tokens(x)

        tokens = self.feature_proj(tokens)  # (B, P, feature_dim)

        q = self.landmark_queries.unsqueeze(0).expand(B, -1, -1)

        qn = self.cross_norm(q)
        attn, _ = self.cross_attn(qn, tokens, tokens)
        q = q + attn
        q = q + self.cross_ff(self.cross_ff_norm(q))

        qn = self.self_norm(q)
        attn, _ = self.self_attn(qn, qn, qn)
        q = q + attn
        q = q + self.self_ff(self.self_ff_norm(q))

        coords_raw = torch.sigmoid(self.coord_head(q))

        coords_out = self.projection(coords_raw) if self.projection is not None else coords_raw
        return coords_raw, coords_out

    def train(self, mode: bool = True):
        """Keep a frozen DINOv2 in eval mode regardless of parent mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
