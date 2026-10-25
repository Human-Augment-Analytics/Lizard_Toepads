"""DinoHeatmap — DINOv2 (ViT) features + the exact HRNet heatmap head.

Why this model exists
---------------------
Across every lightweight model built for lizard (lite_shape CNN, dino_coord
frozen, dino_coord fine-tuned) the error plateaued at ~10-11px median, while the
two heavy models that reach ~6-7px (fused_global, hrnet heatmap) both use a
HEATMAP output. The one variable never changed in the lightweight tier was the
OUTPUT HEAD: all of them used naive coordinate regression (sigmoid over an MLP),
which is known to localize worse than a spatial heatmap + soft-argmax, matching
the observed failure ("points in the right area, not the exact coordinate").

So this model pairs the best-placement backbone found (DINOv2) with the EXACT
same head mechanism as `HRNetHeatmap`: a 1x1 conv stack producing one heatmap
per landmark, decoded with the identical `decode_coords` (windowed soft-argmax).
It reuses `HRNetHeatmap`'s head and decoder verbatim so this is a clean,
single-variable swap of the backbone under a fixed, proven head — not a new head
design.

The only adaptation: DINOv2 emits patch TOKENS (B, P, C), not a conv map, so the
tokens are reshaped back to a (B, C, Hp, Wp) grid before the conv head. Because
DINOv2 patches are stride-14 (coarse), the token grid is optionally upsampled to
`head_size` before the head so the heatmap — and thus the decoded coordinate —
has enough spatial resolution. The established finding that resolution is not the
dataset bottleneck refers to INPUT crop resolution; the HEAD's output resolution
is exactly what the coordinate-regression models lacked, so giving the head a
real spatial grid is the point.

Forward contract
----------------
    forward(x) -> (heatmaps, coords)

Identical to `HRNetHeatmap`, so the engine's `_is_heatmap_on_coords` loss path
(heatmap CE term + coordinate term, Gaussian targets generated inside
`heatmap_loss`) applies unchanged once the dispatch gate includes this variant.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import timm

from .registry import register_model
from .hrnet_heatmap import decode_coords


@register_model("dino_heatmap")
class DinoHeatmap(nn.Module):
    """DINOv2 backbone + the HRNet heatmap head (1x1 conv -> per-landmark map).

    Args:
        num_landmarks: Number of landmarks (required).
        backbone: timm DINOv2 model name (default ViT-S/14 reg4).
        pretrained: Load DINOv2 self-supervised weights.
        freeze_backbone: Keep DINOv2 frozen (default False — the fine-tuned path;
            the heatmap head is the change under test, and HRNet's 6-7px uses a
            trained backbone, so fine-tune by default for a fair comparison).
        dino_input: Patch-aligned square input to the ViT (multiple of 14).
        head_size: Spatial size the token grid is upsampled to before the conv
            head; also the heatmap resolution fed to decode. Mirrors
            HRNetHeatmap.heatmap_size semantics.
        decode_mode / decode_radius: passed to `decode_coords`, identical to
            HRNetHeatmap ("windowed" soft-argmax, radius in heatmap cells).
        bn_momentum: BatchNorm momentum for the head (0.1 is the small-data
            value used by the lizard heatmap configs).
    """

    def __init__(
        self,
        num_landmarks: int,
        backbone: str = "vit_small_patch14_reg4_dinov2",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dino_input: int = 518,
        head_size: int = 128,
        decode_mode: str = "windowed",
        decode_radius: int = 5,
        bn_momentum: float = 0.1,
        input_size: int = 512,
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
        self.head_size = head_size
        self.decode_mode = decode_mode
        self.decode_radius = decode_radius

        # dynamic_img_size lets the ViT accept an input size other than its
        # baked-in default (518): patch_embed stops asserting a fixed size and the
        # position embeddings are interpolated to the actual grid. Without it the
        # model ONLY runs at exactly 518 and any other dino_input raises.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        # Patch grid side for a square dino_input.
        patch = int(getattr(self.backbone, "patch_embed").patch_size[0]) \
            if hasattr(self.backbone, "patch_embed") else 14
        self.grid = dino_input // patch

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # EXACT HRNet heatmap head: 1x1 conv -> BN -> ReLU -> 1x1 conv to N maps.
        # Input channel count is the ViT embed_dim (what the reshaped token grid
        # carries) rather than HRNet's fused channels, but the head STRUCTURE and
        # init match HRNetHeatmap so the readout mechanism is identical.
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, num_landmarks, kernel_size=1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.constant_(self.head[-1].bias, 0)

    def _token_grid(self, x):
        """DINOv2 tokens -> spatial feature map (B, embed_dim, Hp, Wp)."""
        if x.shape[-1] != self.dino_input or x.shape[-2] != self.dino_input:
            x = F.interpolate(
                x, size=(self.dino_input, self.dino_input),
                mode="bilinear", align_corners=False,
            )
        feats = self.backbone.forward_features(x)
        # With dynamic_img_size, timm may return (B, Hp, Wp, C) or (B, T, C).
        if feats.dim() == 4:
            # NHWC spatial output -> (B, C, Hp, Wp) directly (no prefix tokens).
            return feats.permute(0, 3, 1, 2).contiguous()
        patch = feats[:, self.num_prefix:, :]            # (B, P, C)
        B, P, C = patch.shape
        # Derive the grid from the ACTUAL patch count, not a precomputed value,
        # so a non-square or size-mismatched grid cannot silently transpose.
        g = int(round(P ** 0.5))
        if g * g != P:
            raise RuntimeError(
                f"patch count {P} is not a perfect square; cannot reshape to a "
                f"square grid (dino_input={self.dino_input})."
            )
        # forward_features preserves row-major patch order, so this reconstructs
        # the spatial grid.
        return patch.transpose(1, 2).reshape(B, C, g, g)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                fmap = self._token_grid(x)
            fmap = fmap.detach()
        else:
            fmap = self._token_grid(x)

        # Upsample the coarse (stride-14) token grid to the head resolution so
        # the heatmap has the spatial resolution the coord-regression head lacked.
        if fmap.shape[-1] != self.head_size:
            fmap = F.interpolate(
                fmap, size=(self.head_size, self.head_size),
                mode="bilinear", align_corners=False,
            )

        heatmaps = self.head(fmap)                       # (B, N, head_size, head_size)
        coords = decode_coords(
            heatmaps, mode=self.decode_mode, radius=self.decode_radius
        )
        return heatmaps, coords

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
