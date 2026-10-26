"""DinoHeatmapRefine — DINOv2 + heatmap head + image/neighbor refinement.

Motivation (user's intuition, matched to the residual error modes)
------------------------------------------------------------------
The DINOv2 + HRNet-heatmap head reached ~6.3px median but its error has a TAIL
(mean 7.1 > median 6.3): a few landmarks are placed in roughly the right region
but not the exact point. The two observed residual failure modes are:

  - PERPENDICULAR to the contour ("off the finger"): an IMAGE/foreground cue —
    the true point sits where the toe actually is, which local appearance says.
  - TANGENTIAL along the contour ("slides along the edge"): a NEIGHBOR cue — on a
    smooth edge, where-along-it is pinned by where the adjacent landmarks are.

Neither a plain soft-argmax decode (reads only the heatmap) nor a plain offset
head (reads only local features, per-landmark independently) addresses BOTH. This
head does: it refines each coarse heatmap coordinate with a small bounded offset
predicted from (a) local DINOv2 features sampled AT the coarse point (the image
half) and (b) a self-attention pass across all 9 landmarks (the neighbor half).

Design kept minimal on purpose
------------------------------
The coarse stage is the EXACT `dino_heatmap` (backbone + HRNet heatmap head +
windowed decode), which already places well. Only a light refinement stage is
added on top, so this is a clean probe of "does image+neighbor refinement buy
sub-pixel accuracy," not a rebuild. The offset is bounded (tanh * max_offset) so
refinement cannot undo good placement — it only nudges within a small radius.

Forward contract
----------------
    forward(x) -> (heatmaps, refined_coords)

Identical shape to `dino_heatmap`, so it rides the engine's `_is_heatmap_on_coords`
loss path unchanged: `heatmap_loss` shapes the map (coarse placement) while its
coordinate term now supervises the REFINED coords, pushing gradient through the
refinement stage. coarse_coords are used internally as the refinement anchor.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import timm

from .registry import register_model
from .hrnet_heatmap import decode_coords


@register_model("dino_heatmap_refine")
class DinoHeatmapRefine(nn.Module):
    """DINOv2 + heatmap head + image/neighbor bounded-offset refinement.

    Args mirror `dino_heatmap`, plus:
        refine_dim: width of the refinement stage.
        refine_heads: attention heads for the neighbor self-attention.
        max_offset: bound on the refinement offset as a fraction of the full
            image (normalized coords). 0.05 = +/-5% of the canvas, i.e. a local
            nudge, not a re-placement.
    """

    def __init__(
        self,
        num_landmarks: int,
        backbone: str = "vit_small_patch14_reg4_dinov2",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dino_input: int = 742,
        head_size: int = 128,
        decode_mode: str = "windowed",
        decode_radius: int = 5,
        bn_momentum: float = 0.1,
        refine_dim: int = 128,
        refine_heads: int = 4,
        max_offset: float = 0.05,
        input_size: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.freeze_backbone = freeze_backbone
        if dino_input % 14 != 0:
            raise ValueError(f"dino_input must be a multiple of 14, got {dino_input}")
        self.dino_input = dino_input
        self.head_size = head_size
        self.decode_mode = decode_mode
        self.decode_radius = decode_radius
        self.max_offset = max_offset

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # --- Coarse stage: EXACT HRNet heatmap head -------------------------
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, num_landmarks, kernel_size=1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.constant_(self.head[-1].bias, 0)

        # --- Refinement stage ----------------------------------------------
        # Per-landmark token = [local image feature @ coarse point] + [coord enc]
        # + [landmark identity]. Then a neighbor self-attention pass, then a
        # bounded offset. This is the "look at the image AND the neighbors" head.
        self.feat_proj = nn.Linear(self.embed_dim, refine_dim)
        self.coord_enc = nn.Linear(2, refine_dim)
        self.lm_embed = nn.Embedding(num_landmarks, refine_dim)

        self.neighbor_attn = nn.MultiheadAttention(
            refine_dim, refine_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(refine_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(refine_dim),
            nn.Linear(refine_dim, refine_dim * 2), nn.GELU(),
            nn.Linear(refine_dim * 2, refine_dim),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(refine_dim), nn.Linear(refine_dim, 2)
        )
        # Start near-zero offset so training begins at the good coarse placement,
        # but with full-strength WEIGHT so gradient reaches the refinement feats
        # (the zero-weight init bug learned earlier: zero final weight severs the
        # input-gradient path). Zero bias only.
        nn.init.xavier_uniform_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def _token_grid(self, x):
        if x.shape[-1] != self.dino_input or x.shape[-2] != self.dino_input:
            x = F.interpolate(
                x, size=(self.dino_input, self.dino_input),
                mode="bilinear", align_corners=False,
            )
        feats = self.backbone.forward_features(x)
        if feats.dim() == 4:
            return feats.permute(0, 3, 1, 2).contiguous()
        patch = feats[:, self.num_prefix:, :]
        B, P, C = patch.shape
        g = int(round(P ** 0.5))
        if g * g != P:
            raise RuntimeError(f"patch count {P} not a perfect square")
        return patch.transpose(1, 2).reshape(B, C, g, g)

    def _sample_at(self, fmap, coords):
        """Bilinear-sample the feature map at normalized [0,1] coords.

        fmap: (B, C, H, W); coords: (B, N, 2) in [0,1]. Returns (B, N, C).
        """
        grid = (coords * 2.0 - 1.0).unsqueeze(2)  # (B, N, 1, 2)
        s = F.grid_sample(fmap, grid, align_corners=True, mode="bilinear",
                          padding_mode="border")  # (B, C, N, 1)
        return s.squeeze(-1).permute(0, 2, 1)       # (B, N, C)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                fmap_raw = self._token_grid(x)
            fmap_raw = fmap_raw.detach()
        else:
            fmap_raw = self._token_grid(x)

        # Coarse heatmap + decode (the proven placement stage).
        fmap = fmap_raw
        if fmap.shape[-1] != self.head_size:
            fmap = F.interpolate(
                fmap, size=(self.head_size, self.head_size),
                mode="bilinear", align_corners=False,
            )
        heatmaps = self.head(fmap)
        coarse = decode_coords(heatmaps, mode=self.decode_mode, radius=self.decode_radius)

        # Refinement: sample local features at the coarse point (image half),
        # build per-landmark tokens, one neighbor self-attention pass (neighbor
        # half), bounded offset.
        B, N = coarse.shape[0], coarse.shape[1]
        img_feat = self._sample_at(fmap_raw, coarse)           # (B, N, embed_dim)
        tok = (
            self.feat_proj(img_feat)
            + self.coord_enc(coarse)
            + self.lm_embed(torch.arange(N, device=x.device)).unsqueeze(0)
        )
        a, _ = self.neighbor_attn(self.attn_norm(tok), self.attn_norm(tok), self.attn_norm(tok))
        tok = tok + a
        tok = tok + self.ffn(tok)
        offset = torch.tanh(self.offset_head(tok)) * self.max_offset
        refined = (coarse + offset).clamp(0.0, 1.0)

        return heatmaps, refined

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
