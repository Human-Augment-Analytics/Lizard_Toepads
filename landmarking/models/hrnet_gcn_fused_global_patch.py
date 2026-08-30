"""HRNetGNN_FusedGlobal_Patch — fused_global with multi-offset patch sampling.

Identical to `fused_global` (iterative GCN refinement + coordinate/global/
landmark-identity embeddings) EXCEPT the image-feature component samples a small
SET OF OFFSETS around each landmark's current estimate instead of a single point,
then aggregates them with a layout-preserving learned projection.

Motivation (capture radius / convergence basin)
------------------------------------------------
The refiner predicts deltas: "which way is the true landmark from here?" A single
bilinear point sample answers "what is here," not "which way to go." If the current
estimate is off, the point feature gives no directional basis and the error can
compound across iterations (each iteration re-samples the new wrong location).

Sampling multiple offsets gives a *gradient of context*: fine offsets supply local
precision, coarse (dilated) offsets supply long-range directionality that can rescue
a bad initialization. We PRESERVE the offset layout (a learned linear over the
concatenated-in-fixed-order offset features) rather than pooling, because the
directional signal lives in the arrangement — mean-pooling would destroy it.

To avoid the overfitting that sank the naive multi-scale concat (fused_global_ms on
150-image cephalometric), the per-offset features are first projected DOWN per
offset, then combined, so the node-feature dim entering the GCN stays == gnn_hidden
regardless of the number of offsets.

Offset patterns (config: patch_mode)
-------------------------------------
- "point":       single (0,0) offset — reduces to fused_global (sanity/ablation).
- "dense":       (2r+1)x(2r+1) grid at step `patch_step` px (default 5x5).
- "multiradius": center + rings at radii in `patch_radii` px, 4 or 8 dirs each.
Offsets are expressed in FEATURE-MAP pixels and converted to normalized grid units
using the fused map's (H, W), so the physical capture radius scales with input_size.

Note: the finest spatial resolution is still input/4 (branch-0). This widens the
capture radius / basin; it does not raise the sub-cell precision ceiling.
"""

import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

from .registry import register_model

COORD_EMBED_DIM = 16
LANDMARK_EMBED_DIM = 32
GLOBAL_EMBED_DIM = 64


def _build_offsets(patch_mode: str, patch_step: int, patch_radius: int, patch_radii):
    """Return a list of (dy, dx) integer offsets in feature-map pixels.

    The center (0, 0) is always included and placed first.
    """
    if patch_mode == "point":
        return [(0, 0)]

    if patch_mode == "dense":
        r = patch_radius
        offs = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offs.append((dy * patch_step, dx * patch_step))
        # Ensure center first
        offs.remove((0, 0))
        return [(0, 0)] + offs

    if patch_mode == "multiradius":
        offs = [(0, 0)]
        # 8-connected ring directions
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for rad in patch_radii:
            for dy, dx in dirs:
                offs.append((dy * rad, dx * rad))
        return offs

    raise ValueError(f"Unknown patch_mode '{patch_mode}'")


@register_model("fused_global_patch")
class HRNetGNN_FusedGlobal_Patch(nn.Module):
    """fused_global with multi-offset (patch) sampling and layout-preserving proj.

    Args:
        num_landmarks: Number of landmarks (required).
        hrnet_backbone: Backbone name (kept for parity; hrnet_w18).
        feat_dim: Unused (kept for config/signature parity).
        gnn_hidden: GCN hidden dim and node feature dim.
        num_layers: Number of GCNConv layers per refinement iteration.
        num_iters: Number of coordinate refinement iterations.
        patch_mode: "point" | "dense" | "multiradius".
        patch_step: Grid step in feature-map px for "dense".
        patch_radius: Grid half-size for "dense" (k = 2*radius+1).
        patch_radii: Iterable of ring radii in feature-map px for "multiradius".
        patch_proj_dim: Per-offset projected dim before layout-preserving combine.
    """

    def __init__(
        self,
        num_landmarks: int,
        hrnet_backbone: str = "hrnet_w18",
        feat_dim: int = 64,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        num_iters: int = 3,
        patch_mode: str = "multiradius",
        patch_step: int = 1,
        patch_radius: int = 2,
        patch_radii=(2, 6, 14),
        patch_proj_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters
        self.patch_mode = patch_mode

        self.offsets = _build_offsets(patch_mode, patch_step, patch_radius, patch_radii)
        self.num_offsets = len(self.offsets)
        # Register offsets as a (num_offsets, 2) buffer in (dy, dx) feature-px.
        self.register_buffer(
            "offset_grid",
            torch.tensor(self.offsets, dtype=torch.float32),
            persistent=False,
        )

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True,
        )

        # Fused channel count from a dummy forward pass.
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
            self.fused_channels = fused_dummy.shape[1]

        # Per-offset down-projection (shared across offsets), then a
        # layout-preserving linear over the concatenated per-offset vectors so the
        # combined image-feature dim is fixed regardless of num_offsets.
        self.per_offset_proj = nn.Linear(self.fused_channels, patch_proj_dim)
        self.patch_combine = nn.Linear(self.num_offsets * patch_proj_dim, self.fused_channels)

        self.coord_embed = nn.Linear(2, COORD_EMBED_DIM)
        self.global_proj = nn.Linear(self.fused_channels, GLOBAL_EMBED_DIM)
        self.landmark_embed = nn.Embedding(num_landmarks, LANDMARK_EMBED_DIM)

        total_in = self.fused_channels + COORD_EMBED_DIM + GLOBAL_EMBED_DIM + LANDMARK_EMBED_DIM
        self.node_feat_proj = nn.Linear(total_in, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def get_fused_map(self, x):
        feat_maps = self.backbone(x)
        H0, W0 = feat_maps[0].shape[2], feat_maps[0].shape[3]
        return torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H0, W0), mode="bilinear", align_corners=False),
        ], dim=1)

    def sample_patch(self, feat_map, coords):
        """Sample all offsets around each landmark, layout-preserving combine.

        Args:
            feat_map: (B, C, H, W) fused feature map.
            coords: (B, N, 2) normalized [0,1] coords (x, y).

        Returns:
            (B, N, C) combined image feature per landmark.
        """
        B, C, H, W = feat_map.shape
        N = coords.shape[1]
        O = self.num_offsets

        # Convert (dy, dx) feature-px offsets to normalized [-1,1] grid deltas.
        # grid_sample expects (x, y) in [-1, 1]; step in normalized units = 2/(W-1), 2/(H-1).
        off = self.offset_grid.to(feat_map.device)  # (O, 2) as (dy, dx)
        dx_norm = off[:, 1] * (2.0 / max(W - 1, 1))  # (O,)
        dy_norm = off[:, 0] * (2.0 / max(H - 1, 1))  # (O,)
        delta = torch.stack([dx_norm, dy_norm], dim=-1)  # (O, 2) as (x, y)

        base = coords * 2.0 - 1.0  # (B, N, 2) in [-1,1], (x, y)
        # (B, N, O, 2) = base broadcast + delta broadcast
        grid = base.unsqueeze(2) + delta.view(1, 1, O, 2)
        # grid_sample grid shape (B, H_out, W_out, 2): use (N, O) as the output plane.
        sampled = F.grid_sample(
            feat_map, grid, align_corners=True, mode="bilinear",
        )  # (B, C, N, O)
        sampled = sampled.permute(0, 2, 3, 1)  # (B, N, O, C)

        # Per-offset down-projection (shared), keep offset order, then combine.
        proj = F.relu(self.per_offset_proj(sampled))  # (B, N, O, patch_proj_dim)
        proj = proj.reshape(B, N, O * proj.shape[-1])  # (B, N, O*proj_dim) layout preserved
        combined = self.patch_combine(proj)  # (B, N, C)
        return combined

    def forward(self, x, initial_coords, edge_index):
        fused_map = self.get_fused_map(x)
        coords = initial_coords.clone()
        B = x.shape[0]
        N = self.num_landmarks

        global_feat = fused_map.mean(dim=[2, 3])
        global_emb = F.relu(self.global_proj(global_feat))
        global_emb = global_emb.unsqueeze(1).expand(-1, N, -1)

        lm_ids = torch.arange(N, device=x.device)
        lm_emb = self.landmark_embed(lm_ids).unsqueeze(0).expand(B, -1, -1)

        for _ in range(self.num_iters):
            img_feats = self.sample_patch(fused_map, coords)  # (B, N, C)
            coord_emb = F.relu(self.coord_embed(coords))

            node_feats = torch.cat([img_feats, coord_emb, global_emb, lm_emb], dim=-1)
            node_feats = F.relu(self.node_feat_proj(node_feats))

            node_feats_flat = node_feats.view(B * N, -1)
            batch_edge_index = torch.cat(
                [edge_index + b * N for b in range(B)], dim=1
            )

            h = node_feats_flat
            for layer in self.gnn_layers:
                h = layer(h, batch_edge_index)
                h = F.relu(h)

            delta = self.delta_head(h).view(B, N, 2)
            coords = torch.clamp(coords + delta, 0.0, 1.0)

        return coords
