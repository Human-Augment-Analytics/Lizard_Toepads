"""HRNetGNN_FusedGlobal_MS — fused_global with explicit per-scale sampling.

Identical to `fused_global` (iterative GCN refinement + coordinate/global/
landmark-identity embeddings) EXCEPT the image-feature component samples each
HRNet branch at its NATIVE resolution and concatenates the per-scale
descriptors, instead of sampling a single pre-fused (upsampled+concatenated)
map.

Motivation
----------
`fused_global` upsamples branches 1-3 to branch-0 resolution and concatenates
before sampling. That bilinear upsampling blurs the coarser branches. Sampling
each branch at its own resolution gives the graph a genuine coarse-to-fine
descriptor per landmark: coarse branches supply context, the fine branch supplies
local edge detail without upsampling blur.

Note on the precision floor: the FINEST branch is still input/4 spatial
resolution, so this enriches the per-landmark *descriptor* but does not by itself
raise the spatial resolution ceiling below input/4. It is expected to improve
delta quality / reduce gross errors more than it improves sub-cell precision.
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


@register_model("fused_global_ms")
class HRNetGNN_FusedGlobal_MS(nn.Module):
    """fused_global with per-scale (multi-resolution) point sampling.

    Args:
        num_landmarks: Number of landmarks (required).
        hrnet_backbone: Backbone name (kept for signature parity; hrnet_w18).
        feat_dim: Unused (kept for config/signature parity).
        gnn_hidden: GCN hidden dim and node feature dim.
        num_layers: Number of GCNConv layers per refinement iteration.
        num_iters: Number of coordinate refinement iterations.
        scale_indices: Which HRNet branches to sample from (default all 4).
    """

    def __init__(
        self,
        num_landmarks: int,
        hrnet_backbone: str = "hrnet_w18",
        feat_dim: int = 64,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        num_iters: int = 3,
        scale_indices: tuple = (0, 1, 2, 3),
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters
        self.scale_indices = list(scale_indices)

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True,
        )

        # Total channels sampled = sum of channels across the selected branches.
        # For global context we also mean-pool the SAME concatenated per-scale
        # descriptor space, so global_proj takes the same total_sampled dim.
        self.sampled_channels = sum(
            self.backbone.feature_info[i]["num_chs"] for i in self.scale_indices
        )

        self.coord_embed = nn.Linear(2, COORD_EMBED_DIM)
        self.global_proj = nn.Linear(self.sampled_channels, GLOBAL_EMBED_DIM)
        self.landmark_embed = nn.Embedding(num_landmarks, LANDMARK_EMBED_DIM)

        total_in = (
            self.sampled_channels
            + COORD_EMBED_DIM
            + GLOBAL_EMBED_DIM
            + LANDMARK_EMBED_DIM
        )
        self.node_feat_proj = nn.Linear(total_in, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def sample_features(self, feat_map, coords):
        """Bilinear point-sample one branch at the given coords."""
        grid = coords * 2.0 - 1.0
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C_i)

    def sample_multiscale(self, feat_maps, coords):
        """Sample every selected branch at its native res and concat channels."""
        per_scale = [
            self.sample_features(feat_maps[i], coords) for i in self.scale_indices
        ]
        return torch.cat(per_scale, dim=-1)  # (B, N, sum C_i)

    def _global_context(self, feat_maps):
        """Global descriptor in the same per-scale-concat channel space.

        Mean-pools each selected branch spatially and concatenates, matching
        the channel layout of the sampled per-landmark descriptor.
        """
        pooled = [feat_maps[i].mean(dim=[2, 3]) for i in self.scale_indices]
        return torch.cat(pooled, dim=-1)  # (B, sum C_i)

    def forward(self, x, initial_coords, edge_index):
        feat_maps = self.backbone(x)
        coords = initial_coords.clone()
        B = x.shape[0]
        N = self.num_landmarks

        global_feat = self._global_context(feat_maps)  # (B, sum C_i)
        global_emb = F.relu(self.global_proj(global_feat))  # (B, GLOBAL_EMBED_DIM)
        global_emb = global_emb.unsqueeze(1).expand(-1, N, -1)

        lm_ids = torch.arange(N, device=x.device)
        lm_emb = self.landmark_embed(lm_ids).unsqueeze(0).expand(B, -1, -1)

        for _ in range(self.num_iters):
            img_feats = self.sample_multiscale(feat_maps, coords)  # (B, N, sum C_i)
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
