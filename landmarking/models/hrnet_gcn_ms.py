"""HRNetGNN_MS — multi-scale feature variant of HRNetGNN.

Samples from ALL HRNet feature map scales simultaneously, concatenates
per-scale features, and projects to GCN hidden dimension.
"""

import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

from .registry import register_model


@register_model("multiscale")
class HRNetGNN_MS(nn.Module):
    """HRNet-GCN with multi-scale feature sampling.

    Args:
        num_landmarks: Number of landmarks to predict (required).
        gnn_hidden: Hidden dimension for GCN layers.
        num_layers: Number of GCNConv layers per refinement iteration.
        num_iters: Number of coordinate refinement iterations.
        scale_indices: Which HRNet feature map indices to sample from.
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

        total_in_channels = sum(
            self.backbone.feature_info[i]["num_chs"]
            for i in self.scale_indices
        )

        self.node_feat_proj = nn.Linear(total_in_channels, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def sample_features(self, feat_map, coords):
        grid = coords * 2.0 - 1.0
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(self, x, initial_coords, edge_index, return_all_iters=False):
        coords = initial_coords.clone()
        feat_maps = self.backbone(x)

        B = x.shape[0]
        N = coords.shape[1]
        all_coords = []

        for _ in range(self.num_iters):
            scale_feats = [
                self.sample_features(feat_maps[i], coords)
                for i in self.scale_indices
            ]
            node_feats = torch.cat(scale_feats, dim=-1)
            node_feats = self.node_feat_proj(node_feats)
            node_feats = F.relu(node_feats)

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

            if return_all_iters:
                all_coords.append(coords)

        if return_all_iters:
            return all_coords
        return coords
