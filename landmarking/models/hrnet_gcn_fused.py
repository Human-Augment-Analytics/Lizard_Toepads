"""HRNetGNN_Fused — GCN with pre-fused multi-scale feature map.

Fuses all 4 HRNet branches before sampling, providing rich multi-scale
features at high spatial resolution.
"""

import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

from .registry import register_model

COORD_EMBED_DIM = 16


@register_model("fused")
class HRNetGNN_Fused(nn.Module):
    def __init__(
        self,
        num_landmarks: int,
        hrnet_backbone: str = "hrnet_w18",
        feat_dim: int = 64,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        num_iters: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
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
            self.fused_channels = fused_dummy.shape[1]

        self.coord_embed = nn.Linear(2, COORD_EMBED_DIM)
        self.node_feat_proj = nn.Linear(self.fused_channels + COORD_EMBED_DIM, gnn_hidden)

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

    def sample_features(self, feat_map, coords):
        grid = coords * 2.0 - 1.0
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(self, x, initial_coords, edge_index):
        fused_map = self.get_fused_map(x)
        coords = initial_coords.clone()
        B = x.shape[0]
        N = self.num_landmarks

        for _ in range(self.num_iters):
            img_feats = self.sample_features(fused_map, coords)
            coord_emb = F.relu(self.coord_embed(coords))
            node_feats = torch.cat([img_feats, coord_emb], dim=-1)
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

        return coords
