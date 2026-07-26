"""HRNetGNN_Coord — single-scale GCN with learned coordinate embedding
and optional coarse initializer.
"""

import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

from .registry import register_model

COORD_EMBED_DIM = 16


@register_model("coord")
class HRNetGNN_Coord(nn.Module):
    def __init__(
        self,
        num_landmarks: int,
        hrnet_backbone: str = "hrnet_w18",
        feat_dim: int = 64,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        num_iters: int = 3,
        use_coarse_init: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters
        self.backbone_out_idx = -1
        self.use_coarse_init = use_coarse_init

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True,
        )

        backbone_channels = self.backbone.feature_info[self.backbone_out_idx]["num_chs"]

        if self.use_coarse_init:
            self.coarse_init_mlp = nn.Sequential(
                nn.Linear(backbone_channels, 256),
                nn.ReLU(),
                nn.Linear(256, num_landmarks * 2),
            )
            nn.init.zeros_(self.coarse_init_mlp[-1].bias)
            nn.init.normal_(self.coarse_init_mlp[-1].weight, std=0.01)

        self.coord_embed = nn.Linear(2, COORD_EMBED_DIM)
        self.node_feat_proj = nn.Linear(backbone_channels + COORD_EMBED_DIM, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def sample_features(self, feat_map, coords):
        grid = coords * 2.0 - 1.0
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(self, x, initial_coords, edge_index):
        feat_maps = self.backbone(x)
        feat_map = feat_maps[self.backbone_out_idx]

        B = x.shape[0]
        N = self.num_landmarks

        if self.use_coarse_init:
            global_feat = feat_map.mean(dim=[2, 3])
            coarse_flat = self.coarse_init_mlp(global_feat)
            coarse_coords = torch.sigmoid(coarse_flat.view(B, N, 2))
        else:
            coarse_coords = None

        coords = initial_coords.clone()

        for _ in range(self.num_iters):
            img_feats = self.sample_features(feat_map, coords)
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

        if self.use_coarse_init:
            return coords, coarse_coords
        return coords
