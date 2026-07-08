"""
HRNetGNN_Coord — single-scale GCN with learned coordinate embedding.

Identical to the frozen HRNetGNN except each GCN node receives a learned
embedding of its current (x, y) coordinate concatenated to the sampled image
features before the node_feat_proj linear layer.

Why a learned embedding rather than raw (x, y):
  Raw coordinates are in [0, 1] while backbone features span a wider range.
  Concatenating them directly causes the coordinate signal to be swamped by
  the feature dimensions. A small linear projection with ReLU (coord_embed)
  maps (x, y) → 16-dim at a scale the projection layer can use effectively.
  The network also learns to encode positional information in a task-relevant
  way rather than relying on the raw spatial values.

Why coordinate embedding helps:
  Standard GCNConv must infer a node's position from image features alone.
  This is ambiguous when features are similar across nearby cells (e.g., the
  coarse 16×16 map). Explicit positional input enables the GCN to learn
  "I am at (0.3, 0.4) with these features → move 3px left" rather than
  reconstructing position from context. Particularly useful for:
    - Occluded landmarks with weak/ambiguous image features
    - Dense groups (eyes, mouth inner contour) where many points have similar
      local features but different absolute positions
    - Correcting mean-shape initialisation bias under extreme pose

Architecture change vs HRNetGNN:
  coord_embed:        nn.Linear(2, 16) + ReLU  (new)
  node_feat_proj:     backbone_channels + 16 → gnn_hidden  (was backbone_channels)
  Everything else is identical.

Constructor signature matches HRNetGNN for drop-in compatibility.
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

COORD_EMBED_DIM = 16   # dimensionality of the coordinate embedding


class HRNetGNN_Coord(nn.Module):
    def __init__(
        self,
        hrnet_backbone="hrnet_w18",  # kept for API compat
        feat_dim=64,                  # kept for API compat, unused
        gnn_hidden=128,
        num_layers=2,
        num_landmarks=9,
        num_iters=3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters
        self.backbone_out_idx = -1

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True,
        )

        backbone_channels = self.backbone.feature_info[self.backbone_out_idx]["num_chs"]

        # Learned coordinate embedding: (x, y) in [0,1] → COORD_EMBED_DIM
        # Maps raw spatial coordinates to a scale commensurate with image features.
        self.coord_embed = nn.Linear(2, COORD_EMBED_DIM)

        # Projection: image features + coordinate embedding → gnn_hidden
        self.node_feat_proj = nn.Linear(backbone_channels + COORD_EMBED_DIM, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def sample_features(self, feat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample feat_map at landmark positions.

        Args:
            feat_map: (B, C, H, W)
            coords:   (B, N, 2) in [0, 1]

        Returns:
            (B, N, C)
        """
        grid = coords * 2.0 - 1.0       # [0,1] → [-1,1] for grid_sample
        grid = grid.unsqueeze(2)         # (B, N, 1, 2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

    def forward(
        self,
        x: torch.Tensor,
        initial_coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, 3, H, W)
            initial_coords: (B, N, 2) in [0, 1]
            edge_index:     (2, E) graph connectivity

        Returns:
            (B, N, 2) refined coordinates in [0, 1]
        """
        coords = initial_coords.clone()
        feat_maps = self.backbone(x)
        feat_map = feat_maps[self.backbone_out_idx]

        B = x.shape[0]
        N = coords.shape[1]

        for _ in range(self.num_iters):
            # Sample image features at current coordinate estimates
            img_feats = self.sample_features(feat_map, coords)    # (B, N, C)

            # Embed current coordinates into a learned representation
            coord_emb = F.relu(self.coord_embed(coords))          # (B, N, 16)

            # Concatenate image features + coordinate embedding
            node_feats = torch.cat([img_feats, coord_emb], dim=-1) # (B, N, C+16)
            node_feats = self.node_feat_proj(node_feats)            # (B, N, gnn_hidden)
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
