"""
HRNetGNN_MS — multi-scale feature variant of HRNetGNN.

Differences from the frozen hrnet_gcn.py:
  - Samples from ALL HRNet feature map scales (indices 0–3) simultaneously
    rather than only the last (lowest-resolution) map.
  - Concatenates the per-scale sampled features per landmark, then projects
    to gnn_hidden. This gives the GCN both fine spatial detail (index 0,
    128×128) and high semantic context (index 3, 16×16) at each node.
  - backbone_out_idx parameter is removed; all scales are always used.
  - The projection input dimension is sum of all scale channel counts.
  - Constructor signature is otherwise identical to HRNetGNN so it can be
    used as a drop-in replacement in train_wflw.py.

Why this matters:
  The original model uses only the 16×16 feature map (32px per cell on a
  512px input). Tightly clustered landmarks like the 8-point eye contour
  (spanning ~70px) fall within 2–3 cells and receive nearly identical
  features, making it hard for the GCN to differentiate them. Using the
  128×128 map gives 4px per cell, so adjacent eye landmarks sample from
  distinct cells.

Merging back to Lizard:
  The multi-scale projection changes the node_feat_proj input dimension,
  so a checkpoint from HRNetGNN cannot be loaded directly into
  HRNetGNN_MS. Train from scratch (with pretrained HRNet backbone) or
  fine-tune by freezing the backbone and re-initialising node_feat_proj.
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv


class HRNetGNN_MS(nn.Module):
    """HRNet-GCN with multi-scale feature sampling.

    Args:
        hrnet_backbone: Ignored (kept for API compatibility). Always uses hrnet_w18.
        feat_dim:       Unused legacy parameter (kept for API compatibility).
        gnn_hidden:     Hidden dimension for GCN layers and output projection.
        num_layers:     Number of GCNConv layers per refinement iteration.
        num_landmarks:  Number of landmarks to predict.
        num_iters:      Number of coordinate refinement iterations.
        scale_indices:  Which HRNet feature map indices to sample from.
                        Default (0, 1, 2, 3) uses all four scales.
                        Use (0, 3) for a lighter two-scale version.
    """

    def __init__(
        self,
        hrnet_backbone="hrnet_w18",   # kept for API compat, ignored
        feat_dim=64,                   # kept for API compat, unused
        gnn_hidden=128,
        num_layers=2,
        num_landmarks=9,
        num_iters=3,
        scale_indices=(0, 1, 2, 3),
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

        # Total input channels = sum of channels across all sampled scales
        total_in_channels = sum(
            self.backbone.feature_info[i]["num_chs"]
            for i in self.scale_indices
        )

        # Project concatenated multi-scale features to gnn_hidden
        self.node_feat_proj = nn.Linear(total_in_channels, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(p=0.1)

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def sample_features(self, feat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample feat_map at landmark positions.

        Args:
            feat_map: (B, C, H, W)
            coords:   (B, N, 2) in [0, 1]

        Returns:
            (B, N, C) sampled features
        """
        # grid_sample expects grid in [-1, 1], shape (B, N, 1, 2)
        grid = coords * 2.0 - 1.0          # [0,1] → [-1,1]
        grid = grid.unsqueeze(2)            # (B, N, 1, 2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)   # (B, N, C)

    def forward(
        self,
        x: torch.Tensor,
        initial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        return_all_iters: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, 3, H, W) input images
            initial_coords: (B, N, 2) initial landmark positions in [0, 1]
            edge_index:     (2, E) graph connectivity (single-graph, batched inside)
            return_all_iters: if True, returns list of (B, N, 2) coords per iteration
                              instead of just the final coords. Used for intermediate
                              supervision during training.

        Returns:
            If return_all_iters=False: (B, N, 2) final refined landmark positions
            If return_all_iters=True:  list of (B, N, 2), one per iteration
        """
        coords = initial_coords.clone()
        feat_maps = self.backbone(x)  # list of feature maps at each scale

        B = x.shape[0]
        N = coords.shape[1]
        all_coords = []

        for _ in range(self.num_iters):
            # Sample from each selected scale and concatenate along channel dim
            scale_feats = [
                self.sample_features(feat_maps[i], coords)
                for i in self.scale_indices
            ]
            node_feats = torch.cat(scale_feats, dim=-1)   # (B, N, sum_C)

            node_feats = self.node_feat_proj(node_feats)   # (B, N, gnn_hidden)
            node_feats = F.relu(node_feats)

            # Flatten for PyG batch processing
            node_feats_flat = node_feats.view(B * N, -1)

            # Build batched edge index
            batch_edge_index = torch.cat(
                [edge_index + b * N for b in range(B)], dim=1
            )

            h = node_feats_flat
            for layer in self.gnn_layers:
                h = layer(h, batch_edge_index)
                h = F.relu(h)
                h = self.dropout(h)

            delta = self.delta_head(h).view(B, N, 2)
            coords = torch.clamp(coords + delta, 0.0, 1.0)

            if return_all_iters:
                all_coords.append(coords)

        if return_all_iters:
            return all_coords
        return coords
