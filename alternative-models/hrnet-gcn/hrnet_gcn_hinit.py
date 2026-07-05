"""
HRNetGNNWithInit — HRNet-GCN with image-conditioned initial prediction.

This model replaces the external fixed mean-shape initialization used by
HRNetGNN with an image-conditioned MLP head that predicts initial landmark
coordinates directly from the backbone features.

Architecture:
    backbone(x) → feat_map [B, C, H, W]
         │
    AdaptiveAvgPool2d(1) → Flatten → Linear(C, 256) → ReLU
         │
    Linear(256, N*2) → Sigmoid → reshape [B, N, 2]  ← initial_coords
         │
    GCN refinement (num_iters iterations):
      sample_features(feat_map, coords) → node_feats
      node_feat_proj → ReLU → GCNConv layers → delta_head
      coords = coords + delta
         │
    (initial_coords, final_coords)

The forward signature is (x, edge_index) — no external initial_coords argument.
Returns a tuple (initial_coords, final_coords) to support dual-loss training.

This file is standalone and does NOT import from hrnet_gcn.py.
"""
import torch
import torch.nn.functional as F
import timm
from torch import nn
from torch_geometric.nn import GCNConv


class HRNetGNNWithInit(nn.Module):
    """HRNet-GCN with image-conditioned landmark initialization.

    Args:
        hrnet_backbone: timm model name for the HRNet backbone.
        feat_dim:       Unused (kept for API compatibility with HRNetGNN).
        gnn_hidden:     Hidden dimension for GCN layers and node feature projection.
        num_layers:     Number of GCNConv layers per refinement iteration.
        num_landmarks:  Number of output landmarks.
        num_iters:      Number of cascade refinement iterations.
    """

    def __init__(
        self,
        hrnet_backbone: str = "hrnet_w18",
        feat_dim: int = 64,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        num_landmarks: int = 9,
        num_iters: int = 3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters

        # Backbone
        self.backbone = timm.create_model(
            hrnet_backbone,
            pretrained=True,
            features_only=True,
        )
        self.backbone_out_idx = -1
        backbone_channels = self.backbone.feature_info[self.backbone_out_idx]["num_chs"]

        # Init_Head: global avg pool → MLP → (B, N, 2) in [0, 1]
        self.init_pool = nn.AdaptiveAvgPool2d(1)
        self.init_fc1 = nn.Linear(backbone_channels, 256)
        self.init_fc2 = nn.Linear(256, num_landmarks * 2)

        # GCN refinement (independent copy — does not import from hrnet_gcn.py)
        self.node_feat_proj = nn.Linear(backbone_channels, gnn_hidden)
        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )
        self.delta_head = nn.Linear(gnn_hidden, 2)

    def _init_head(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Predict initial landmark coordinates from global average-pooled features.

        Args:
            feat_map: [B, C, H, W] backbone feature map.

        Returns:
            initial_coords: [B, N, 2] tensor with values in [0, 1].
        """
        B = feat_map.shape[0]
        pooled = self.init_pool(feat_map)     # [B, C, 1, 1]
        pooled = pooled.flatten(1)            # [B, C]
        h = F.relu(self.init_fc1(pooled))     # [B, 256]
        out = torch.sigmoid(self.init_fc2(h)) # [B, N*2]
        return out.view(B, self.num_landmarks, 2)

    def sample_features(
        self, feat_map: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Bilinearly sample backbone features at landmark coordinate positions.

        Args:
            feat_map: [B, C, H, W] feature map.
            coords:   [B, N, 2] normalized coordinates in [0, 1].

        Returns:
            sampled: [B, N, C] per-landmark feature vectors.
        """
        grid = coords.clone()
        grid = (grid * 2) - 1          # rescale [0,1] → [-1,1] for grid_sample
        grid = grid.unsqueeze(2)       # [B, N, 1, 2]
        sampled = F.grid_sample(feat_map, grid, align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple:
        """Forward pass.

        Args:
            x:          [B, 3, H, W] input image batch.
            edge_index: [2, E] graph edge indices (PyTorch Geometric format).

        Returns:
            Tuple of:
                initial_coords: [B, N, 2] — sigmoid-bounded initial prediction
                final_coords:   [B, N, 2] — after GCN cascade refinement
        """
        feat_maps = self.backbone(x)
        feat_map = feat_maps[self.backbone_out_idx]  # [B, C, H, W]

        # Image-conditioned initial prediction
        initial_coords = self._init_head(feat_map)   # [B, N, 2]

        # GCN cascade refinement
        coords = initial_coords.clone()
        B, N = coords.shape[:2]

        # Build batched edge index for the full batch
        batch_edge_index = torch.cat(
            [edge_index + b * N for b in range(B)], dim=1
        )

        for _ in range(self.num_iters):
            node_feats = self.sample_features(feat_map, coords)  # [B, N, C]
            node_feats = F.relu(self.node_feat_proj(node_feats))  # [B, N, gnn_hidden]
            h = node_feats.view(B * N, -1)                        # [B*N, gnn_hidden]

            for layer in self.gnn_layers:
                h = F.relu(layer(h, batch_edge_index))

            delta = self.delta_head(h).view(B, N, 2)
            coords = coords + delta

        return initial_coords, coords
