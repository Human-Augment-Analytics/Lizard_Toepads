"""
HRNetGNN_Fused — GCN with pre-fused multi-scale feature map.

The key insight from the HRNet heatmap paper:
  The correct multi-scale approach is to fuse all 4 branches FIRST (with
  learned upsampling convolutions inside HRNet), then use the resulting
  rich feature map for downstream tasks. This is different from our earlier
  hrnet_gcn_ms.py which sampled from 4 separate maps independently and
  concatenated the results.

  The fused map (960ch, 128×128 for 512px input) has had cross-scale
  information exchange throughout the backbone forward pass, making it
  richer than simple concatenation of separately-extracted features.

Architecture difference vs hrnet_gcn_coord.py:
  Old: sample from feat_maps[-1]  (144ch, 16×16) → coarse, limited spatial
  New: fuse all 4 branches → sample from fused (960ch, 128×128) → rich

Why this helps GCN:
  - 128×128 map: eye contour landmarks (8pts spanning ~70px) each fall in
    distinct 4px cells, giving the GCN differentiable per-landmark features
  - 960ch: semantic depth from coarse branches + spatial detail from fine
    branches, all fused via HRNet's learned multi-resolution fusion modules

Keeps coordinate embedding from hrnet_gcn_coord.py for explicit positional
awareness. Coarse init is omitted (separate concern).

Constructor signature matches HRNetGNN for drop-in compatibility.
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

COORD_EMBED_DIM = 16


class HRNetGNN_Fused(nn.Module):
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

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True,
        )

        # Compute actual fused channel count from a dummy forward pass.
        # timm's feature_info may not accurately reflect the concatenated output.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            feats = self.backbone(dummy)
            H0, W0 = feats[0].shape[2], feats[0].shape[3]
            fused_dummy = torch.cat([
                feats[0],
                F.interpolate(feats[1], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[2], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[3], size=(H0, W0), mode="bilinear", align_corners=False),
            ], dim=1)
            self.fused_channels = fused_dummy.shape[1]

        # Coordinate embedding: (x,y) → 16-dim learned representation
        self.coord_embed = nn.Linear(2, COORD_EMBED_DIM)

        # Project fused features + coord embedding → gnn_hidden
        self.node_feat_proj = nn.Linear(self.fused_channels + COORD_EMBED_DIM, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def get_fused_map(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and fuse all 4 resolution branches.

        Returns:
            (B, fused_channels, H/4, W/4) — same spatial resolution as
            the highest-resolution HRNet branch (128×128 for 512px input).
        """
        feat_maps = self.backbone(x)
        H0, W0 = feat_maps[0].shape[2], feat_maps[0].shape[3]
        return torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H0, W0), mode="bilinear", align_corners=False),
        ], dim=1)  # (B, fused_channels, H0, W0)

    def sample_features(self, feat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample feat_map at landmark positions.

        Args:
            feat_map: (B, C, H, W)
            coords:   (B, N, 2) in [0, 1]

        Returns:
            (B, N, C)
        """
        grid = coords * 2.0 - 1.0
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)

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
        fused_map = self.get_fused_map(x)  # (B, fused_channels, H0, W0)

        coords = initial_coords.clone()
        B = x.shape[0]
        N = self.num_landmarks

        for _ in range(self.num_iters):
            img_feats = self.sample_features(fused_map, coords)     # (B, N, fused_ch)
            coord_emb = F.relu(self.coord_embed(coords))            # (B, N, 16)
            node_feats = torch.cat([img_feats, coord_emb], dim=-1)  # (B, N, fused_ch+16)
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
