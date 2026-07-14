"""
HRNetGNN_FusedGlobal — GCN with fused multi-scale features + global context
                       + learned landmark identity embeddings + coordinate encoding.

Extends hrnet_gcn_fused.py with three additional per-node signals:
  1. Global context (GAP): fused feature map → global avg pool → Linear → 64-dim
     Broadcast to all nodes. Gives each node awareness of the full face layout.
  2. Landmark identity embedding: nn.Embedding(num_landmarks, 32)
     Tells each GCN node which landmark it represents, enabling landmark-specific
     refinement behaviour without extra parameters per landmark.
  3. Coordinate embedding: Linear(2, 16) — same as hrnet_gcn_fused.py

Per-node feature vector per iteration:
  [local_sample (fused_ch) | coord_emb (16) | global (64) | landmark_id (32)]
  → node_feat_proj → gnn_hidden

Usage in config: "model_variant": "fused_global"
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

COORD_EMBED_DIM    = 16
LANDMARK_EMBED_DIM = 32
GLOBAL_EMBED_DIM   = 64


class HRNetGNN_FusedGlobal(nn.Module):
    def __init__(
        self,
        hrnet_backbone="hrnet_w18",
        feat_dim=64,
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

        # Compute fused channel count from a dummy forward pass
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

        # ── Per-node feature components ───────────────────────────────────
        self.coord_embed    = nn.Linear(2, COORD_EMBED_DIM)
        self.global_proj    = nn.Linear(self.fused_channels, GLOBAL_EMBED_DIM)
        self.landmark_embed = nn.Embedding(num_landmarks, LANDMARK_EMBED_DIM)

        # Project concatenated features to gnn_hidden
        total_in = self.fused_channels + COORD_EMBED_DIM + GLOBAL_EMBED_DIM + LANDMARK_EMBED_DIM
        self.node_feat_proj = nn.Linear(total_in, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        self.delta_head = nn.Linear(gnn_hidden, 2)

    def get_fused_map(self, x: torch.Tensor) -> torch.Tensor:
        feat_maps = self.backbone(x)
        H0, W0 = feat_maps[0].shape[2], feat_maps[0].shape[3]
        return torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H0, W0), mode="bilinear", align_corners=False),
        ], dim=1)

    def sample_features(self, feat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
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
        fused_map = self.get_fused_map(x)

        coords = initial_coords.clone()
        B = x.shape[0]
        N = self.num_landmarks

        # Global context — computed once, reused every iteration
        global_feat = fused_map.mean(dim=[2, 3])                        # (B, fused_ch)
        global_emb  = F.relu(self.global_proj(global_feat))             # (B, 64)
        global_emb  = global_emb.unsqueeze(1).expand(-1, N, -1)         # (B, N, 64)

        # Landmark identity — fixed per sample, same across batch
        lm_ids = torch.arange(N, device=x.device)
        lm_emb = self.landmark_embed(lm_ids).unsqueeze(0).expand(B, -1, -1)  # (B, N, 32)

        for _ in range(self.num_iters):
            img_feats = self.sample_features(fused_map, coords)         # (B, N, fused_ch)
            coord_emb = F.relu(self.coord_embed(coords))                # (B, N, 16)

            node_feats = torch.cat([img_feats, coord_emb, global_emb, lm_emb], dim=-1)
            node_feats = F.relu(self.node_feat_proj(node_feats))        # (B, N, gnn_hidden)

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
