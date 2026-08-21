"""STAR Head — GCN with STAR uncertainty + intermediate supervision.

Extends fused_global with:
1. Per-iteration coordinate outputs for intermediate MSE supervision
2. STAR uncertainty head on the final iteration for adaptive ambiguity reduction
3. Configurable gnn_hidden (default 256 for richer representations)

Training mode returns all intermediate coords + final sigma.
Eval mode returns only final coords + sigma (same interface as before).
"""

import torch
from torch import nn
import torch.nn.functional as F

from .registry import register_model

COORD_EMBED_DIM = 16
LANDMARK_EMBED_DIM = 32
GLOBAL_EMBED_DIM = 64


@register_model("fused_global_star")
class HRNetGNN_FusedGlobal_STAR(nn.Module):
    """GCN fused_global with STAR uncertainty and intermediate supervision.

    Forward returns (training):
        all_coords: list of (B, N, 2) per iteration (length = num_iters)
        log_sigma: (B, N, 3) Cholesky parameters from final iteration

    Forward returns (eval):
        coords: (B, N, 2) final predicted coordinates
        log_sigma: (B, N, 3) Cholesky parameters
    """

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

        import timm
        from torch_geometric.nn import GCNConv

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
        self.global_proj = nn.Linear(self.fused_channels, GLOBAL_EMBED_DIM)
        self.landmark_embed = nn.Embedding(num_landmarks, LANDMARK_EMBED_DIM)

        total_in = self.fused_channels + COORD_EMBED_DIM + GLOBAL_EMBED_DIM + LANDMARK_EMBED_DIM
        self.node_feat_proj = nn.Linear(total_in, gnn_hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

        # Coordinate prediction head
        self.delta_head = nn.Linear(gnn_hidden, 2)

        # STAR uncertainty head: predicts Cholesky parameters per landmark
        # Output: [log_L11, L21, log_L22] — 3 values per landmark
        self.sigma_head = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gnn_hidden // 2, 3),
        )
        # Initialize sigma head output to near-zero (isotropic start)
        nn.init.zeros_(self.sigma_head[-1].weight)
        nn.init.zeros_(self.sigma_head[-1].bias)

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
        """Forward pass with intermediate supervision + STAR uncertainty.

        Args:
            x: (B, 3, H, W) input images.
            initial_coords: (B, N, 2) initial landmark coordinates.
            edge_index: (2, E) graph edge index.

        Returns (training mode):
            all_coords: list of (B, N, 2) — one per iteration
            log_sigma: (B, N, 3) Cholesky covariance parameters (final iter)

        Returns (eval mode):
            coords: (B, N, 2) final coordinates only
            log_sigma: (B, N, 3) Cholesky covariance parameters
        """
        fused_map = self.get_fused_map(x)
        coords = initial_coords.clone()
        B = x.shape[0]
        N = self.num_landmarks

        global_feat = fused_map.mean(dim=[2, 3])
        global_emb = F.relu(self.global_proj(global_feat))
        global_emb = global_emb.unsqueeze(1).expand(-1, N, -1)

        lm_ids = torch.arange(N, device=x.device)
        lm_emb = self.landmark_embed(lm_ids).unsqueeze(0).expand(B, -1, -1)

        all_coords = []
        h_final = None

        for it in range(self.num_iters):
            img_feats = self.sample_features(fused_map, coords)
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
            all_coords.append(coords)

            if it == self.num_iters - 1:
                h_final = h

        # Predict per-landmark anisotropic uncertainty from final GCN features
        log_sigma = self.sigma_head(h_final).view(B, N, 3)

        if self.training:
            return all_coords, log_sigma
        else:
            return coords, log_sigma
