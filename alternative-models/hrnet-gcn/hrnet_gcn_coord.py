"""
HRNetGNN_Coord — single-scale GCN with learned coordinate embedding
                 and optional coarse initializer.

Two enhancements over the frozen HRNetGNN:

1. Coordinate embedding (always active):
   Each GCN node receives a learned 16-dim embedding of its current (x, y)
   coordinate concatenated to the sampled image features. This gives the GCN
   explicit positional awareness, reducing ambiguity for dense landmark groups
   and occluded landmarks with weak image features.

2. Coarse initializer (optional, controlled by use_coarse_init):
   A lightweight MLP operating on globally-pooled backbone features produces
   an image-specific initial coordinate estimate (98×2) instead of the
   dataset mean shape. This directly addresses the mean-shape limitation for
   hard cases (extreme pose, unusual scale) where the mean shape may start
   30-50px from the true landmark positions.

   Architecture:
     global_pool(feat_map) → [144] → Linear(144, 256) → ReLU
                                   → Linear(256, num_landmarks * 2) → sigmoid
                                   → (B, num_landmarks, 2)

   Training:
     The coarse init output is supervised with a landmark_loss term with a
     ramping weight (0→1 over coarse_init_ramp_epochs). This prevents large
     early-epoch gradients from the random MLP destabilising backbone training.
     The coarse init loss and GCN loss both backprop into the backbone, so
     the ramp is important.

   At inference (eval mode):
     The coarse init prediction replaces the mean-shape input. No noise is
     added (noise is only for training robustness).

Constructor signature matches HRNetGNN for drop-in compatibility.
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

COORD_EMBED_DIM = 16   # dimensionality of the coordinate positional embedding


class HRNetGNN_Coord(nn.Module):
    def __init__(
        self,
        hrnet_backbone="hrnet_w18",   # kept for API compat
        feat_dim=64,                   # kept for API compat, unused
        gnn_hidden=128,
        num_layers=2,
        num_landmarks=9,
        num_iters=3,
        use_coarse_init: bool = True,
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

        # ── Coarse initializer ────────────────────────────────────────────
        # Global average pool → small MLP → (num_landmarks, 2) in [0, 1]
        if self.use_coarse_init:
            self.coarse_init_mlp = nn.Sequential(
                nn.Linear(backbone_channels, 256),
                nn.ReLU(),
                nn.Linear(256, num_landmarks * 2),
            )
            # Initialise final layer to predict near (0.5, 0.5) so early
            # predictions are centred rather than random
            nn.init.zeros_(self.coarse_init_mlp[-1].bias)
            nn.init.normal_(self.coarse_init_mlp[-1].weight, std=0.01)

        # ── Coordinate embedding ──────────────────────────────────────────
        self.coord_embed = nn.Linear(2, COORD_EMBED_DIM)

        # ── GCN head ──────────────────────────────────────────────────────
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
        grid = coords * 2.0 - 1.0
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        initial_coords: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        """
        Args:
            x:              (B, 3, H, W)
            initial_coords: (B, N, 2) in [0, 1] — used as fallback when
                            use_coarse_init=False, or as the noise-augmented
                            mean shape during training (ignored at eval if
                            use_coarse_init=True).
            edge_index:     (2, E) graph connectivity

        Returns:
            If use_coarse_init=True:
                (gcn_coords, coarse_coords)
                  gcn_coords:    (B, N, 2) final refined coordinates
                  coarse_coords: (B, N, 2) coarse initializer output (for loss)
            If use_coarse_init=False:
                gcn_coords only: (B, N, 2)
        """
        feat_maps = self.backbone(x)
        feat_map = feat_maps[self.backbone_out_idx]   # (B, C, H, W)

        B = x.shape[0]
        N = self.num_landmarks

        # ── Coarse initializer ────────────────────────────────────────────
        if self.use_coarse_init:
            global_feat = feat_map.mean(dim=[2, 3])               # (B, C)
            coarse_flat = self.coarse_init_mlp(global_feat)        # (B, N*2)
            coarse_coords = torch.sigmoid(coarse_flat.view(B, N, 2))  # (B, N, 2)
        else:
            coarse_coords = None

        # GCN always starts from initial_coords (mean shape at train, caller-provided at eval).
        # The coarse_coords output is used as an auxiliary loss target during training
        # and can be passed as initial_coords by the eval caller once trained.
        coords = initial_coords.clone()
        else:
            coords = initial_coords.clone()
            coarse_coords = None

        # ── GCN refinement ────────────────────────────────────────────────
        for _ in range(self.num_iters):
            img_feats = self.sample_features(feat_map, coords)      # (B, N, C)
            coord_emb = F.relu(self.coord_embed(coords))            # (B, N, 16)
            node_feats = torch.cat([img_feats, coord_emb], dim=-1)  # (B, N, C+16)
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

        if self.use_coarse_init:
            return coords, coarse_coords
        return coords
