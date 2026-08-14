"""GraphCondHeatmap — Graph-conditioned heatmap landmark detection.

Single-pass architecture: HRNet backbone → fused multi-scale features →
cross-attention (landmark tokens as queries) → GCN reasoning → dot-product
heatmap generation → soft-argmax coordinate extraction.
"""

import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import timm

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError(
        "torch_geometric is required for GraphCondHeatmap. "
        "Install it with: pip install torch-geometric"
    )

from .registry import register_model
from .hrnet_heatmap import soft_argmax


def make_2d_sinusoidal_pe(H: int, W: int, dim: int, device: torch.device = None) -> Tensor:
    """Generate 2D sinusoidal positional encoding.

    Uses half the dimensions for x-axis frequencies and half for y-axis
    frequencies, with standard sin/cos interleaving.

    Args:
        H: Height of the spatial grid.
        W: Width of the spatial grid.
        dim: Total embedding dimension (must be divisible by 4).
        device: Target device for the output tensor.

    Returns:
        Positional encoding tensor of shape (dim, H, W).
    """
    assert dim % 4 == 0, f"PE dim must be divisible by 4, got {dim}"
    half_dim = dim // 2
    quarter_dim = dim // 4

    # Frequency bands
    omega = torch.arange(quarter_dim, dtype=torch.float32, device=device)
    omega = 1.0 / (10000.0 ** (omega / quarter_dim))

    # Spatial grids
    y_pos = torch.arange(H, dtype=torch.float32, device=device)
    x_pos = torch.arange(W, dtype=torch.float32, device=device)

    # Outer products: (quarter_dim, H) and (quarter_dim, W)
    y_enc = torch.outer(omega, y_pos)  # (quarter_dim, H)
    x_enc = torch.outer(omega, x_pos)  # (quarter_dim, W)

    # Expand to (quarter_dim, H, W) via broadcasting
    # y_enc: (quarter_dim, H, 1) → (quarter_dim, H, W)
    # x_enc: (quarter_dim, 1, W) → (quarter_dim, H, W)
    y_enc = y_enc.unsqueeze(2).expand(-1, -1, W)
    x_enc = x_enc.unsqueeze(1).expand(-1, H, -1)

    # Interleave sin/cos: (half_dim, H, W) for each axis
    pe = torch.cat([
        y_enc.sin(),
        y_enc.cos(),
        x_enc.sin(),
        x_enc.cos(),
    ], dim=0)  # (dim, H, W)

    return pe


@register_model("graph_cond_heatmap")
class GraphCondHeatmap(nn.Module):
    """Graph-conditioned heatmap landmark detection model.

    Single-pass architecture combining cross-attention and GCN reasoning
    to produce per-landmark heatmaps via dot-product similarity.

    Args:
        num_landmarks: Number of landmarks (N).
        gnn_hidden: Token dimension and GCN hidden size.
        num_layers: Number of GCN layers.
        num_heads: Number of cross-attention heads.
        heatmap_size: Output heatmap spatial resolution.
    """

    def __init__(
        self,
        num_landmarks: int,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        heatmap_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.token_dim = gnn_hidden
        self.heatmap_size = heatmap_size

        if gnn_hidden % num_heads != 0:
            raise ValueError(
                f"gnn_hidden ({gnn_hidden}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        # --- Backbone ---
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

        # --- Feature Projection ---
        self.feat_proj = nn.Conv2d(self.fused_channels, gnn_hidden, kernel_size=1)

        # --- Landmark Identity Embeddings ---
        self.landmark_embed = nn.Embedding(num_landmarks, gnn_hidden)

        # --- Cross-Attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=gnn_hidden,
            num_heads=num_heads,
            batch_first=True,
        )

        # --- GCN Layers ---
        self.gcn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )

    def get_fused_map(self, x: Tensor) -> Tensor:
        """Extract and fuse multi-scale HRNet features."""
        feat_maps = self.backbone(x)
        H0, W0 = feat_maps[0].shape[2], feat_maps[0].shape[3]
        return torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H0, W0), mode="bilinear", align_corners=False),
        ], dim=1)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple:
        """Single-pass forward: cross-attention → GCN → dot-product heatmaps.

        Args:
            x: (B, 3, H_in, W_in) input images.
            edge_index: (2, E) graph connectivity for GCN.

        Returns:
            heatmaps: (B, N, H_out, W_out) per-landmark heatmaps.
            coords: (B, N, 2) coordinates in [0, 1].
        """
        B = x.shape[0]
        N = self.num_landmarks

        # 1. Backbone + fusion
        fused_map = self.get_fused_map(x)  # (B, fused_channels, H, W)

        # 2. Project to token_dim
        spatial_feats = self.feat_proj(fused_map)  # (B, token_dim, H, W)
        _, _, H, W = spatial_feats.shape

        # 3. Add 2D sinusoidal positional encoding
        pe = make_2d_sinusoidal_pe(H, W, self.token_dim, device=x.device)
        spatial_feats = spatial_feats + pe.unsqueeze(0)  # broadcast over batch

        # 4. Flatten for attention K/V: (B, H*W, token_dim)
        kv = spatial_feats.flatten(2).permute(0, 2, 1)  # (B, HW, token_dim)

        # 5. Landmark queries: (B, N, token_dim)
        lm_ids = torch.arange(N, device=x.device)
        queries = self.landmark_embed(lm_ids).unsqueeze(0).expand(B, -1, -1)

        # 6. Cross-attention
        attended, _ = self.cross_attn(queries, kv, kv)  # (B, N, token_dim)

        # 7. GCN reasoning
        batch_edge_index = torch.cat(
            [edge_index + b * N for b in range(B)], dim=1
        )
        h = attended.reshape(B * N, self.token_dim)
        for layer in self.gcn_layers:
            h = F.relu(layer(h, batch_edge_index))
        refined_tokens = h.view(B, N, self.token_dim)

        # 8. Dot-product heatmaps
        heatmaps = torch.einsum('bnd,bdhw->bnhw', refined_tokens, spatial_feats)

        # 9. Optionally interpolate to heatmap_size
        if H != self.heatmap_size or W != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        # 10. Soft-argmax → coordinates in [0, 1]
        coords = soft_argmax(heatmaps)

        return heatmaps, coords
