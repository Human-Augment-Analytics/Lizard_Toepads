"""
HRNetGNN_HInit — GCN with frozen HRNet heatmap as coarse initializer.

Architecture:
  1. Frozen HRNetHeatmap produces per-landmark heatmaps from the input image.
  2. hard_argmax extracts image-specific (x, y) initial coordinates.
  3. HRNetGNN_Fused refines those coordinates using the fused multi-scale
     feature map and GCN message passing.

Why this works:
  - The heatmap model provides an image-specific initial guess that already
    handles pose variation, scale, and occlusion through its global spatial
    search.
  - The GCN refines from that guess, adding structural consistency and
    landmark relationship awareness that the heatmap model lacks.
  - Training: the heatmap model is FROZEN so only the GCN parameters are
    updated. The heatmap predictions are used as detached initial coordinates.
  - Inference: identical to training — no angle or external prior needed.

Compared to mean-shape initialization:
  - Mean shape: same upright frontal prior for all images regardless of pose
  - HInit: image-specific prior that adapts to the actual face orientation

This directly addresses the GCN's pose subset weakness without requiring
rotation augmentation coordination.

Usage in config: "model_variant": "hinit"
Requires: "heatmap_checkpoint" path in config pointing to a trained
          hrnet_heatmap_wflw_best.pth checkpoint.
"""
import sys
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path

# Import the fused GCN base
sys.path.insert(0, str(Path(__file__).parent))
from hrnet_gcn_fused import HRNetGNN_Fused

# Import heatmap model — resolve path relative to this file
_HRNET_DIR = Path(__file__).parent.parent / "hrnet"
sys.path.insert(0, str(_HRNET_DIR))
from hrnet_heatmap import HRNetHeatmap, hard_argmax


class HRNetGNN_HInit(nn.Module):
    """GCN with frozen heatmap model as coarse initializer.

    Args:
        heatmap_checkpoint: Path to trained HRNetHeatmap checkpoint (.pth).
        hrnet_backbone:     Kept for API compat (ignored).
        feat_dim:           Kept for API compat (ignored).
        gnn_hidden:         GCN hidden dimension.
        num_layers:         GCN conv layers per iteration.
        num_landmarks:      Number of landmarks (98 for WFLW).
        num_iters:          GCN refinement iterations.
        heatmap_size:       Heatmap spatial size (must match checkpoint, 64).
    """

    def __init__(
        self,
        heatmap_checkpoint: str,
        hrnet_backbone="hrnet_w18",
        feat_dim=64,
        gnn_hidden=128,
        num_layers=2,
        num_landmarks=98,
        num_iters=4,
        heatmap_size: int = 64,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks

        # ── Frozen heatmap initializer ────────────────────────────────────
        self.heatmap_model = HRNetHeatmap(
            num_landmarks=num_landmarks,
            pretrained=False,  # weights loaded from checkpoint
            heatmap_size=heatmap_size,
        )
        state = torch.load(heatmap_checkpoint, map_location="cpu")
        self.heatmap_model.load_state_dict(state)
        # Freeze all heatmap parameters — only GCN trains
        for p in self.heatmap_model.parameters():
            p.requires_grad = False
        self.heatmap_model.eval()

        # ── GCN refinement head ───────────────────────────────────────────
        self.gcn = HRNetGNN_Fused(
            hrnet_backbone=hrnet_backbone,
            feat_dim=feat_dim,
            gnn_hidden=gnn_hidden,
            num_layers=num_layers,
            num_landmarks=num_landmarks,
            num_iters=num_iters,
        )

        # Expose the GCN's backbone as a top-level attribute so that
        # train_wflw.py can access model.backbone for differential LR setup.
        self.backbone = self.gcn.backbone

    def forward(
        self,
        x: torch.Tensor,
        initial_coords: torch.Tensor,  # kept for API compat — ignored
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, 3, H, W) input images
            initial_coords: ignored — heatmap provides image-specific init
            edge_index:     (2, E) GCN graph connectivity

        Returns:
            (B, N, 2) refined coordinates in [0, 1]
        """
        # Coarse init from frozen heatmap model
        with torch.no_grad():
            heatmaps, _ = self.heatmap_model(x)
            coarse_coords = hard_argmax(heatmaps)  # (B, N, 2) in [0, 1]

        # GCN refinement from heatmap predictions
        refined_coords = self.gcn(x, coarse_coords, edge_index)
        return refined_coords

    def train(self, mode: bool = True):
        """Keep heatmap model always in eval mode regardless of training mode."""
        super().train(mode)
        self.heatmap_model.eval()
        return self
