"""HRNetGNN_HInit — GCN with frozen HRNet heatmap as coarse initializer.

Requires a trained heatmap checkpoint for initialization.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .registry import register_model
from .hrnet_heatmap import HRNetHeatmap, hard_argmax
from .hrnet_gcn_fused import HRNetGNN_Fused


@register_model("hinit")
class HRNetGNN_HInit(nn.Module):
    """GCN with frozen heatmap model as coarse initializer.

    Args:
        num_landmarks: Number of landmarks (required).
        heatmap_checkpoint: Path to trained HRNetHeatmap checkpoint.
        gnn_hidden: GCN hidden dimension.
        num_layers: GCN conv layers per iteration.
        num_iters: GCN refinement iterations.
        heatmap_size: Heatmap spatial size (must match checkpoint).
    """

    def __init__(
        self,
        num_landmarks: int,
        heatmap_checkpoint: str = "",
        hrnet_backbone: str = "hrnet_w18",
        feat_dim: int = 64,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        num_iters: int = 4,
        heatmap_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Frozen heatmap initializer
        self.heatmap_model = HRNetHeatmap(
            num_landmarks=num_landmarks,
            pretrained=False,
            heatmap_size=heatmap_size,
        )

        if heatmap_checkpoint:
            state = torch.load(heatmap_checkpoint, map_location="cpu")
            self.heatmap_model.load_state_dict(state)

        # Freeze all heatmap parameters
        for p in self.heatmap_model.parameters():
            p.requires_grad = False
        self.heatmap_model.eval()

        # GCN refinement head
        self.gcn = HRNetGNN_Fused(
            num_landmarks=num_landmarks,
            hrnet_backbone=hrnet_backbone,
            feat_dim=feat_dim,
            gnn_hidden=gnn_hidden,
            num_layers=num_layers,
            num_iters=num_iters,
        )

        self.backbone = self.gcn.backbone

    def forward(self, x, initial_coords, edge_index):
        with torch.no_grad():
            heatmaps, _ = self.heatmap_model(x)
            coarse_coords = hard_argmax(heatmaps)

        refined_coords = self.gcn(x, coarse_coords, edge_index)
        return refined_coords

    def train(self, mode=True):
        """Keep heatmap model always in eval mode."""
        super().train(mode)
        self.heatmap_model.eval()
        return self
