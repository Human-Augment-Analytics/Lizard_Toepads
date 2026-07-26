"""Loss functions for landmark detection training.

Provides coordinate-space MSE loss and optional graph-regularized
distance loss for chain topologies.
"""

import torch
import torch.nn.functional as F


def landmark_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss between predicted and ground truth coordinates.

    Args:
        pred: (B, N, 2) predicted landmark coordinates.
        gt: (B, N, 2) ground truth landmark coordinates.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred, gt)


def dist_loss(
    pred: torch.Tensor, gt: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Graph-regularized distance loss for chain topologies.

    Penalizes differences in inter-landmark distances between predictions
    and ground truth along graph edges. This encourages the model to
    preserve relative spacing between connected landmarks.

    Args:
        pred: (B, N, 2) predicted landmark coordinates.
        gt: (B, N, 2) ground truth landmark coordinates.
        edge_index: (2, E) edge index tensor defining connectivity.

    Returns:
        Scalar distance loss.
    """
    src = edge_index[0]  # source node indices
    dst = edge_index[1]  # destination node indices

    # Compute inter-landmark distances along edges
    pred_dists = torch.norm(pred[:, src] - pred[:, dst], dim=-1)
    gt_dists = torch.norm(gt[:, src] - gt[:, dst], dim=-1)

    return F.mse_loss(pred_dists, gt_dists)


def heatmap_loss(
    pred_heatmaps: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    heatmap_size: int,
    sigma: float = 1.5,
    coord_weight: float = 100.0,
) -> torch.Tensor:
    """Combined heatmap MSE + coordinate loss for heatmap regression models.

    Generates Gaussian target heatmaps from gt_coords, then computes:
        loss = MSE(pred_heatmaps, gt_heatmaps) + coord_weight * MSE(pred_coords, gt_coords)

    Args:
        pred_heatmaps: (B, K, H, W) predicted heatmap logits.
        pred_coords: (B, K, 2) soft-argmax decoded coordinates in [0, 1].
        gt_coords: (B, K, 2) ground truth coordinates in [0, 1].
        heatmap_size: Spatial size of target heatmaps (H = W).
        sigma: Gaussian sigma for target heatmap generation.
        coord_weight: Scalar weight for the coordinate loss term.

    Returns:
        Scalar combined loss.
    """
    B, K, _ = gt_coords.shape
    device = gt_coords.device

    # Generate Gaussian target heatmaps
    px = gt_coords[:, :, 0] * (heatmap_size - 1)  # (B, K)
    py = gt_coords[:, :, 1] * (heatmap_size - 1)

    ys = torch.arange(heatmap_size, device=device, dtype=torch.float32)
    xs = torch.arange(heatmap_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    # (B, K, H, W) distances
    dx = grid_x.unsqueeze(0).unsqueeze(0) - px.unsqueeze(-1).unsqueeze(-1)
    dy = grid_y.unsqueeze(0).unsqueeze(0) - py.unsqueeze(-1).unsqueeze(-1)
    gt_heatmaps = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    # Heatmap MSE
    hm_loss = F.mse_loss(pred_heatmaps, gt_heatmaps)

    # Coordinate MSE
    coord_loss = F.mse_loss(pred_coords, gt_coords)

    return hm_loss + coord_weight * coord_loss
