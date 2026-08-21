"""Loss functions for landmark detection training.

Provides coordinate-space MSE loss, STAR loss (Self-adapTive Ambiguity
Reduction), and optional graph-regularized distance loss for chain topologies.
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


def star_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    log_sigma: torch.Tensor,
    omega: float = 1.0,
    eigenvalue_clamp: float = 6.0,
) -> torch.Tensor:
    """STAR Loss: Self-adapTive Ambiguity Reduction loss for landmark detection.

    Implements the core idea from Zhou et al., CVPR 2023. The model predicts
    per-landmark anisotropic uncertainty (2×2 covariance via Cholesky
    parameterization). The loss decomposes prediction error into principal
    directions and down-weights the direction with higher predicted variance
    (the semantically ambiguous direction).

    The model outputs a lower-triangular Cholesky factor L per landmark such
    that Σ = L @ L^T. The loss is the negative log-likelihood under the
    predicted Gaussian:

        loss_i = (1/2) * err^T @ Σ^{-1} @ err + (1/2) * log|Σ|

    where err = pred - gt and Σ = L @ L^T.

    The log-determinant term (regularizer) prevents the model from predicting
    infinite variance to trivially reduce the NLL. The eigenvalue_clamp
    prevents premature convergence by bounding log-eigenvalues.

    Args:
        pred: (B, N, 2) predicted landmark coordinates.
        gt: (B, N, 2) ground truth landmark coordinates.
        log_sigma: (B, N, 3) Cholesky parameters [log_L11, L21, log_L22].
            L11 = exp(log_L11), L22 = exp(log_L22), L21 is unconstrained.
        omega: Weight for the log-determinant regularization term.
        eigenvalue_clamp: Maximum absolute value for log-diagonal Cholesky
            elements. Prevents collapse (too certain) or explosion (too
            uncertain). Corresponds to eigenvalue restriction in the paper.

    Returns:
        Scalar STAR loss (mean over batch and landmarks).
    """
    B, N, _ = pred.shape

    # Extract Cholesky parameters with eigenvalue restriction
    log_L11 = log_sigma[:, :, 0].clamp(-eigenvalue_clamp, eigenvalue_clamp)
    L21 = log_sigma[:, :, 1]
    log_L22 = log_sigma[:, :, 2].clamp(-eigenvalue_clamp, eigenvalue_clamp)

    # Build lower-triangular Cholesky factor L: Σ = L @ L^T
    L11 = torch.exp(log_L11)  # (B, N)
    L22 = torch.exp(log_L22)  # (B, N)

    # Compute error vector
    err = pred - gt  # (B, N, 2)
    err_x = err[:, :, 0]  # (B, N)
    err_y = err[:, :, 1]  # (B, N)

    # Solve L @ z = err for z (forward substitution for 2×2 lower triangular)
    # L = [[L11, 0], [L21, L22]]
    # z_x = err_x / L11
    # z_y = (err_y - L21 * z_x) / L22
    z_x = err_x / (L11 + 1e-8)
    z_y = (err_y - L21 * z_x) / (L22 + 1e-8)

    # Mahalanobis distance: err^T @ Σ^{-1} @ err = z^T @ z = z_x² + z_y²
    mahal = z_x ** 2 + z_y ** 2  # (B, N)

    # Log-determinant of Σ: log|Σ| = 2 * log|L| = 2 * (log L11 + log L22)
    log_det = 2.0 * (log_L11 + log_L22)  # (B, N)

    # NLL loss: (1/2) * mahal + (omega/2) * log_det
    per_landmark_loss = 0.5 * mahal + 0.5 * omega * log_det

    return per_landmark_loss.mean()


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
