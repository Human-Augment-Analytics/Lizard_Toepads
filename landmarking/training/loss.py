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
    mode: str = "ce",
) -> torch.Tensor:
    """Combined heatmap + coordinate loss for heatmap regression models.

    Two modes for the heatmap term:

    ``mode="mse"`` (legacy):
        MSE(pred_heatmaps, gt_heatmaps) + coord_weight * MSE(pred_coords, gt_coords)

        BROKEN IN COMBINATION. The Gaussian target is ~0 in 16320 of 16384 cells at
        heatmap_size=128, so predicting zero everywhere is already near-optimal for
        the MSE term: its total achievable gain is ~4.3e-4 while the coordinate term
        is ~4.3. Measured, the heatmap term is 0.010% of the loss and 0.010% of the
        gradient, so it cannot shape the map at all and training degenerates to pure
        coordinate regression through the decoder. Kept only to reproduce old runs.

    ``mode="ce"`` (default):
        Treats each landmark's heatmap as a DISTRIBUTION and minimizes the cross
        entropy against the normalized (sum-to-1) Gaussian target:
            -sum_x  target_norm(x) * log_softmax(pred)(x)
        This is scale-appropriate (it starts near log(H*W) ~ 9.7 and falls as the
        peak forms), it directly shapes the softmax that the decoder consumes, and
        it is the correct "likelihood" framing for Bayesian fusion. Verified to
        overfit a 4-image batch to ~11px where the MSE combination stalls at ~247px
        against a 258px centre-prediction baseline.

    Args:
        pred_heatmaps: (B, K, H, W) predicted heatmap logits.
        pred_coords: (B, K, 2) decoded coordinates in [0, 1].
        gt_coords: (B, K, 2) ground truth coordinates in [0, 1].
        heatmap_size: Spatial size of target heatmaps (H = W).
        sigma: Gaussian sigma for target heatmap generation.
        coord_weight: Scalar weight for the coordinate loss term.
        mode: "ce" (default) or "mse".

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

    if mode == "mse":
        hm_loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
    elif mode == "ce":
        # Normalize the target into a proper per-landmark distribution, then take
        # cross entropy against the predicted log-distribution.
        tgt = gt_heatmaps.view(B, K, -1)
        tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        logp = F.log_softmax(pred_heatmaps.view(B, K, -1), dim=-1)
        hm_loss = -(tgt * logp).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown heatmap loss mode {mode!r}; expected 'ce' or 'mse'.")

    # Coordinate MSE
    coord_loss = F.mse_loss(pred_coords, gt_coords)

    return hm_loss + coord_weight * coord_loss


def pipnet_loss(
    cls: torch.Tensor,
    off_x: torch.Tensor,
    off_y: torch.Tensor,
    nb_x: torch.Tensor,
    nb_y: torch.Tensor,
    gt_coords: torch.Tensor,
    meanface_indices: torch.Tensor,
    cls_loss_weight: float = 10.0,
    reg_loss_weight: float = 1.0,
):
    """PIPNet composite loss — reference-faithful.

    Reproduces the reference ``gen_target_pip`` (target construction) and
    ``compute_loss_pip`` (gather-at-GT-cell loss) from PIPNet's ``lib``. Targets
    are built on the fly from normalized ground-truth coordinates:

      - Score map target: one-hot, 1 at the ground-truth cell
        ``(mu_y, mu_x) = (floor(y*Hg), floor(x*Wg))`` (clamped to bounds), else 0.
      - Local offset targets at that cell: ``shift_x = x*Wg - mu_x``,
        ``shift_y = y*Hg - mu_y`` (both in [0, 1)).
      - Neighbor offset targets at that cell, channel ``num_nb*i + j``:
        ``nb_x = x_j*Wg - mu_x_i``, ``nb_y = y_j*Hg - mu_y_i`` where j indexes
        landmark i's mean-shape neighbors.

    The score term is MSE over the whole map; the four offset terms are L1
    evaluated ONLY at the ground-truth cell (gathered via the label argmax, as
    the reference does). The total is::

        cls_loss_weight * loss_map
        + reg_loss_weight * (loss_x + loss_y + loss_nb_x + loss_nb_y)

    Args:
        cls: (B, N, Hg, Wg) predicted score map (raw logits).
        off_x: (B, N, Hg, Wg) predicted within-cell x offsets (raw).
        off_y: (B, N, Hg, Wg) predicted within-cell y offsets (raw).
        nb_x: (B, num_nb*N, Hg, Wg) predicted neighbor x offsets (raw).
        nb_y: (B, num_nb*N, Hg, Wg) predicted neighbor y offsets (raw).
        gt_coords: (B, N, 2) ground-truth coordinates in [0, 1].
        meanface_indices: (N, num_nb) long neighbor indices.
        cls_loss_weight: Weight for the score-map MSE term (default 10).
        reg_loss_weight: Weight for each L1 offset term (default 1).

    Returns:
        A tuple ``(total, loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y)`` where
        ``total`` is the scalar weighted sum and the rest are the unweighted
        component losses (useful for logging and tests).
    """
    b, n, gh, gw = cls.shape
    num_nb = meanface_indices.shape[1]
    device = cls.device

    gx = gt_coords[:, :, 0]  # (B, N) in [0,1]
    gy = gt_coords[:, :, 1]

    # Ground-truth cell (matches gen_target_pip: floor then clamp to bounds).
    mu_x = torch.clamp(torch.floor(gx * gw).long(), 0, gw - 1)  # (B, N)
    mu_y = torch.clamp(torch.floor(gy * gh).long(), 0, gh - 1)
    g_flat = mu_y * gw + mu_x  # (B, N) flat cell index

    # --- Score-map target: one-hot at the GT cell ---
    target_map = torch.zeros(b, n, gh * gw, device=device)
    target_map.scatter_(2, g_flat.unsqueeze(-1), 1.0)
    target_map = target_map.view(b, n, gh, gw)
    loss_map = F.mse_loss(cls, target_map)

    # --- Local offset targets (at the GT cell) ---
    target_x = gx * gw - mu_x.float()  # (B, N) in [0,1)
    target_y = gy * gh - mu_y.float()

    # Gather predictions at the GT cell: reshape (B*N, Hg*Wg), gather flat index.
    g_bn = g_flat.view(b * n, 1)
    off_x_sel = torch.gather(off_x.view(b * n, -1), 1, g_bn).view(b, n)
    off_y_sel = torch.gather(off_y.view(b * n, -1), 1, g_bn).view(b, n)
    loss_x = F.l1_loss(off_x_sel, target_x)
    loss_y = F.l1_loss(off_y_sel, target_y)

    # --- Neighbor offset targets (at the GT cell) ---
    # neighbor coords: gather gt of each landmark's neighbors -> (B, N, num_nb)
    nb_idx = meanface_indices.to(device)  # (N, num_nb)
    nb_idx_b = nb_idx.unsqueeze(0).expand(b, n, num_nb)  # (B, N, num_nb)
    nbr_x = torch.gather(
        gx.unsqueeze(1).expand(b, n, n), 2, nb_idx_b
    )  # (B, N, num_nb)
    nbr_y = torch.gather(gy.unsqueeze(1).expand(b, n, n), 2, nb_idx_b)

    target_nb_x = nbr_x * gw - mu_x.float().unsqueeze(-1)  # (B, N, num_nb)
    target_nb_y = nbr_y * gh - mu_y.float().unsqueeze(-1)

    # Predicted neighbor offsets: nb_x is (B, num_nb*N, Hg, Wg) with channel
    # layout num_nb*i + j (reference). Gather at the same GT cell per (i, j).
    # Reshape to (B, N, num_nb, Hg*Wg) then gather the GT cell of landmark i.
    nb_x_r = nb_x.view(b, n, num_nb, gh * gw)
    nb_y_r = nb_y.view(b, n, num_nb, gh * gw)
    g_idx_nb = g_flat.view(b, n, 1, 1).expand(b, n, num_nb, 1)  # (B,N,num_nb,1)
    nb_x_sel = torch.gather(nb_x_r, 3, g_idx_nb).squeeze(-1)  # (B, N, num_nb)
    nb_y_sel = torch.gather(nb_y_r, 3, g_idx_nb).squeeze(-1)
    loss_nb_x = F.l1_loss(nb_x_sel, target_nb_x)
    loss_nb_y = F.l1_loss(nb_y_sel, target_nb_y)

    total = cls_loss_weight * loss_map + reg_loss_weight * (
        loss_x + loss_y + loss_nb_x + loss_nb_y
    )
    return total, loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y


def pipnet_star_loss(
    cls: torch.Tensor,
    off_x: torch.Tensor,
    off_y: torch.Tensor,
    nb_x: torch.Tensor,
    nb_y: torch.Tensor,
    sigma: torch.Tensor,
    gt_coords: torch.Tensor,
    meanface_indices: torch.Tensor,
    input_size: int,
    net_stride: int,
    cls_loss_weight: float = 10.0,
    reg_loss_weight: float = 1.0,
    star_weight: float = 1.0,
    star_omega: float = 1.0,
    star_eigenvalue_clamp: float = 6.0,
):
    """PIPNet loss + STAR term on the decoded coordinates (Option A).

    The paper-faithful PIPNet loss (score-map MSE + L1 offset/neighbor terms) is
    computed UNCHANGED via ``pipnet_loss``. On top of it, a STAR term is added:
    the coordinates are decoded (argmax cell + within-cell offset), a per-landmark
    anisotropic covariance is read from ``sigma`` at each landmark's ground-truth
    cell, and ``star_loss`` reweights the decoded-coordinate error to down-weight
    the semantically ambiguous direction.

    Gathering sigma at the GROUND-TRUTH cell (not the argmax cell) matches how the
    offsets are supervised and keeps the term well-defined when the classifier is
    still wrong early in training. The decode's argmax index is detached (as in
    ``decode_pip``), so STAR's gradient flows through the offset values and the
    sigma head, not through the discrete cell selection.

    Args:
        cls, off_x, off_y: (B, N, Hg, Wg) score / offset maps.
        nb_x, nb_y: (B, num_nb*N, Hg, Wg) neighbor-offset maps.
        sigma: (B, 3*N, Hg, Wg) Cholesky-param map from the sigma head.
        gt_coords: (B, N, 2) ground-truth coords in [0, 1].
        meanface_indices: (N, num_nb) long neighbor indices.
        input_size, net_stride: for decoding coords.
        cls_loss_weight, reg_loss_weight: PIPNet term weights.
        star_weight: weight on the STAR coordinate term.
        star_omega, star_eigenvalue_clamp: passed to ``star_loss``.

    Returns:
        (total, loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y, loss_star).
    """
    from ..models.pipnet import decode_pip

    # Paper-faithful PIPNet terms, untouched.
    pip_total, loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = pipnet_loss(
        cls, off_x, off_y, nb_x, nb_y, gt_coords, meanface_indices,
        cls_loss_weight=cls_loss_weight, reg_loss_weight=reg_loss_weight,
    )

    b, n, gh, gw = cls.shape

    # Ground-truth cell (same formula as pipnet_loss / gen_target_pip).
    gx = gt_coords[:, :, 0]
    gy = gt_coords[:, :, 1]
    mu_x = torch.clamp(torch.floor(gx * gw).long(), 0, gw - 1)
    mu_y = torch.clamp(torch.floor(gy * gh).long(), 0, gh - 1)
    g_flat = (mu_y * gw + mu_x).view(b, n, 1)  # (B, N, 1)

    # Decode coords (differentiable through offsets; argmax detached).
    pred_coords = decode_pip(cls, off_x, off_y, input_size, net_stride)

    # Gather the 3 Cholesky params at the GT cell. sigma is (B, 3N, Hg, Wg) with
    # channel layout [param, landmark] flattened as 3*i + p? We defined the head
    # as Conv2d(-, 3*N): treat channels as (N, 3) with landmark-major layout to
    # match a simple (B, N, 3, HW) view.
    sigma_r = sigma.view(b, n, 3, gh * gw)  # (B, N, 3, HW)
    g_idx = g_flat.view(b, n, 1, 1).expand(b, n, 3, 1)  # (B, N, 3, 1)
    log_sigma = torch.gather(sigma_r, 3, g_idx).squeeze(-1)  # (B, N, 3)

    loss_star = star_loss(
        pred_coords, gt_coords, log_sigma,
        omega=star_omega, eigenvalue_clamp=star_eigenvalue_clamp,
    )

    total = pip_total + star_weight * loss_star
    return total, loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y, loss_star
