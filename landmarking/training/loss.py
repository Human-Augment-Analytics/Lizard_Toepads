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


def heatmap_star_loss(
    pred_heatmaps: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    sigma: torch.Tensor,
    heatmap_size: int,
    sigma_gauss: float = 1.5,
    coord_weight: float = 100.0,
    mode: str = "ce",
    star_weight: float = 1.0,
    star_omega: float = 1.0,
    star_eigenvalue_clamp: float = 6.0,
):
    """HRNet heatmap loss + STAR term on the decoded coordinates (Option A).

    The heatmap term (CE or MSE) and coordinate MSE are computed UNCHANGED via
    ``heatmap_loss``. On top, a STAR term reweights the decoded-coordinate error
    by a per-landmark anisotropic covariance read from ``sigma`` at each
    landmark's PREDICTED (argmax) heatmap cell.

    Gathering sigma at the argmax cell (rather than a GT cell) is the natural
    choice for a dense heatmap model: the coordinate the STAR term supervises is
    itself decoded from that peak, so the uncertainty is read where the model
    localized. The argmax index is detached, so STAR's gradient flows through the
    sigma values and the decoded coordinate, not the discrete cell pick.

    Args:
        pred_heatmaps: (B, N, Hs, Ws) predicted heatmap logits.
        pred_coords: (B, N, 2) decoded coords in [0, 1].
        gt_coords: (B, N, 2) ground-truth coords in [0, 1].
        sigma: (B, 3*N, Hs, Ws) Cholesky-param map from the sigma head.
        heatmap_size: spatial size of the heatmap (H = W).
        sigma_gauss: Gaussian sigma for the heatmap target (passed to heatmap_loss).
        coord_weight: coord-term weight inside heatmap_loss.
        mode: "ce" (default) or "mse" for the heatmap term.
        star_weight: weight on the STAR coordinate term.
        star_omega, star_eigenvalue_clamp: passed to ``star_loss``.

    Returns:
        (total, hm_coord_loss, loss_star).
    """
    b, n, hs, ws = pred_heatmaps.shape

    hm_coord = heatmap_loss(
        pred_heatmaps, pred_coords, gt_coords, heatmap_size,
        sigma=sigma_gauss, coord_weight=coord_weight, mode=mode,
    )

    # Gather the 3 Cholesky params at each landmark's argmax cell (detached idx).
    flat = pred_heatmaps.view(b, n, -1)
    argmax_idx = flat.argmax(dim=-1).detach()  # (B, N)
    sigma_r = sigma.view(b, n, 3, hs * ws)  # (B, N, 3, HW)
    g_idx = argmax_idx.view(b, n, 1, 1).expand(b, n, 3, 1)  # (B, N, 3, 1)
    log_sigma = torch.gather(sigma_r, 3, g_idx).squeeze(-1)  # (B, N, 3)

    loss_star = star_loss(
        pred_coords, gt_coords, log_sigma,
        omega=star_omega, eigenvalue_clamp=star_eigenvalue_clamp,
    )

    total = hm_coord + star_weight * loss_star
    return total, hm_coord, loss_star


def _global_soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Full-map soft-argmax used for the CASCADE coordinate TRAINING term.

    Deliberately NOT the windowed decoder. The windowed soft-argmax takes the
    spatial expectation only inside a window around the DETACHED argmax cell, so
    its coordinate gradient cannot move a prediction out of a wrong (e.g. corner)
    cell — the argmax pins the window and the coord loss then only sharpens
    whatever cell was chosen. Combined with a large coord weight and per-stage
    supervision, that self-locks landmarks whose argmax drifts to a border cell
    (the observed (0,0) corner spikes with no recovery after a few epochs).

    Global soft-argmax has a full-map gradient: a corner prediction is pulled
    toward the true location by the coordinate loss, so it does not trap. It is
    used ONLY as the differentiable coordinate readout for the cascade's training
    loss; inference still decodes with the windowed decoder (unbiased readout).
    """
    b, n, h, w = heatmaps.shape
    device = heatmaps.device
    weights = F.softmax(heatmaps.view(b, n, -1), dim=-1).view(b, n, h, w)
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    x = (weights.sum(dim=2) * xs.view(1, 1, w)).sum(dim=-1)
    y = (weights.sum(dim=3) * ys.view(1, 1, h)).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def cascade_heatmap_loss(
    stage_heatmaps,
    stage_coords,
    gt_coords: torch.Tensor,
    heatmap_size: int,
    sigma: float = 1.5,
    coord_weight: float = 1.0,
    mode: str = "ce",
    stage_weights=None,
):
    """Intermediate-supervision loss for the hrnet_cascade variant.

    Applies the framework's existing ``heatmap_loss`` (default CE) to EVERY
    refinement stage against the same ground truth, then returns the weighted
    mean over stages. Every stage is supervised as both a heatmap and its decoded
    coordinates, so a poor stage is penalized in place rather than silently
    corrupting the final output.

    Two deliberate differences from a naive per-stage ``heatmap_loss`` call, both
    to fix an observed degeneracy (landmarks snapping to the (0,0) corner and not
    recovering after ~epoch 5):

    1. The coordinate term is computed from a GLOBAL soft-argmax of the stage
       heatmap (``_global_soft_argmax``), NOT the windowed decode passed in
       ``stage_coords``. The windowed decode's gradient is trapped in a window
       around a possibly-wrong argmax cell and cannot pull a corner prediction
       back; the global readout has a full-map gradient and does not trap.
    2. ``coord_weight`` defaults to 1.0 (not 100.0). At weight 100 the trapped
       coordinate term dominated the loss and reinforced whatever cell the argmax
       landed in; the CE term (which shapes the whole map, globally) should lead.

    ``stage_coords`` is accepted for interface compatibility but is NOT used for
    the loss (inference decoding still uses the windowed decoder elsewhere).

    Args:
        stage_heatmaps: list of K tensors, each (B, N, Hs, Ws).
        stage_coords: list of K tensors — accepted but unused (see above).
        gt_coords: (B, N, 2) ground-truth coords in [0, 1].
        heatmap_size: spatial size of the heatmap target (H = W).
        sigma: Gaussian sigma for the heatmap target.
        coord_weight: coordinate-term weight (default 1.0).
        mode: "ce" (default) or "mse".
        stage_weights: optional list of K weights; None => equal weighting.

    Returns:
        (total, per_stage_losses) where total is the weighted mean scalar and
        per_stage_losses is a list of the K unweighted component losses.
    """
    k = len(stage_heatmaps)
    if k == 0:
        raise ValueError("cascade_heatmap_loss requires at least one stage")
    if stage_weights is None or len(stage_weights) == 0:
        stage_weights = [1.0] * k
    if len(stage_weights) != k:
        raise ValueError(
            f"stage_weights length {len(stage_weights)} != num stages {k}"
        )

    per_stage = []
    weighted_sum = 0.0
    for w, hm in zip(stage_weights, stage_heatmaps):
        # Decode the coord term from a GLOBAL soft-argmax of THIS stage's heatmap
        # (full-map gradient, no argmax trap). Ignores the passed windowed
        # stage_coords for the loss.
        co = _global_soft_argmax(hm)
        ls = heatmap_loss(
            hm, co, gt_coords, heatmap_size,
            sigma=sigma, coord_weight=coord_weight, mode=mode,
        )
        per_stage.append(ls)
        weighted_sum = weighted_sum + w * ls

    total = weighted_sum / float(sum(stage_weights))
    return total, per_stage
