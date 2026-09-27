"""Statistical shape-space analysis: Procrustes alignment, PCA, and the
oracle reconstruction bound for shape-coefficient landmark models.

Why this module exists
----------------------
A heatmap model with N landmarks at resolution H predicts N*H*H values and is
free to emit a configuration no specimen has ever had. The shapes themselves
occupy at most ``2N - 4`` degrees of freedom once translation, uniform scale,
and rotation are factored out (``2N - 3`` if rotation is treated as real
signal rather than nuisance). For Lizard, N=9: at most 14 DOF described by
147,456 heatmap outputs.

A model that instead predicts ``(similarity transform, k shape coefficients)``
solves a far smaller problem -- but it is also *capped* by how well k
components reconstruct a shape it has never seen. This module measures that
cap before any model is trained:

    reconstruction_error(k) = oracle per-landmark error using k components

``reconstruction_bound`` fits BOTH the similarity transform and the shape
coefficients to the ground-truth shape, so the result is a strict lower bound
on what any such model can achieve even with a perfect image encoder. If the
bound at your chosen k already exceeds the error of an existing unconstrained
model, a k-component shape model cannot win and the basis needs more
components (or must be nonlinear).

Methodology note: the PCA basis must be fitted on TRAIN shapes and the bound
evaluated on HELD-OUT shapes. Fitting and evaluating on the same shapes
reports the in-sample reconstruction error, which is optimistically biased and
tends to zero as k approaches the effective rank regardless of whether the
basis generalizes. ``reconstruction_bound`` does not enforce this -- the
caller is responsible for passing disjoint sets.

Conventions
-----------
- Shapes are ``(N, 2)`` arrays of ``(x, y)``; stacks are ``(S, N, 2)``.
- Units are whatever the caller passes in. Passing canvas pixels makes the
  reported errors directly comparable to ``metrics_lizard.compute_pixel_error``.
- Reflections are never permitted: a mirrored specimen is a different
  anatomical side, not a pose.
- All functions are pure numpy, mirroring ``metrics_lizard.py``.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "Similarity",
    "ShapeModel",
    "centroid_size",
    "procrustes_fit",
    "generalized_procrustes",
    "fit_shape_model",
    "effective_shape_rank",
    "project_shape",
    "reconstruct_shape",
    "reconstruction_bound",
]


# --------------------------------------------------------------------------
# Similarity transforms
# --------------------------------------------------------------------------


@dataclass
class Similarity:
    """A 2-D similarity transform mapping a source shape onto a target frame.

    The forward map is::

        dst = scale * (src - src_centroid) @ rotation + dst_centroid

    Stored in this decomposed form (rather than as a 3x3 matrix) because the
    inverse is then exact and cheap, which matters for ``reconstruction_bound``:
    it reconstructs in the aligned frame and must map back to the original
    coordinate frame to report an error in the caller's units.

    Attributes:
        scale: Uniform scale factor (positive).
        rotation: (2, 2) rotation matrix with determinant +1.
        src_centroid: (2,) centroid of the source shape.
        dst_centroid: (2,) centroid of the target frame.
    """

    scale: float
    rotation: np.ndarray
    src_centroid: np.ndarray
    dst_centroid: np.ndarray

    def apply(self, shape: np.ndarray) -> np.ndarray:
        """Map ``shape`` from the source frame into the target frame."""
        centered = np.asarray(shape, dtype=np.float64) - self.src_centroid
        return self.scale * (centered @ self.rotation) + self.dst_centroid

    def inverse_apply(self, shape: np.ndarray) -> np.ndarray:
        """Map ``shape`` from the target frame back into the source frame."""
        centered = np.asarray(shape, dtype=np.float64) - self.dst_centroid
        return (centered @ self.rotation.T) / self.scale + self.src_centroid


def centroid_size(shape: np.ndarray) -> float:
    """Root sum of squared deviations from the centroid (Procrustes scale).

    Args:
        shape: (N, 2) array of landmark coordinates.

    Returns:
        Centroid size as a float. Zero only for a fully degenerate shape.
    """
    arr = np.asarray(shape, dtype=np.float64)
    centered = arr - arr.mean(axis=0)
    return float(np.sqrt((centered ** 2).sum()))


def procrustes_fit(
    src: np.ndarray, dst: np.ndarray, allow_rotation: bool = True
) -> Similarity:
    """Least-squares similarity aligning ``src`` onto ``dst``.

    Minimizes ``||scale * (src - mu_src) @ R + mu_dst - dst||_F`` over scale and
    (optionally) rotation. Reflections are excluded: if the optimal orthogonal
    matrix has determinant -1, the smaller singular direction is flipped to
    recover a proper rotation.

    Args:
        src: (N, 2) source shape.
        dst: (N, 2) target shape.
        allow_rotation: When False, the rotation is fixed to the identity and
            only translation and scale are fitted. Appropriate when an upstream
            step (e.g. an oriented-bounding-box crop) has already canonicalized
            orientation, so residual rotation is real signal a model must
            predict rather than nuisance to be factored out.

    Returns:
        The fitted :class:`Similarity`.

    Raises:
        ValueError: If shapes are not (N, 2) with matching N, or if ``src`` is
            degenerate (zero centroid size).
    """
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"src must be (N, 2), got {a.shape}")
    if b.shape != a.shape:
        raise ValueError(f"shape mismatch: src {a.shape} vs dst {b.shape}")

    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    ac = a - mu_a
    bc = b - mu_b

    denom = float((ac ** 2).sum())
    if denom <= 0.0:
        raise ValueError("src has zero centroid size; cannot fit a similarity")

    if allow_rotation:
        # Orthogonal Procrustes: maximize tr(R^T M) with M = ac^T bc.
        # SVD M = U S V^T  =>  R = U V^T. Flip to force det(R) = +1.
        m = ac.T @ bc
        u, s, vt = np.linalg.svd(m)
        d = np.sign(np.linalg.det(u @ vt))
        if d < 0:
            u = u.copy()
            u[:, -1] *= -1.0
            s = s.copy()
            s[-1] *= -1.0
        rotation = u @ vt
        scale = float(s.sum() / denom)
    else:
        rotation = np.eye(2)
        scale = float((ac * bc).sum() / denom)

    # A non-positive scale would mean the shapes are anti-correlated; clamp to a
    # tiny positive value so the transform stays invertible. In practice this
    # only fires on degenerate input.
    if scale <= 0.0:
        scale = 1e-12

    return Similarity(
        scale=scale,
        rotation=rotation,
        src_centroid=mu_a,
        dst_centroid=mu_b,
    )


# --------------------------------------------------------------------------
# Generalized Procrustes alignment
# --------------------------------------------------------------------------


def generalized_procrustes(
    shapes: np.ndarray,
    allow_rotation: bool = True,
    n_iter: int = 25,
    tol: float = 1e-10,
) -> tuple:
    """Generalized Procrustes alignment of a stack of shapes.

    Iterates: align every shape to the current mean, recompute the mean, and
    renormalize it to unit centroid size at the origin. Renormalizing the mean
    each round is what keeps the procedure from collapsing: aligning a
    unit-size shape to a unit-size reference has an optimal scale slightly
    below 1 (regression toward the mean), so without renormalization the whole
    configuration shrinks a little every iteration.

    Args:
        shapes: (S, N, 2) stack of shapes.
        allow_rotation: Passed through to :func:`procrustes_fit`.
        n_iter: Maximum iterations.
        tol: Convergence threshold on the Frobenius change in the mean shape.

    Returns:
        Tuple ``(aligned, mean_shape, n_iter_run)`` where ``aligned`` is
        (S, N, 2) in the common frame, ``mean_shape`` is (N, 2) with unit
        centroid size centred at the origin, and ``n_iter_run`` is the number
        of iterations actually performed.

    Raises:
        ValueError: If ``shapes`` is not (S, N, 2) or contains fewer than one
            shape.
    """
    arr = np.asarray(shapes, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"shapes must be (S, N, 2), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("need at least one shape")

    # Pre-normalize: centre at origin, unit centroid size.
    centered = arr - arr.mean(axis=1, keepdims=True)
    sizes = np.sqrt((centered ** 2).sum(axis=(1, 2)))
    if np.any(sizes <= 0.0):
        raise ValueError("at least one shape has zero centroid size")
    normalized = centered / sizes[:, None, None]

    reference = normalized[0].copy()
    aligned = normalized.copy()
    iters_run = 0

    for iteration in range(1, n_iter + 1):
        iters_run = iteration
        aligned = np.stack(
            [
                procrustes_fit(s, reference, allow_rotation=allow_rotation).apply(s)
                for s in normalized
            ]
        )
        new_mean = aligned.mean(axis=0)
        new_mean = new_mean - new_mean.mean(axis=0)
        size = np.sqrt((new_mean ** 2).sum())
        if size <= 0.0:
            raise ValueError("mean shape collapsed during alignment")
        new_mean = new_mean / size

        delta = float(np.sqrt(((new_mean - reference) ** 2).sum()))
        reference = new_mean
        if delta < tol:
            break

    return aligned, reference, iters_run


def effective_shape_rank(num_landmarks: int, allow_rotation: bool = True) -> int:
    """Number of shape DOF remaining after factoring out the similarity group.

    Removing translation costs 2 DOF and uniform scale costs 1. Rotation costs
    a further 1 when it is treated as nuisance (``allow_rotation=True``).

    Args:
        num_landmarks: Number of landmarks N.
        allow_rotation: Whether rotation is factored out.

    Returns:
        ``2N - 4`` when rotation is factored out, else ``2N - 3``; never
        negative.
    """
    nuisance = 4 if allow_rotation else 3
    return max(0, 2 * int(num_landmarks) - nuisance)


# --------------------------------------------------------------------------
# Shape model (mean + PCA basis)
# --------------------------------------------------------------------------


@dataclass
class ShapeModel:
    """A point-distribution model: mean shape plus an orthonormal PCA basis.

    Attributes:
        mean: (N, 2) mean shape, unit centroid size, centred at the origin.
        components: (2N, n_components) orthonormal basis, columns ordered by
            descending explained variance. Column j maps a coefficient to a
            flattened (x0, y0, x1, y1, ...) displacement from the mean.
        explained_variance: (n_components,) variance along each component.
        allow_rotation: Whether rotation was factored out during alignment.
        num_landmarks: N.
        n_train: Number of shapes the model was fitted on.
        effective_rank: ``2N - 4`` or ``2N - 3``; components beyond this index
            describe the similarity group, not shape, and carry ~zero variance.
    """

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    allow_rotation: bool
    num_landmarks: int
    n_train: int
    effective_rank: int

    @property
    def n_components(self) -> int:
        """Number of available components."""
        return int(self.components.shape[1])

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component share of total variance, summing to <= 1."""
        total = float(self.explained_variance.sum())
        if total <= 0.0:
            return np.zeros_like(self.explained_variance)
        return self.explained_variance / total

    @property
    def cumulative_variance_ratio(self) -> np.ndarray:
        """Cumulative explained-variance ratio by component index."""
        return np.cumsum(self.explained_variance_ratio)

    def max_useful_k(self) -> int:
        """Largest k worth evaluating: available components capped at the rank."""
        return int(min(self.n_components, self.effective_rank))


def fit_shape_model(
    shapes: np.ndarray,
    allow_rotation: bool = True,
    n_components: Optional[int] = None,
    n_iter: int = 25,
) -> ShapeModel:
    """Fit a point-distribution model (GPA + PCA) to a stack of shapes.

    Args:
        shapes: (S, N, 2) training shapes, in any units.
        allow_rotation: Whether rotation is factored out as nuisance.
        n_components: Components to retain. Defaults to
            ``min(S - 1, 2N)``, i.e. everything the data can support.
        n_iter: Maximum GPA iterations.

    Returns:
        The fitted :class:`ShapeModel`.

    Raises:
        ValueError: If fewer than two shapes are supplied (PCA needs variance)
            or the stack is malformed.
    """
    arr = np.asarray(shapes, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"shapes must be (S, N, 2), got {arr.shape}")
    n_samples, num_landmarks = arr.shape[0], arr.shape[1]
    if n_samples < 2:
        raise ValueError(f"need at least 2 shapes to fit a basis, got {n_samples}")

    aligned, mean_shape, _ = generalized_procrustes(
        arr, allow_rotation=allow_rotation, n_iter=n_iter
    )

    flat = aligned.reshape(n_samples, -1)
    residuals = flat - mean_shape.reshape(1, -1)

    # SVD of the residuals: right singular vectors are the principal axes.
    # Using SVD rather than an explicit covariance eigendecomposition keeps the
    # small-sample case (S < 2N, which is the norm here) well conditioned.
    _, sing, vt = np.linalg.svd(residuals, full_matrices=False)
    variances = (sing ** 2) / max(n_samples - 1, 1)

    max_available = min(n_samples - 1, 2 * num_landmarks)
    keep = max_available if n_components is None else min(int(n_components), max_available)
    keep = max(keep, 1)

    return ShapeModel(
        mean=mean_shape,
        components=vt[:keep].T.copy(),
        explained_variance=variances[:keep].copy(),
        allow_rotation=allow_rotation,
        num_landmarks=int(num_landmarks),
        n_train=int(n_samples),
        effective_rank=effective_shape_rank(num_landmarks, allow_rotation),
    )


# --------------------------------------------------------------------------
# Projection and reconstruction
# --------------------------------------------------------------------------


def project_shape(
    shape: np.ndarray,
    model: ShapeModel,
    k: int,
    n_fit_iter: int = 8,
    init_transform: Optional[Similarity] = None,
) -> tuple:
    """Fit the oracle similarity and k shape coefficients for one shape.

    Both the transform and the coefficients are fitted to ``shape`` itself,
    which is what makes the resulting error a lower bound rather than a
    prediction.

    The two unknowns are coupled -- the best transform depends on the
    reconstruction, and the best coefficients depend on the transform -- so
    this alternates between them (the classic ASM fitting loop). Fitting the
    transform to the mean shape once and stopping is only the first step of
    that alternation and generally leaves the objective above its optimum,
    which would make the reported bound *looser* than achievable. Since the
    whole purpose here is to decide whether k components can reach a target
    error, erring high is the dangerous direction, so the loop runs to
    convergence.

    The joint objective is NOT convex -- it contains the product of the scale
    and the coefficient vector -- so the alternation finds a local optimum that
    depends on where it starts. ``init_transform`` exists so callers sweeping k
    can seed each fit from the previous k's solution; see
    :func:`reconstruction_bound` for why that matters.

    Args:
        shape: (N, 2) shape in the caller's coordinate frame.
        model: A fitted :class:`ShapeModel`.
        k: Number of components to use; 0 gives the mean shape under the best
            similarity.
        n_fit_iter: Maximum alternating refinement steps. Each step is optimal
            in one block given the other, so the objective is non-increasing.
        init_transform: Optional starting transform. Defaults to aligning the
            bare mean shape onto ``shape``.

    Returns:
        Tuple ``(coefficients, transform)`` where ``coefficients`` is a (k,)
        array and ``transform`` maps the model frame onto the caller's frame.

    Raises:
        ValueError: If ``k`` is negative or exceeds the available components,
            or the landmark count disagrees with the model.
    """
    arr = np.asarray(shape, dtype=np.float64)
    if arr.shape != (model.num_landmarks, 2):
        raise ValueError(
            f"shape must be ({model.num_landmarks}, 2), got {arr.shape}"
        )
    if k < 0 or k > model.n_components:
        raise ValueError(f"k must be in [0, {model.n_components}], got {k}")

    mean_flat = model.mean.reshape(-1)

    # Align the model mean onto the observed shape. Fitting in this direction
    # means the transform maps model frame -> caller frame directly, so no
    # inversion is needed to report error in the caller's units.
    if init_transform is None:
        transform = procrustes_fit(
            model.mean, arr, allow_rotation=model.allow_rotation
        )
    else:
        transform = init_transform
    if k == 0:
        return np.zeros(0), transform

    basis = model.components[:, :k]
    coeffs = np.zeros(k)
    prev_error = np.inf

    for _ in range(max(1, n_fit_iter)):
        # Coefficients given the transform: project the observation, pulled back
        # into the model frame, onto the basis.
        residual = transform.inverse_apply(arr).reshape(-1) - mean_flat
        coeffs = basis.T @ residual

        # Transform given the coefficients: realign using the current
        # reconstruction rather than the bare mean.
        recon_model_frame = (mean_flat + basis @ coeffs).reshape(
            model.num_landmarks, 2
        )
        transform = procrustes_fit(
            recon_model_frame, arr, allow_rotation=model.allow_rotation
        )

        error = float(
            np.linalg.norm(transform.apply(recon_model_frame) - arr, axis=1).mean()
        )
        if prev_error - error <= 1e-12:
            break
        prev_error = error

    return coeffs, transform


def reconstruct_shape(
    shape: np.ndarray,
    model: ShapeModel,
    k: int,
    n_fit_iter: int = 8,
    init_transform: Optional[Similarity] = None,
) -> np.ndarray:
    """Best k-component reconstruction of ``shape``, in the caller's frame.

    Args:
        shape: (N, 2) shape in the caller's coordinate frame.
        model: A fitted :class:`ShapeModel`.
        k: Number of components to use.
        n_fit_iter: Alternating refinement steps, see :func:`project_shape`.
        init_transform: Optional starting transform, see :func:`project_shape`.

    Returns:
        (N, 2) reconstruction, directly comparable to ``shape``.
    """
    recon, _ = _reconstruct_with_transform(
        shape, model, k, n_fit_iter=n_fit_iter, init_transform=init_transform
    )
    return recon


def _reconstruct_with_transform(
    shape: np.ndarray,
    model: ShapeModel,
    k: int,
    n_fit_iter: int = 8,
    init_transform: Optional[Similarity] = None,
) -> tuple:
    """Reconstruct and also return the fitted transform, for warm starting."""
    coeffs, transform = project_shape(
        shape, model, k, n_fit_iter=n_fit_iter, init_transform=init_transform
    )
    flat = model.mean.reshape(-1).copy()
    if k > 0:
        flat = flat + model.components[:, :k] @ coeffs
    return transform.apply(flat.reshape(model.num_landmarks, 2)), transform


def reconstruction_bound(
    shapes: np.ndarray,
    model: ShapeModel,
    k_values: Optional[Sequence[int]] = None,
    n_fit_iter: int = 8,
) -> dict:
    """Oracle reconstruction error as a function of component count.

    For each k, reconstructs every shape using the best-fit similarity and k
    coefficients, then reports the per-landmark Euclidean error. This is a
    strict lower bound on the error of any model that predicts a similarity
    transform plus k shape coefficients.

    ``shapes`` should be disjoint from the shapes ``model`` was fitted on;
    otherwise the result is in-sample and optimistically biased.

    Monotonicity
    ------------
    ``mean_error`` is guaranteed non-increasing in k. That is not automatic: the
    joint fit over (transform, coefficients) is non-convex because it contains
    the product of the scale and the coefficient vector, so an independent
    alternation at each k can converge to a worse local optimum with MORE
    components. Observed on real data whose shapes sit far from the model
    (mixed classes with rotation left in): error rose from 24.2 to 26.2 px
    between k=1 and k=2, which is incoherent for something labelled a bound.

    Two measures, both needed:

    1. Each k is fitted twice, from a cold start (align the bare mean) and from
       a warm start (the transform behind the best result so far), keeping
       whichever is better. Neither start dominates -- measured on real data,
       warm beat cold at some k and lost by ~0.06 px at others -- and for a
       bound, looser is the dangerous direction, so taking the minimum is
       strictly preferable to picking one.
    2. A per-shape running minimum across ascending k. This is legitimate, not
       a cosmetic clamp: the solution found at any k' <= k is feasible at k
       (pad the coefficients with zeros), so a value achieved at k' is an upper
       bound on the true optimum at k. Reporting it therefore keeps the
       quantity a valid bound while making it coherent in k.

    ``k_values`` is sorted internally for the chain to be well defined; results
    come back in the caller's requested order. ``median_error`` and
    ``p90_error`` are order statistics over per-landmark errors and are only
    near-monotone: a shape's mean can fall while one of its landmarks worsens.

    Args:
        shapes: (S, N, 2) held-out shapes in the caller's units.
        model: A fitted :class:`ShapeModel`.
        k_values: Component counts to evaluate. Defaults to
            ``0 .. model.max_useful_k()``.
        n_fit_iter: Alternating refinement steps, see :func:`project_shape`.

    Returns:
        Dict with:
            ``k_values``: list of evaluated k.
            ``mean_error``: list, mean over all landmarks and shapes, per k.
            ``median_error``: list, median over all per-landmark errors, per k.
            ``p90_error``: list, 90th percentile of per-landmark errors, per k.
            ``per_landmark_mean``: list of length len(k_values), each an (N,)
                list of per-landmark mean errors.
            ``per_shape_mean``: list of length len(k_values), each an (S,) list
                of per-shape mean errors (for computing confidence intervals).
            ``n_eval``: number of shapes evaluated.

    Raises:
        ValueError: If ``shapes`` is malformed or disagrees with the model.
    """
    arr = np.asarray(shapes, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"shapes must be (S, N, 2), got {arr.shape}")
    if arr.shape[1] != model.num_landmarks:
        raise ValueError(
            f"model expects {model.num_landmarks} landmarks, got {arr.shape[1]}"
        )

    if k_values is None:
        k_values = list(range(0, model.max_useful_k() + 1))
    requested = [int(k) for k in k_values]

    # Ascending order is required for the running-minimum chain to be valid.
    # Dedupe so a repeated k cannot disturb the ordering assumption.
    ascending = sorted(set(requested))

    n_shapes = arr.shape[0]
    # errors_by_k[k] is the (S, N) per-landmark error matrix at that k.
    errors_by_k = {}
    # Best-so-far state per shape, carried forward across ascending k.
    best_per_landmark = [None] * n_shapes
    best_mean = np.full(n_shapes, np.inf)
    best_transform = [None] * n_shapes

    for k in ascending:
        rows = []
        for i, shape in enumerate(arr):
            candidates = []

            # Cold start: align the bare mean shape.
            recon, transform = _reconstruct_with_transform(
                shape, model, k, n_fit_iter=n_fit_iter, init_transform=None
            )
            candidates.append((recon, transform))

            # Warm start: reuse the transform behind the best result so far.
            if best_transform[i] is not None:
                recon_w, transform_w = _reconstruct_with_transform(
                    shape,
                    model,
                    k,
                    n_fit_iter=n_fit_iter,
                    init_transform=best_transform[i],
                )
                candidates.append((recon_w, transform_w))

            per_landmark, mean_err, chosen_transform = None, np.inf, None
            for recon_c, transform_c in candidates:
                errs = np.linalg.norm(recon_c - shape, axis=1)
                m = float(errs.mean())
                if m < mean_err:
                    per_landmark, mean_err, chosen_transform = errs, m, transform_c

            # Running minimum: a solution achieved at any k' <= k stays feasible
            # at k, so never report worse than the best seen so far.
            if mean_err < best_mean[i]:
                best_per_landmark[i] = per_landmark
                best_mean[i] = mean_err
                best_transform[i] = chosen_transform

            rows.append(best_per_landmark[i])
        errors_by_k[k] = np.stack(rows)

    result = {
        "k_values": requested,
        "mean_error": [],
        "median_error": [],
        "p90_error": [],
        "per_landmark_mean": [],
        "per_shape_mean": [],
        "n_eval": int(arr.shape[0]),
    }

    for k in requested:
        errors = errors_by_k[k]
        result["mean_error"].append(float(errors.mean()))
        result["median_error"].append(float(np.median(errors)))
        result["p90_error"].append(float(np.percentile(errors, 90)))
        result["per_landmark_mean"].append([float(v) for v in errors.mean(axis=0)])
        result["per_shape_mean"].append([float(v) for v in errors.mean(axis=1)])

    return result
