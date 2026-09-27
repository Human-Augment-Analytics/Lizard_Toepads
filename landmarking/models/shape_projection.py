"""Differentiable ASM-style shape projection, and the `heatmap_shape` variant.

Motivation
----------
A heatmap model with N landmarks at resolution H emits ``N * H * H`` values and
is free to produce a configuration no specimen has ever had. The shapes
themselves occupy at most ``2N - 4`` degrees of freedom once translation, scale,
and rotation are factored out: 14 for Lizard's N=9, described by 147,456 heatmap
outputs.

This module constrains the decoded coordinates to a point-distribution model
fitted offline by ``landmarking.scripts.shape_bound_report``. Each forward pass
fits a similarity transform from the mean shape onto the predicted coordinates,
projects onto the first k principal components, optionally clamps the
coefficients to a plausible range, and reconstructs. The output cannot leave the
learned shape manifold.

The oracle bound measured on Lizard (rotation factored out, held-out, 512
canvas) says what this can buy. Per-landmark px at k components:

    k       0      2      3      6      9
    finger  8.66   6.45   5.18   2.65   1.44
    toe     9.91   5.58   4.47   2.09   1.17

with the unconstrained heatmap model sitting at ~6.5 px, i.e. the effective
fidelity of 2-3 components. The bound is an oracle (it fits the transform and
coefficients to ground truth), so a trained model lands above it -- but the gap
between 6.5 and the k=6..9 rows is the headroom this variant is chasing.

Why closed form rather than SVD
-------------------------------
In 2D the optimal rotation has a closed form, so no SVD is needed. Maximizing
``tr(R^T M)`` over rotations with ``M = ac^T @ bc`` and
``R = [[c, -s], [s, c]]`` gives::

    tr(R^T M) = c * (M00 + M11) + s * (M10 - M01)

so with ``A = M00 + M11`` and ``B = M10 - M01`` the optimum is
``(c, s) = (A, B) / sqrt(A^2 + B^2)`` and ``scale = sqrt(A^2 + B^2) / ||ac||^2``.
This is smooth, always yields a proper rotation (no reflection, matching
``shape_space.procrustes_fit``), and avoids ``torch.linalg.svd``, whose backward
is ill-conditioned when singular values coincide -- which is exactly what
happens early in training when predictions are near-degenerate.

Scope
-----
A single basis is used for the whole batch. Per-class (finger/toe) bases measure
better -- PC1 of a pooled basis carries 79% of variance and is mostly "which
appendage is this" -- but selecting per sample needs class labels threaded into
``forward``, which would break the ``forward(x) -> (heatmaps, coords)`` contract
this variant shares with ``heatmap``. Point ``shape_basis_group`` at a
single-class basis and train one model per class to get that benefit today.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .hrnet_heatmap import HRNetHeatmap
from .registry import register_model

__all__ = ["ShapeProjection", "HRNetHeatmapShape", "load_shape_basis"]


def load_shape_basis(
    path: str, group: str = "all", variant: str = "similarity"
) -> dict:
    """Load one shape basis from an .npz written by ``shape_bound_report``.

    Args:
        path: Path to the .npz.
        group: Group key, e.g. ``"all"``, ``"finger"``, ``"toe"``.
        variant: ``"similarity"`` (rotation factored out) or ``"no_rotation"``.

    Returns:
        Dict with ``mean`` (N, 2), ``components`` (2N, n_components),
        ``explained_variance`` (n_components,), and ``meta``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the requested group/variant is absent, listing what is.
    """
    basis_path = Path(path)
    if not basis_path.exists():
        raise FileNotFoundError(
            f"shape basis not found: {basis_path}. Generate it with "
            f"`python -m landmarking.scripts.shape_bound_report --config ... "
            f"--save-basis {basis_path}`."
        )

    data = np.load(str(basis_path), allow_pickle=False)
    prefix = f"{group}__{variant}__"
    if prefix + "mean" not in data.files:
        available = sorted(
            k[: -len("__mean")] for k in data.files if k.endswith("__mean")
        )
        raise KeyError(
            f"group/variant '{group}/{variant}' not in {basis_path}. "
            f"Available: {available}"
        )

    meta = {}
    if "__meta__" in data.files:
        try:
            meta = json.loads(str(data["__meta__"]))["groups"][group][variant]
        except (KeyError, ValueError):
            meta = {}

    return {
        "mean": data[prefix + "mean"],
        "components": data[prefix + "components"],
        "explained_variance": data[prefix + "explained_variance"],
        "meta": meta,
    }


class ShapeProjection(nn.Module):
    """Project predicted landmark coordinates onto a point-distribution model.

    The basis is fixed (registered as buffers, not parameters): it is a prior
    estimated from ground-truth shapes, and letting gradient descent edit it
    would let the model widen the manifold to admit whatever it already predicts,
    destroying the constraint.

    Args:
        mean: (N, 2) mean shape from ``shape_space.fit_shape_model``.
        components: (2N, n_components) orthonormal PCA basis.
        explained_variance: (n_components,) variance per component, used for
            coefficient clamping.
        num_components: Components to keep (k). Clipped to what the basis has.
        n_sigma: Clamp coefficients to +/- ``n_sigma * sqrt(variance)``, the
            classic Active Shape Model plausibility constraint. Set 0 to
            disable. Note that a saturated coefficient receives zero gradient,
            which is the intended "the prior refuses to go there" behaviour but
            can stall learning if set too tight.
        allow_rotation: Fit rotation as part of the similarity. Should match how
            the basis was fitted. On Lizard the bound showed residual rotation
            is the dominant variance direction after the OBB crop, so leaving
            this True and letting the transform absorb it is strongly preferred.
        blend: Output is ``(1 - blend) * predicted + blend * projected``. 1.0 is
            a hard projection (the regime the oracle bound describes); lower
            values make the prior a soft regularizer.
        n_fit_iter: Alternating closed-form refinement steps for the coupled
            (transform, coefficients) fit; see :meth:`forward`. 1 reproduces the
            naive single pass, which is measurably loose. Convergence is linear
            and slow in the worst case -- exactness on synthetic shapes sitting
            1 sigma out on every component improved 3.45 -> 1.17 -> 0.10 px at
            1 / 10 / 30 iterations -- but the quantity that matters converges
            fast: on real held-out shapes the error was already below the
            offline alternating reference by 10 iterations. Cost is trivial
            next to the backbone: 20 iterations measured 11.7 ms forward plus
            backward at batch 8, against 81.7 ms for a single 1x1 conv in the
            head alone.
        eps: Numerical floor for the rotation normalization.
    """

    def __init__(
        self,
        mean: np.ndarray,
        components: np.ndarray,
        explained_variance: np.ndarray,
        num_components: int = 8,
        n_sigma: float = 3.0,
        allow_rotation: bool = True,
        blend: float = 1.0,
        n_fit_iter: int = 20,
        eps: float = 1e-8,
    ):
        super().__init__()

        mean_arr = np.asarray(mean, dtype=np.float32)
        comp_arr = np.asarray(components, dtype=np.float32)
        var_arr = np.asarray(explained_variance, dtype=np.float32)

        if mean_arr.ndim != 2 or mean_arr.shape[1] != 2:
            raise ValueError(f"mean must be (N, 2), got {mean_arr.shape}")
        num_landmarks = mean_arr.shape[0]
        if comp_arr.shape[0] != 2 * num_landmarks:
            raise ValueError(
                f"components must have {2 * num_landmarks} rows for N="
                f"{num_landmarks}, got {comp_arr.shape}"
            )
        if var_arr.shape[0] != comp_arr.shape[1]:
            raise ValueError(
                f"explained_variance length {var_arr.shape[0]} != component "
                f"count {comp_arr.shape[1]}"
            )
        if num_components < 1:
            raise ValueError(f"num_components must be >= 1, got {num_components}")
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        if n_fit_iter < 1:
            raise ValueError(f"n_fit_iter must be >= 1, got {n_fit_iter}")

        k = int(min(num_components, comp_arr.shape[1]))
        self.num_landmarks = num_landmarks
        self.num_components = k
        self.n_sigma = float(n_sigma)
        self.allow_rotation = bool(allow_rotation)
        self.blend = float(blend)
        self.n_fit_iter = int(n_fit_iter)
        self.eps = float(eps)

        self.register_buffer("mean_shape", torch.from_numpy(mean_arr))
        self.register_buffer("components", torch.from_numpy(comp_arr[:, :k].copy()))
        self.register_buffer(
            "coeff_limit",
            torch.from_numpy(
                (self.n_sigma * np.sqrt(np.maximum(var_arr[:k], 0.0))).astype(
                    np.float32
                )
            ),
        )
        # The mean shape is centred at the origin with unit centroid size, but do
        # not rely on that: a basis could come from elsewhere.
        centred = mean_arr - mean_arr.mean(axis=0, keepdims=True)
        self.register_buffer("mean_centred", torch.from_numpy(centred))

    def fit_similarity(self, src: Tensor, coords: Tensor) -> tuple:
        """Closed-form least-squares similarity mapping ``src`` onto ``coords``.

        Args:
            src: (N, 2) or (B, N, 2) source shape, typically the current
                reconstruction in the model frame.
            coords: (B, N, 2) target coordinates.

        Returns:
            Tuple ``(scale, rotation, dst_centroid, src_centroid)`` with shapes
            (B, 1, 1), (B, 2, 2), (B, 1, 2), and broadcastable (…, 1, 2).
        """
        if src.dim() == 2:
            src = src.unsqueeze(0)
        src_centroid = src.mean(dim=-2, keepdim=True)         # (B|1, 1, 2)
        src_c = src - src_centroid                            # (B|1, N, 2)
        dst_centroid = coords.mean(dim=1, keepdim=True)        # (B, 1, 2)
        dst = coords - dst_centroid                           # (B, N, 2)

        # M = src^T @ dst, matching shape_space.procrustes_fit's convention so
        # the torch and numpy paths agree numerically.
        m = torch.einsum("bni,bnj->bij", src_c.expand_as(dst), dst)  # (B, 2, 2)
        denom = (src_c ** 2).sum(dim=(-2, -1)).clamp_min(self.eps)   # (B|1,)

        if self.allow_rotation:
            a = m[:, 0, 0] + m[:, 1, 1]
            b = m[:, 1, 0] - m[:, 0, 1]
            # The epsilon goes INSIDE the sqrt. Clamping afterwards would still
            # evaluate sqrt at exactly 0 for a degenerate prediction (all points
            # coincident), and d/dx sqrt(x) is infinite there, which produced
            # non-finite gradients. Adding eps^2 under the radical keeps the
            # whole expression smooth.
            norm = torch.sqrt(a * a + b * b + self.eps * self.eps)
            cos = a / norm
            sin = b / norm
            rotation = torch.stack(
                [
                    torch.stack([cos, -sin], dim=-1),
                    torch.stack([sin, cos], dim=-1),
                ],
                dim=-2,
            )                                                # (B, 2, 2)
            scale = norm / denom
        else:
            rotation = (
                torch.eye(2, dtype=coords.dtype, device=coords.device)
                .expand(coords.shape[0], 2, 2)
                .contiguous()
            )
            scale = (m[:, 0, 0] + m[:, 1, 1]) / denom

        return scale.reshape(-1, 1, 1), rotation, dst_centroid, src_centroid

    def forward(self, coords: Tensor) -> Tensor:
        """Project coords onto the shape manifold.

        Args:
            coords: (B, N, 2) predicted coordinates, any consistent units.

        Returns:
            (B, N, 2) projected coordinates, same units.

        Raises:
            ValueError: If the landmark count disagrees with the basis.
        """
        if coords.dim() != 3 or coords.shape[1:] != (self.num_landmarks, 2):
            raise ValueError(
                f"coords must be (B, {self.num_landmarks}, 2), got "
                f"{tuple(coords.shape)}"
            )

        batch = coords.shape[0]
        mean_centred = self.mean_centred.to(coords.dtype)
        basis = self.components.to(coords.dtype)
        limit = self.coeff_limit.to(coords.dtype)

        # The transform and the coefficients are coupled: the best transform
        # depends on the reconstruction and vice versa. A single pass (fit the
        # transform to the bare MEAN, then solve coefficients once) is only the
        # first step, and it is measurably loose -- it left 3.4 px of error on
        # shapes lying exactly in the shape space, where a true projection is
        # exact. Alternating closed-form steps removes that. Every step is
        # differentiable, so gradients still flow to the encoder.
        recon_model = mean_centred.unsqueeze(0).expand(batch, -1, -1)
        scale = rotation = dst_centroid = None

        for _ in range(self.n_fit_iter):
            scale, rotation, dst_centroid, src_centroid = self.fit_similarity(
                recon_model, coords
            )

            # Pull the observation back into the model frame. The forward map is
            # `scale * (src - src_c) @ R + dst_c`, so the inverse applies R
            # TRANSPOSED -- "bni,bji->bnj" contracts over R's second index,
            # giving X @ R^T. Guard the scale so a degenerate prediction cannot
            # divide by ~zero, preserving its sign (the no-rotation branch can
            # produce a negative scale for anti-correlated shapes).
            safe_scale = torch.where(
                scale.abs() < self.eps,
                torch.full_like(scale, self.eps),
                scale,
            )
            model_frame = (
                torch.einsum(
                    "bni,bji->bnj",
                    (coords - dst_centroid) / safe_scale,
                    rotation,
                )
                + src_centroid
            )

            # Coefficients, then the ASM plausibility clamp.
            residual = (model_frame - mean_centred).reshape(batch, -1)
            coeffs = residual @ basis
            if self.n_sigma > 0:
                coeffs = torch.clamp(coeffs, -limit, limit)

            recon_model = (
                mean_centred.reshape(1, -1) + coeffs @ basis.T
            ).view(batch, self.num_landmarks, 2)

        # Final transform for the refreshed reconstruction.
        scale, rotation, dst_centroid, src_centroid = self.fit_similarity(
            recon_model, coords
        )

        # Push back into the caller's frame: `scale * (recon - src_c) @ R + dst_c`.
        # "bni,bij->bnj" is plain X @ R, the mirror of the inverse above.
        projected = (
            scale
            * torch.einsum("bni,bij->bnj", recon_model - src_centroid, rotation)
            + dst_centroid
        )

        if self.blend >= 1.0:
            return projected
        return (1.0 - self.blend) * coords + self.blend * projected

    def extra_repr(self) -> str:
        return (
            f"num_landmarks={self.num_landmarks}, k={self.num_components}, "
            f"n_sigma={self.n_sigma}, allow_rotation={self.allow_rotation}, "
            f"blend={self.blend}, n_fit_iter={self.n_fit_iter}"
        )


@register_model("heatmap_shape")
class HRNetHeatmapShape(HRNetHeatmap):
    """HRNet heatmap regression with the decoded coordinates shape-constrained.

    Subclasses :class:`HRNetHeatmap` so the backbone, 4-branch fusion, head, and
    ``decode_coords`` are reused verbatim and the baseline model file stays
    untouched. Only the decoded coordinates change, so the forward contract
    ``forward(x) -> (heatmaps, coords)`` is identical to ``heatmap`` and the
    existing ``heatmap_loss`` training path applies with no changes: the heatmap
    term still shapes the raw map, while the coordinate term now supervises the
    projected output, pushing gradients back through the projection.

    Args:
        num_landmarks: Number of landmarks.
        shape_basis_path: .npz from ``shape_bound_report --save-basis``.
        shape_basis_group: Group key in the basis file (``all``/``finger``/...).
        shape_basis_variant: ``similarity`` or ``no_rotation``.
        shape_components: Components to keep (k).
        shape_n_sigma: ASM coefficient clamp in standard deviations; 0 disables.
        shape_blend: 1.0 for hard projection, lower for a soft prior.
        shape_fit_iters: Alternating refinement steps in the projection.
        Remaining kwargs are forwarded to :class:`HRNetHeatmap`.

    Raises:
        ValueError: If ``shape_basis_path`` is empty, or ``use_star`` is set
            (the STAR head reads uncertainty at the heatmap argmax, which is
            unrelated to the projected coordinate, so combining them would
            supervise two inconsistent readouts).
    """

    def __init__(
        self,
        num_landmarks: int,
        shape_basis_path: str = "",
        shape_basis_group: str = "all",
        shape_basis_variant: str = "similarity",
        shape_components: int = 8,
        shape_n_sigma: float = 3.0,
        shape_blend: float = 1.0,
        shape_fit_iters: int = 20,
        **kwargs,
    ):
        if kwargs.get("use_star", False):
            raise ValueError(
                "heatmap_shape does not support use_star: the STAR head reads "
                "uncertainty at the heatmap argmax cell, which is not where the "
                "projected coordinate lies."
            )
        kwargs.pop("use_star", None)
        super().__init__(num_landmarks=num_landmarks, **kwargs)

        if not shape_basis_path:
            raise ValueError(
                "heatmap_shape requires model.shape_basis_path. Generate one "
                "with `python -m landmarking.scripts.shape_bound_report "
                "--config <cfg> --save-basis <path.npz>`."
            )

        basis = load_shape_basis(
            shape_basis_path, group=shape_basis_group, variant=shape_basis_variant
        )
        if basis["mean"].shape[0] != num_landmarks:
            raise ValueError(
                f"basis has {basis['mean'].shape[0]} landmarks, model expects "
                f"{num_landmarks}"
            )

        self.shape_basis_path = shape_basis_path
        self.shape_basis_group = shape_basis_group
        self.shape_basis_variant = shape_basis_variant
        self.projection = ShapeProjection(
            mean=basis["mean"],
            components=basis["components"],
            explained_variance=basis["explained_variance"],
            num_components=shape_components,
            n_sigma=shape_n_sigma,
            allow_rotation=shape_basis_variant == "similarity",
            blend=shape_blend,
            n_fit_iter=shape_fit_iters,
        )

    def forward(self, x: Tensor) -> tuple:
        """Returns ``(heatmaps, projected_coords)``; coords in [0, 1]."""
        heatmaps, coords = super().forward(x)
        return heatmaps, self.projection(coords)

    def forward_star(self, x: Tensor):
        """Unsupported: see the ``use_star`` note in :meth:`__init__`."""
        raise NotImplementedError(
            "heatmap_shape does not support the STAR head."
        )
