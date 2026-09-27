"""Unit tests for statistical shape-space analysis."""

import numpy as np
import pytest

from landmarking.common.shape_space import (
    Similarity,
    centroid_size,
    effective_shape_rank,
    fit_shape_model,
    generalized_procrustes,
    procrustes_fit,
    project_shape,
    reconstruct_shape,
    reconstruction_bound,
)


def rotation_matrix(theta: float) -> np.ndarray:
    """2x2 proper rotation by theta radians."""
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def similarity_tangent_basis(mean_shape: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the similarity group's tangent space at mean_shape.

    The four directions are translation in x, translation in y, uniform scale
    (the shape itself), and rotation (the shape rotated 90 degrees), all
    flattened as (x0, y0, x1, y1, ...).
    """
    n = mean_shape.shape[0]
    t_x = np.tile([1.0, 0.0], n)
    t_y = np.tile([0.0, 1.0], n)
    t_scale = mean_shape.reshape(-1)
    t_rot = np.stack(
        [-mean_shape[:, 1], mean_shape[:, 0]], axis=1
    ).reshape(-1)
    basis, _ = np.linalg.qr(np.stack([t_x, t_y, t_scale, t_rot], axis=1))
    return basis


def make_rank_r_dataset(
    n_samples: int = 400,
    num_landmarks: int = 9,
    rank: int = 4,
    seed: int = 1,
    amp_max: float = 0.10,
):
    """Synthesize shapes whose variation is (near-)exactly rank-dimensional.

    The modes are projected off the similarity tangent space before use. That
    matters: with arbitrary random modes, generalized Procrustes removes the
    component overlapping the similarity group, so post-alignment variation
    becomes a nonlinear warp of the latent coefficients and reconstruction error
    decays smoothly rather than collapsing at the true rank.

    Orthogonality only holds to FIRST order, though. Because the modes are
    orthonormal and orthogonal to the scale direction (which is the mean shape
    itself), ``mean + M c`` has centroid size ``sqrt(1 + |c|^2)``, so GPA
    rescales by ``1 / sqrt(1 + |c|^2)`` -- a nonlinear function of ``c``. That
    injects leakage into higher components scaling as ``|c|^2``. Measured
    cumulative variance at the true rank: 0.977 at ``amp_max=0.10``
    (``|c|^2 ~ 1.3e-1``), 0.992 at 0.05, 0.9987 at 0.02. At ``amp_max=0.10``
    the leakage is large enough that the r-th principal component captures
    leakage instead of the r-th true mode and the cliff disappears entirely.

    So: pass a small ``amp_max`` when asserting rank recovery, and leave the
    larger default when you want shapes that deviate enough from the mean to
    exercise the iterative fitting path.

    Returns:
        Tuple ``(shapes, mean_shape, modes)`` with shapes of form (S, N, 2),
        each specimen carrying an independent random similarity transform.
    """
    rng = np.random.default_rng(seed)

    mean_shape = rng.normal(size=(num_landmarks, 2))
    mean_shape -= mean_shape.mean(axis=0)
    mean_shape /= np.sqrt((mean_shape ** 2).sum())

    tangent = similarity_tangent_basis(mean_shape)
    raw = rng.normal(size=(2 * num_landmarks, rank))
    modes = raw - tangent @ (tangent.T @ raw)
    modes, _ = np.linalg.qr(modes)

    amplitudes = np.linspace(amp_max, amp_max / 5.0, rank)
    coeffs = rng.normal(size=(n_samples, rank)) * amplitudes
    flat = mean_shape.reshape(-1)[None] + coeffs @ modes.T
    canonical = flat.reshape(n_samples, num_landmarks, 2)

    out = np.empty_like(canonical)
    for i, shape in enumerate(canonical):
        rot = rotation_matrix(rng.uniform(-0.3, 0.3))
        scale = rng.uniform(80.0, 140.0)
        offset = rng.uniform(-50.0, 50.0, size=2)
        out[i] = scale * (shape - shape.mean(axis=0)) @ rot + offset

    return out, mean_shape, modes


class TestCentroidSize:
    def test_known_value(self):
        # Unit square corners about the centroid: each at distance sqrt(0.5).
        square = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        assert centroid_size(square) == pytest.approx(np.sqrt(8.0))

    def test_translation_invariant(self):
        rng = np.random.default_rng(0)
        shape = rng.normal(size=(9, 2))
        shifted = shape + np.array([100.0, -250.0])
        assert centroid_size(shifted) == pytest.approx(centroid_size(shape))

    def test_scales_linearly(self):
        rng = np.random.default_rng(0)
        shape = rng.normal(size=(9, 2))
        assert centroid_size(3.5 * shape) == pytest.approx(3.5 * centroid_size(shape))


class TestProcrustesFit:
    def test_recovers_known_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(9, 2))
        rot = rotation_matrix(0.7)
        dst = 2.5 * (src - src.mean(axis=0)) @ rot + np.array([10.0, -4.0])

        fit = procrustes_fit(src, dst)

        assert fit.scale == pytest.approx(2.5)
        np.testing.assert_allclose(fit.apply(src), dst, atol=1e-9)

    def test_rotation_is_proper(self):
        rng = np.random.default_rng(2)
        src = rng.normal(size=(9, 2))
        # Reflect the target: a reflection is a different anatomical side, not a
        # pose, so the fit must stay a proper rotation rather than matching it.
        dst = src.copy()
        dst[:, 1] *= -1.0

        fit = procrustes_fit(src, dst)

        assert np.linalg.det(fit.rotation) == pytest.approx(1.0)

    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(3)
        src = rng.normal(size=(9, 2))
        dst = 4.0 * src @ rotation_matrix(-1.1) + np.array([3.0, 3.0])

        fit = procrustes_fit(src, dst)

        np.testing.assert_allclose(fit.inverse_apply(fit.apply(src)), src, atol=1e-9)

    def test_no_rotation_keeps_identity(self):
        rng = np.random.default_rng(4)
        src = rng.normal(size=(9, 2))
        dst = 2.0 * src @ rotation_matrix(0.5)

        fit = procrustes_fit(src, dst, allow_rotation=False)

        np.testing.assert_allclose(fit.rotation, np.eye(2), atol=1e-12)

    def test_translation_and_scale_only_is_exact_without_rotation(self):
        rng = np.random.default_rng(5)
        src = rng.normal(size=(9, 2))
        dst = 3.0 * (src - src.mean(axis=0)) + np.array([-7.0, 11.0])

        fit = procrustes_fit(src, dst, allow_rotation=False)

        assert fit.scale == pytest.approx(3.0)
        np.testing.assert_allclose(fit.apply(src), dst, atol=1e-9)

    def test_degenerate_source_raises(self):
        degenerate = np.zeros((9, 2))
        target = np.random.default_rng(0).normal(size=(9, 2))
        with pytest.raises(ValueError, match="zero centroid size"):
            procrustes_fit(degenerate, target)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            procrustes_fit(np.zeros((9, 2)) + 1.0, np.zeros((8, 2)) + 1.0)

    def test_bad_dimensionality_raises(self):
        with pytest.raises(ValueError, match=r"src must be \(N, 2\)"):
            procrustes_fit(np.ones((9, 3)), np.ones((9, 3)))


class TestGeneralizedProcrustes:
    def test_mean_is_normalized(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=50)
        _, mean_shape, _ = generalized_procrustes(shapes)

        assert centroid_size(mean_shape) == pytest.approx(1.0)
        np.testing.assert_allclose(mean_shape.mean(axis=0), 0.0, atol=1e-12)

    def test_output_shape_preserved(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=30)
        aligned, mean_shape, iters = generalized_procrustes(shapes)

        assert aligned.shape == shapes.shape
        assert mean_shape.shape == shapes.shape[1:]
        assert iters >= 1

    def test_identical_shapes_align_identically(self):
        rng = np.random.default_rng(7)
        one = rng.normal(size=(9, 2))
        shapes = np.stack([one] * 12)

        aligned, _, _ = generalized_procrustes(shapes)

        # Every aligned copy must coincide, i.e. zero across-sample variance.
        assert aligned.std(axis=0).max() == pytest.approx(0.0, abs=1e-9)

    def test_bad_dimensionality_raises(self):
        with pytest.raises(ValueError, match=r"shapes must be \(S, N, 2\)"):
            generalized_procrustes(np.ones((5, 9, 3)))

    def test_zero_size_shape_raises(self):
        shapes = np.stack([np.zeros((9, 2)), np.ones((9, 2))])
        with pytest.raises(ValueError, match="zero centroid size"):
            generalized_procrustes(shapes)


class TestEffectiveShapeRank:
    def test_similarity_removes_four_dof(self):
        assert effective_shape_rank(9, allow_rotation=True) == 14

    def test_without_rotation_removes_three_dof(self):
        assert effective_shape_rank(9, allow_rotation=False) == 15

    def test_never_negative(self):
        assert effective_shape_rank(1, allow_rotation=True) == 0


class TestFitShapeModel:
    def test_components_are_orthonormal(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=120)
        model = fit_shape_model(shapes)

        gram = model.components.T @ model.components
        np.testing.assert_allclose(gram, np.eye(model.n_components), atol=1e-8)

    def test_variance_is_descending(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=120)
        model = fit_shape_model(shapes)

        assert np.all(np.diff(model.explained_variance) <= 1e-12)

    def test_component_count_capped_by_samples(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=6, num_landmarks=9)
        model = fit_shape_model(shapes)

        # min(S - 1, 2N) = min(5, 18) = 5
        assert model.n_components == 5

    def test_component_count_capped_by_dimension(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=200, num_landmarks=9)
        model = fit_shape_model(shapes)

        assert model.n_components == 18

    def test_explicit_n_components_respected(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=200)
        model = fit_shape_model(shapes, n_components=5)

        assert model.n_components == 5
        assert model.components.shape == (18, 5)

    def test_cumulative_variance_reaches_one(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=200)
        model = fit_shape_model(shapes)

        assert model.cumulative_variance_ratio[-1] == pytest.approx(1.0)

    def test_max_useful_k_respects_rank(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=200, num_landmarks=9)
        model = fit_shape_model(shapes, allow_rotation=True)

        # 18 components available, but only 2N - 4 = 14 describe shape.
        assert model.max_useful_k() == 14

    def test_metadata_recorded(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=77, num_landmarks=9)
        model = fit_shape_model(shapes, allow_rotation=False)

        assert model.n_train == 77
        assert model.num_landmarks == 9
        assert model.allow_rotation is False
        assert model.effective_rank == 15

    def test_single_shape_raises(self):
        with pytest.raises(ValueError, match="at least 2 shapes"):
            fit_shape_model(np.random.default_rng(0).normal(size=(1, 9, 2)))

    def test_bad_dimensionality_raises(self):
        with pytest.raises(ValueError, match=r"shapes must be \(S, N, 2\)"):
            fit_shape_model(np.ones((10, 9, 3)))


class TestProjectAndReconstruct:
    def test_k_zero_returns_mean_under_similarity(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=60)
        model = fit_shape_model(shapes[:50])

        coeffs, transform = project_shape(shapes[55], model, k=0)
        recon = reconstruct_shape(shapes[55], model, k=0)

        assert coeffs.shape == (0,)
        np.testing.assert_allclose(recon, transform.apply(model.mean), atol=1e-9)

    def test_coefficient_count_matches_k(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=60)
        model = fit_shape_model(shapes[:50])

        coeffs, _ = project_shape(shapes[55], model, k=6)

        assert coeffs.shape == (6,)

    def test_reconstruction_is_similarity_equivariant(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=80)
        model = fit_shape_model(shapes[:60])
        target = shapes[70]

        rot = rotation_matrix(0.4)
        transformed = 3.0 * (target - target.mean(axis=0)) @ rot + np.array([7.0, 2.0])

        err_plain = np.linalg.norm(
            reconstruct_shape(target, model, k=5) - target, axis=1
        ).mean()
        err_transformed = np.linalg.norm(
            reconstruct_shape(transformed, model, k=5) - transformed, axis=1
        ).mean()

        assert err_transformed == pytest.approx(3.0 * err_plain, rel=1e-6)

    def test_full_rank_reconstruction_is_near_exact(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=200, num_landmarks=9)
        model = fit_shape_model(shapes)

        recon = reconstruct_shape(shapes[0], model, k=model.n_components)
        err = np.linalg.norm(recon - shapes[0], axis=1).mean()
        scale = centroid_size(shapes[0])

        assert err / scale < 1e-6

    def test_alternating_fit_is_no_worse_than_single_step(self):
        # project_shape alternates the transform and coefficient fits because the
        # two are coupled. A single step (fit the transform to the bare mean,
        # then the coefficients) leaves the objective above its optimum, which
        # would report a LOOSER bound than achievable -- the dangerous direction
        # for a tool used to decide feasibility.
        shapes, _, _ = make_rank_r_dataset(n_samples=120)
        model = fit_shape_model(shapes[:90])

        worse_count = 0
        for target in shapes[90:110]:
            err_one = np.linalg.norm(
                reconstruct_shape(target, model, k=4, n_fit_iter=1) - target, axis=1
            ).mean()
            err_many = np.linalg.norm(
                reconstruct_shape(target, model, k=4, n_fit_iter=25) - target, axis=1
            ).mean()
            assert err_many <= err_one + 1e-9
            if err_many < err_one - 1e-12:
                worse_count += 1

        # The refinement should actually bite on at least some specimens,
        # otherwise this test would pass trivially on a no-op implementation.
        assert worse_count > 0

    def test_negative_k_raises(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=30)
        model = fit_shape_model(shapes)
        with pytest.raises(ValueError, match="k must be in"):
            project_shape(shapes[0], model, k=-1)

    def test_k_above_available_raises(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=30)
        model = fit_shape_model(shapes)
        with pytest.raises(ValueError, match="k must be in"):
            project_shape(shapes[0], model, k=model.n_components + 1)

    def test_wrong_landmark_count_raises(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=30, num_landmarks=9)
        model = fit_shape_model(shapes)
        with pytest.raises(ValueError, match=r"shape must be \(9, 2\)"):
            project_shape(np.ones((7, 2)), model, k=2)


class TestReconstructionBound:
    def test_monotone_non_increasing_in_k(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=200)
        model = fit_shape_model(shapes[:150])

        bound = reconstruction_bound(shapes[150:], model)

        assert np.all(np.diff(bound["mean_error"]) <= 1e-9)

    def test_recovers_true_rank_with_orthogonalized_modes(self):
        # Small amp_max keeps the |c|^2 similarity leakage negligible, so the
        # synthetic variation really is rank-dimensional and the cliff is a
        # legitimate assertion. See make_rank_r_dataset for the measurements.
        rank = 4
        shapes, _, _ = make_rank_r_dataset(n_samples=400, rank=rank, amp_max=0.02)
        model = fit_shape_model(shapes[:300])

        bound = reconstruction_bound(
            shapes[300:], model, k_values=list(range(rank + 2))
        )
        errors = bound["mean_error"]

        # Going from rank-1 to rank components should collapse the error.
        assert errors[rank] < 0.05 * errors[rank - 1]
        assert errors[rank] < 1e-3 * errors[0]

    def test_no_cliff_when_leakage_is_large(self):
        # The companion to the test above, pinning WHY it needs a small
        # amplitude: at a realistic deformation amplitude the second-order
        # similarity leakage is ~13% of the signal, the r-th component captures
        # leakage rather than the r-th true mode, and no cliff appears. This is a
        # property of the synthetic construction, not a defect in the estimator,
        # and it is also why real morphometric data shows a smooth decay.
        rank = 4
        shapes, _, _ = make_rank_r_dataset(n_samples=400, rank=rank, amp_max=0.10)
        model = fit_shape_model(shapes[:300])

        bound = reconstruction_bound(
            shapes[300:], model, k_values=list(range(rank + 2))
        )
        errors = bound["mean_error"]

        assert errors[rank] > 0.2 * errors[rank - 1]
        assert model.cumulative_variance_ratio[rank - 1] < 0.99

    def test_variance_saturates_at_true_rank(self):
        rank = 4
        shapes, _, _ = make_rank_r_dataset(n_samples=400, rank=rank, amp_max=0.02)
        model = fit_shape_model(shapes[:300])

        # Not exactly 1.0: the construction leaks O(|c|^2) into higher
        # components, measured at ~1.3e-3 for this amplitude.
        assert model.cumulative_variance_ratio[rank - 1] > 0.998

    def test_default_k_values_span_zero_to_max_useful(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=200, num_landmarks=9)
        model = fit_shape_model(shapes[:150])

        bound = reconstruction_bound(shapes[150:], model)

        assert bound["k_values"][0] == 0
        assert bound["k_values"][-1] == model.max_useful_k()

    def test_result_structure_and_lengths(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=100, num_landmarks=9)
        model = fit_shape_model(shapes[:80])
        val = shapes[80:]

        bound = reconstruction_bound(val, model, k_values=[0, 3])

        assert bound["n_eval"] == val.shape[0]
        for key in ("mean_error", "median_error", "p90_error"):
            assert len(bound[key]) == 2
        assert len(bound["per_landmark_mean"]) == 2
        assert len(bound["per_landmark_mean"][0]) == 9
        assert len(bound["per_shape_mean"]) == 2
        assert len(bound["per_shape_mean"][0]) == val.shape[0]

    def test_per_shape_mean_averages_to_mean_error(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=100)
        model = fit_shape_model(shapes[:80])

        bound = reconstruction_bound(shapes[80:], model, k_values=[4])

        assert float(np.mean(bound["per_shape_mean"][0])) == pytest.approx(
            bound["mean_error"][0]
        )

    def test_per_landmark_mean_averages_to_mean_error(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=100)
        model = fit_shape_model(shapes[:80])

        bound = reconstruction_bound(shapes[80:], model, k_values=[4])

        assert float(np.mean(bound["per_landmark_mean"][0])) == pytest.approx(
            bound["mean_error"][0]
        )

    def test_error_scales_with_global_similarity(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=140)
        model = fit_shape_model(shapes[:100])
        val = shapes[100:]

        rot = rotation_matrix(0.4)
        scaled = np.stack(
            [3.0 * (v - v.mean(axis=0)) @ rot + np.array([7.0, 2.0]) for v in val]
        )

        base = reconstruction_bound(val, model, k_values=[3])["mean_error"][0]
        big = reconstruction_bound(scaled, model, k_values=[3])["mean_error"][0]

        assert big == pytest.approx(3.0 * base, rel=1e-6)

    def test_landmark_count_mismatch_raises(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=40, num_landmarks=9)
        model = fit_shape_model(shapes)
        with pytest.raises(ValueError, match="model expects 9 landmarks"):
            reconstruction_bound(np.ones((3, 7, 2)), model)

    def test_bad_dimensionality_raises(self):
        shapes, _, _ = make_rank_r_dataset(n_samples=40)
        model = fit_shape_model(shapes)
        with pytest.raises(ValueError, match=r"shapes must be \(S, N, 2\)"):
            reconstruction_bound(np.ones((3, 9, 3)), model)


class TestHeldOutMattersForValidity:
    def test_in_sample_bound_is_optimistic(self):
        # The module documents that the caller must pass disjoint train/val sets.
        # This pins the reason: evaluating on the training shapes reports a
        # strictly more optimistic number, so an accidental overlap would
        # silently overstate how well a low-k model can do.
        shapes, _, _ = make_rank_r_dataset(n_samples=60, rank=6, seed=11)
        train, val = shapes[:30], shapes[30:]
        model = fit_shape_model(train)

        k = [8]
        in_sample = reconstruction_bound(train, model, k_values=k)["mean_error"][0]
        held_out = reconstruction_bound(val, model, k_values=k)["mean_error"][0]

        assert in_sample < held_out


class TestSimilarityDataclass:
    def test_apply_matches_explicit_formula(self):
        rng = np.random.default_rng(12)
        shape = rng.normal(size=(9, 2))
        transform = Similarity(
            scale=2.0,
            rotation=rotation_matrix(0.25),
            src_centroid=np.array([1.0, 2.0]),
            dst_centroid=np.array([-3.0, 0.5]),
        )

        expected = (
            2.0 * ((shape - np.array([1.0, 2.0])) @ rotation_matrix(0.25))
            + np.array([-3.0, 0.5])
        )

        np.testing.assert_allclose(transform.apply(shape), expected, atol=1e-12)
