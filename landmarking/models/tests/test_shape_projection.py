"""Tests for the differentiable shape projection and the heatmap_shape variant."""

import numpy as np
import pytest
import torch

from landmarking.common.shape_space import (
    fit_shape_model,
    procrustes_fit,
    reconstruct_shape,
)
from landmarking.models.registry import MODEL_REGISTRY, get_model
from landmarking.models.shape_projection import (
    ShapeProjection,
    load_shape_basis,
)

NUM_LANDMARKS = 9
K = 6


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def toepad_like_shapes(n_samples: int = 200, seed: int = 0) -> np.ndarray:
    """Synthetic shapes in [0, 1] with Lizard-like bilateral structure.

    Four vertically-aligned pairs up a long axis plus an unpaired tip, matching
    the real landmark layout, then a per-specimen similarity so the model has
    pose to factor out.
    """
    rng = np.random.default_rng(seed)
    base = []
    for i in range(4):
        base.append([i * 1.0, 0.45 - 0.05 * i])
        base.append([i * 1.0, -0.45 + 0.05 * i])
    base.append([4.2, 0.0])
    base = np.asarray(base, dtype=np.float64)

    out = np.empty((n_samples, NUM_LANDMARKS, 2))
    for i in range(n_samples):
        shape = base + rng.normal(scale=0.05, size=base.shape)
        theta = rng.normal(scale=0.25)
        out[i] = (
            rng.uniform(0.30, 0.42)
            * (shape - shape.mean(axis=0))
            @ rotation_matrix(theta)
            + np.array([0.5, 0.5])
        )
    return out


@pytest.fixture(scope="module")
def shape_model():
    return fit_shape_model(toepad_like_shapes(200), allow_rotation=True)


@pytest.fixture(scope="module")
def val_shapes():
    return toepad_like_shapes(40, seed=7)


@pytest.fixture
def projection(shape_model):
    return ShapeProjection(
        mean=shape_model.mean,
        components=shape_model.components,
        explained_variance=shape_model.explained_variance,
        num_components=K,
        n_sigma=0.0,
        allow_rotation=True,
    )


@pytest.fixture
def basis_npz(tmp_path, shape_model):
    """A basis .npz written by the REAL exporter.

    Deliberately calls ``shape_bound_report.save_bases`` rather than hand-rolling
    the archive, so a change to the on-disk layout on either side of the
    writer/reader boundary breaks these tests instead of silently producing
    bases the model cannot load.
    """
    from landmarking.scripts.shape_bound_report import save_bases

    path = tmp_path / "shape_basis.npz"
    save_bases(
        {"all": {"similarity": shape_model}},
        str(path),
        extra={"source": "test"},
    )
    return str(path)


class TestConstruction:
    def test_bad_mean_shape_raises(self, shape_model):
        with pytest.raises(ValueError, match=r"mean must be \(N, 2\)"):
            ShapeProjection(
                mean=np.zeros((9, 3)),
                components=shape_model.components,
                explained_variance=shape_model.explained_variance,
            )

    def test_component_row_mismatch_raises(self, shape_model):
        with pytest.raises(ValueError, match="components must have 18 rows"):
            ShapeProjection(
                mean=shape_model.mean,
                components=np.zeros((20, 4)),
                explained_variance=np.ones(4),
            )

    def test_variance_length_mismatch_raises(self, shape_model):
        with pytest.raises(ValueError, match="explained_variance length"):
            ShapeProjection(
                mean=shape_model.mean,
                components=shape_model.components,
                explained_variance=np.ones(shape_model.n_components - 1),
            )

    def test_zero_components_raises(self, shape_model):
        with pytest.raises(ValueError, match="num_components must be >= 1"):
            ShapeProjection(
                mean=shape_model.mean,
                components=shape_model.components,
                explained_variance=shape_model.explained_variance,
                num_components=0,
            )

    def test_blend_out_of_range_raises(self, shape_model):
        with pytest.raises(ValueError, match=r"blend must be in \[0, 1\]"):
            ShapeProjection(
                mean=shape_model.mean,
                components=shape_model.components,
                explained_variance=shape_model.explained_variance,
                blend=1.5,
            )

    def test_zero_fit_iters_raises(self, shape_model):
        with pytest.raises(ValueError, match="n_fit_iter must be >= 1"):
            ShapeProjection(
                mean=shape_model.mean,
                components=shape_model.components,
                explained_variance=shape_model.explained_variance,
                n_fit_iter=0,
            )

    def test_num_components_clipped_to_available(self, shape_model):
        proj = ShapeProjection(
            mean=shape_model.mean,
            components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=999,
        )
        assert proj.num_components == shape_model.n_components

    def test_basis_is_buffers_not_parameters(self, projection):
        # The basis is a prior estimated from ground truth. If it were learnable,
        # gradient descent could widen the manifold to admit whatever the model
        # already predicts, which would silently remove the constraint.
        assert list(projection.parameters()) == []
        names = dict(projection.named_buffers())
        for key in ("mean_shape", "components", "coeff_limit", "mean_centred"):
            assert key in names


class TestSimilarityFit:
    def test_matches_numpy_reference(self, projection, shape_model, val_shapes):
        coords = torch.from_numpy(val_shapes).float()
        scale, rot, _, _ = projection.fit_similarity(projection.mean_centred, coords)

        for i in range(len(val_shapes)):
            ref = procrustes_fit(shape_model.mean, val_shapes[i], allow_rotation=True)
            assert float(scale[i, 0, 0]) == pytest.approx(ref.scale, rel=1e-4)
            np.testing.assert_allclose(rot[i].numpy(), ref.rotation, atol=1e-4)

    def test_rotation_is_proper(self, projection, val_shapes):
        coords = torch.from_numpy(val_shapes).float()
        _, rot, _, _ = projection.fit_similarity(projection.mean_centred, coords)

        np.testing.assert_allclose(rot.det().numpy(), 1.0, atol=1e-5)

    def test_no_rotation_mode_returns_identity(self, shape_model, val_shapes):
        proj = ShapeProjection(
            mean=shape_model.mean,
            components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=K,
            allow_rotation=False,
        )
        coords = torch.from_numpy(val_shapes).float()
        _, rot, _, _ = proj.fit_similarity(proj.mean_centred, coords)

        np.testing.assert_allclose(
            rot.numpy(), np.eye(2)[None].repeat(len(val_shapes), 0), atol=1e-6
        )


class TestProjection:
    def test_output_shape_and_dtype(self, projection, val_shapes):
        coords = torch.from_numpy(val_shapes).float()
        out = projection(coords)

        assert out.shape == coords.shape
        assert out.dtype == coords.dtype

    def test_float64_input_preserved(self, projection, val_shapes):
        out = projection(torch.from_numpy(val_shapes).double())
        assert out.dtype == torch.float64

    def test_wrong_shape_raises(self, projection):
        with pytest.raises(ValueError, match=r"coords must be \(B, 9, 2\)"):
            projection(torch.zeros(2, 7, 2))
        with pytest.raises(ValueError, match=r"coords must be \(B, 9, 2\)"):
            projection(torch.zeros(9, 2))

    def test_similarity_equivariance(self, projection, val_shapes):
        # Projecting a transformed shape must equal transforming the projection.
        # This is the property that makes the layer a geometric operation rather
        # than something tied to the canvas frame.
        coords = torch.from_numpy(val_shapes).float()
        rot = rotation_matrix(0.4)
        offset = np.array([0.05, -0.02])
        moved = np.stack(
            [1.6 * (v - v.mean(axis=0)) @ rot + offset for v in val_shapes]
        )

        with torch.no_grad():
            base = projection(coords).numpy()
            got = projection(torch.from_numpy(moved).float()).numpy()
        expect = np.stack(
            [1.6 * (b - b.mean(axis=0)) @ rot + offset for b in base]
        )

        np.testing.assert_allclose(got, expect, atol=1e-5)

    def test_exact_on_shapes_already_in_the_subspace(self, projection, shape_model):
        # A projection must be a no-op on its own range. Coefficients are set a
        # full standard deviation out on every component, which is a deliberately
        # extreme shape, so this also exercises convergence of the alternation.
        rng = np.random.default_rng(3)
        coeffs = rng.normal(size=(12, K)) * np.sqrt(
            shape_model.explained_variance[:K]
        )
        flat = (
            shape_model.mean.reshape(-1)[None]
            + coeffs @ shape_model.components[:, :K].T
        )
        synth = flat.reshape(12, NUM_LANDMARKS, 2)
        rot = rotation_matrix(0.3)
        placed = np.stack(
            [0.35 * (s - s.mean(axis=0)) @ rot + np.array([0.5, 0.5]) for s in synth]
        )

        with torch.no_grad():
            got = projection(torch.from_numpy(placed).float()).numpy()

        # Tolerance in canvas pixels at 512 rather than normalized units, since
        # that is the scale the result is judged at.
        assert np.abs(got - placed).max() * 512 < 1.0

    def test_approximately_idempotent(self, projection, val_shapes):
        coords = torch.from_numpy(val_shapes).float()
        with torch.no_grad():
            once = projection(coords)
            twice = projection(once)

        assert (twice - once).abs().max().item() * 512 < 1.0

    def test_at_least_as_good_as_offline_reference(
        self, projection, shape_model, val_shapes
    ):
        # The offline numpy path is the same alternating fit used to compute the
        # published oracle bound. The layer should match or beat it, otherwise the
        # bound would overstate what this variant can reach.
        with torch.no_grad():
            got = projection(torch.from_numpy(val_shapes).float()).numpy()
        torch_err = np.linalg.norm(got - val_shapes, axis=2).mean()

        ref = np.stack(
            [reconstruct_shape(v, shape_model, K, n_fit_iter=8) for v in val_shapes]
        )
        numpy_err = np.linalg.norm(ref - val_shapes, axis=2).mean()

        assert torch_err <= numpy_err * 1.02

    @staticmethod
    def _proj_err(shape_model, shapes, iters):
        proj = ShapeProjection(
            mean=shape_model.mean,
            components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=K,
            n_sigma=0.0,
            n_fit_iter=iters,
        )
        with torch.no_grad():
            out = proj(torch.from_numpy(shapes).float()).numpy()
        return float(np.linalg.norm(out - shapes, axis=2).mean())

    def test_iterations_help_on_shapes_far_from_the_mean(self, shape_model):
        # Where the alternation actually matters. On shapes near the mean a single
        # pass is already near-optimal, so testing there proves nothing; these
        # coefficients sit a full standard deviation out on every component.
        rng = np.random.default_rng(9)
        coeffs = rng.normal(size=(16, K)) * np.sqrt(
            shape_model.explained_variance[:K]
        )
        flat = (
            shape_model.mean.reshape(-1)[None]
            + coeffs @ shape_model.components[:, :K].T
        )
        synth = flat.reshape(16, NUM_LANDMARKS, 2)
        rot = rotation_matrix(0.3)
        placed = np.stack(
            [0.35 * (s - s.mean(axis=0)) @ rot + np.array([0.5, 0.5]) for s in synth]
        )

        one = self._proj_err(shape_model, placed, 1)
        many = self._proj_err(shape_model, placed, 20)

        assert many < 0.5 * one

    def test_more_iterations_do_not_hurt(self, shape_model, val_shapes):
        # On ordinary shapes the alternation has effectively converged after one
        # step, so the two agree to float32 precision and the sign of the last
        # digit is noise. Assert no meaningful regression rather than exact
        # ordering.
        one = self._proj_err(shape_model, val_shapes, 1)
        many = self._proj_err(shape_model, val_shapes, 20)

        assert many <= one * (1.0 + 1e-4)

    def test_basis_is_scale_invariant(self, val_shapes):
        # This matters operationally. shape_bound_report fits the basis from .pt
        # crops whose `tps` arrays are in CANVAS PIXELS, while the model consumes
        # coordinates normalized to [0, 1]. That is safe only because
        # generalized_procrustes rescales every shape to unit centroid size
        # before the PCA, so mean/components/variances are all scale-free and the
        # layer's fitted similarity absorbs whatever units arrive. If that ever
        # changed, the coefficient clamp would silently be in the wrong units.
        pixel_shapes = toepad_like_shapes(200) * 512.0
        model_px = fit_shape_model(pixel_shapes, allow_rotation=True)
        model_norm = fit_shape_model(toepad_like_shapes(200), allow_rotation=True)

        coords = torch.from_numpy(val_shapes).float()
        outs = []
        for model in (model_px, model_norm):
            proj = ShapeProjection(
                mean=model.mean,
                components=model.components,
                explained_variance=model.explained_variance,
                num_components=K,
                n_sigma=3.0,
            )
            with torch.no_grad():
                outs.append(proj(coords).numpy())

        np.testing.assert_allclose(outs[0], outs[1], atol=1e-5)

    def test_fewer_components_is_never_better(self, shape_model, val_shapes):
        coords = torch.from_numpy(val_shapes).float()
        errs = []
        for k in (2, 6, 12):
            proj = ShapeProjection(
                mean=shape_model.mean,
                components=shape_model.components,
                explained_variance=shape_model.explained_variance,
                num_components=k,
                n_sigma=0.0,
            )
            with torch.no_grad():
                out = proj(coords).numpy()
            errs.append(np.linalg.norm(out - val_shapes, axis=2).mean())

        assert errs[1] <= errs[0]
        assert errs[2] <= errs[1]


class TestClampAndBlend:
    def test_clamp_constrains_output(self, shape_model, val_shapes):
        coords = torch.from_numpy(val_shapes).float()
        loose = ShapeProjection(
            mean=shape_model.mean, components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=K, n_sigma=0.0,
        )
        tight = ShapeProjection(
            mean=shape_model.mean, components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=K, n_sigma=0.25,
        )
        with torch.no_grad():
            err_loose = np.linalg.norm(
                loose(coords).numpy() - val_shapes, axis=2
            ).mean()
            err_tight = np.linalg.norm(
                tight(coords).numpy() - val_shapes, axis=2
            ).mean()

        # A tight plausibility clamp must restrict the reachable set, so fit
        # quality gets worse. If it did not, the clamp is not being applied.
        assert err_tight > err_loose

    def test_blend_zero_returns_input(self, shape_model, val_shapes):
        proj = ShapeProjection(
            mean=shape_model.mean, components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=K, blend=0.0,
        )
        coords = torch.from_numpy(val_shapes).float()
        with torch.no_grad():
            out = proj(coords)

        torch.testing.assert_close(out, coords, atol=1e-5, rtol=0)

    def test_blend_half_is_the_midpoint(self, shape_model, val_shapes):
        coords = torch.from_numpy(val_shapes).float()
        hard = ShapeProjection(
            mean=shape_model.mean, components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=K, n_sigma=0.0, blend=1.0,
        )
        half = ShapeProjection(
            mean=shape_model.mean, components=shape_model.components,
            explained_variance=shape_model.explained_variance,
            num_components=K, n_sigma=0.0, blend=0.5,
        )
        with torch.no_grad():
            expect = 0.5 * coords + 0.5 * hard(coords)
            got = half(coords)

        torch.testing.assert_close(got, expect, atol=1e-5, rtol=0)


class TestGradients:
    def test_gradient_flows_to_input(self, projection, val_shapes):
        coords = torch.from_numpy(val_shapes[:4]).float().requires_grad_(True)
        projection(coords).square().sum().backward()

        assert coords.grad is not None
        assert torch.isfinite(coords.grad).all()
        assert coords.grad.abs().max() > 0

    def test_degenerate_input_is_finite(self, projection):
        # All landmarks coincident makes the similarity fit degenerate. The
        # epsilon lives INSIDE the sqrt for exactly this case: clamping after the
        # sqrt still evaluates it at 0, whose derivative is infinite.
        coords = torch.full((3, NUM_LANDMARKS, 2), 0.5, requires_grad=True)
        out = projection(coords)
        out.square().sum().backward()

        assert torch.isfinite(out).all()
        assert torch.isfinite(coords.grad).all()

    def test_no_nan_for_tiny_spread(self, projection):
        coords = (
            torch.full((2, NUM_LANDMARKS, 2), 0.5)
            + torch.randn(2, NUM_LANDMARKS, 2) * 1e-7
        ).requires_grad_(True)
        out = projection(coords)
        out.square().sum().backward()

        assert torch.isfinite(out).all()
        assert torch.isfinite(coords.grad).all()


class TestLoadShapeBasis:
    def test_round_trip(self, basis_npz, shape_model):
        basis = load_shape_basis(basis_npz, group="all", variant="similarity")

        np.testing.assert_allclose(basis["mean"], shape_model.mean)
        np.testing.assert_allclose(basis["components"], shape_model.components)
        assert basis["meta"]["num_landmarks"] == NUM_LANDMARKS

    def test_missing_file_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="shape_bound_report"):
            load_shape_basis(str(tmp_path / "nope.npz"))

    def test_unknown_group_lists_available(self, basis_npz):
        with pytest.raises(KeyError) as exc:
            load_shape_basis(basis_npz, group="finger")

        assert "all__similarity" in str(exc.value)


class TestHeatmapShapeVariant:
    def test_registered(self):
        assert "heatmap_shape" in MODEL_REGISTRY

    def test_requires_basis_path(self):
        with pytest.raises(ValueError, match="requires model.shape_basis_path"):
            get_model(
                "heatmap_shape",
                num_landmarks=NUM_LANDMARKS,
                pretrained=False,
                shape_basis_path="",
            )

    def test_rejects_star(self, basis_npz):
        with pytest.raises(ValueError, match="does not support use_star"):
            get_model(
                "heatmap_shape",
                num_landmarks=NUM_LANDMARKS,
                pretrained=False,
                shape_basis_path=basis_npz,
                use_star=True,
            )

    def test_landmark_count_mismatch_raises(self, basis_npz):
        with pytest.raises(ValueError, match="basis has 9 landmarks"):
            get_model(
                "heatmap_shape",
                num_landmarks=19,
                pretrained=False,
                shape_basis_path=basis_npz,
            )

    def test_forward_shapes_and_projection_applied(self, basis_npz):
        model = get_model(
            "heatmap_shape",
            num_landmarks=NUM_LANDMARKS,
            pretrained=False,
            heatmap_size=32,
            shape_basis_path=basis_npz,
            shape_components=K,
            shape_n_sigma=0.0,
        )
        model.eval()
        with torch.no_grad():
            heatmaps, coords = model(torch.randn(2, 3, 64, 64))

        assert heatmaps.shape == (2, NUM_LANDMARKS, 32, 32)
        assert coords.shape == (2, NUM_LANDMARKS, 2)
        # The returned coords must already lie on the manifold, so re-projecting
        # them is (near) a no-op. This is what distinguishes the variant from the
        # plain heatmap model.
        with torch.no_grad():
            again = model.projection(coords)
        assert (again - coords).abs().max().item() < 1e-2

    def test_forward_star_unsupported(self, basis_npz):
        model = get_model(
            "heatmap_shape",
            num_landmarks=NUM_LANDMARKS,
            pretrained=False,
            shape_basis_path=basis_npz,
        )
        with pytest.raises(NotImplementedError):
            model.forward_star(torch.randn(1, 3, 64, 64))

    def test_checkpoint_roundtrip(self, basis_npz):
        kwargs = dict(
            num_landmarks=NUM_LANDMARKS,
            pretrained=False,
            heatmap_size=32,
            shape_basis_path=basis_npz,
            shape_components=K,
        )
        trained = get_model("heatmap_shape", **kwargs)
        rebuilt = get_model("heatmap_shape", **kwargs)

        missing, unexpected = rebuilt.load_state_dict(
            trained.state_dict(), strict=True
        )

        assert not missing and not unexpected

    def test_projection_adds_no_parameters(self, basis_npz):
        kwargs = dict(
            num_landmarks=NUM_LANDMARKS, pretrained=False, heatmap_size=32,
        )
        plain = get_model("heatmap", **kwargs)
        shaped = get_model(
            "heatmap_shape", shape_basis_path=basis_npz, **kwargs
        )

        assert sum(p.numel() for p in shaped.parameters()) == sum(
            p.numel() for p in plain.parameters()
        )


# --------------------------------------------------------------------------- #
# Engine / eval / config integration
# --------------------------------------------------------------------------- #


class _SynthLizard(torch.utils.data.Dataset):
    def __init__(self, n_items, num_lms, input_size):
        g = torch.Generator().manual_seed(7)
        self.imgs = torch.randn(n_items, 3, input_size, input_size, generator=g)
        self.coords = torch.rand(n_items, num_lms, 2, generator=g)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        return (
            self.imgs[i],
            self.coords[i],
            {"orig_size": torch.tensor([512.0, 512.0])},
        )


def _engine_config(tmp_path, basis_npz, heatmap_size, input_size, variant):
    from landmarking.config.schema import LandmarkingConfig

    model = {
        "variant": variant,
        "heatmap_size": heatmap_size,
        "bn_momentum": 0.1,
        "sigma": 1.5,
    }
    if variant == "heatmap_shape":
        model.update({
            "shape_basis_path": basis_npz,
            "shape_basis_group": "all",
            "shape_basis_variant": "similarity",
            "shape_components": K,
            "shape_n_sigma": 3.0,
            "shape_blend": 1.0,
            "shape_fit_iters": 4,
        })
    cfg = LandmarkingConfig.from_dict({
        "paths": {"output_root": str(tmp_path / "runs")},
        "dataset": {
            "name": "lizard",
            "num_landmarks": NUM_LANDMARKS,
            "input_size": input_size,
            "graph_topology": "chain",
        },
        "model": model,
        "training": {
            "epochs": 1, "batch_size": 2, "val_batch_size": 2,
            "lr": 1e-4, "lr_backbone": 1e-4, "device": "cpu",
            "heatmap_loss_mode": "ce",
        },
    })
    cfg.resolve_paths()
    return cfg


class TestEngineIntegration:
    def test_variant_flags_route_through_the_heatmap_coord_path(
        self, tmp_path, basis_npz
    ):
        # heatmap_shape must reuse the existing heatmap-on-coords loss path: the
        # heatmap term shapes the raw map, the coordinate term supervises the
        # projected output. STAR must stay off, since the STAR head reads
        # uncertainty at the heatmap argmax rather than at the projection.
        from landmarking.training.engine import heatmap_family_flags

        cfg = _engine_config(tmp_path, basis_npz, 32, 128, "heatmap_shape")
        flags = heatmap_family_flags(cfg)

        assert flags["is_heatmap_on_coords"] is True
        assert flags["is_heatmap_model"] is False
        assert flags["heatmap_use_star"] is False

    def test_plain_heatmap_routing_is_unchanged(self, tmp_path, basis_npz):
        # Guard against the new variant altering the baseline's dispatch, which
        # would confound the A/B comparison.
        from landmarking.training.engine import heatmap_family_flags

        cfg = _engine_config(tmp_path, basis_npz, 32, 128, "heatmap")
        lizard = heatmap_family_flags(cfg)
        assert lizard == {
            "is_heatmap_model": False,
            "is_heatmap_on_coords": True,
            "heatmap_use_star": False,
        }

        cfg.dataset.name = "wflw"
        wflw = heatmap_family_flags(cfg)
        assert wflw["is_heatmap_model"] is True
        assert wflw["is_heatmap_on_coords"] is False

    def test_star_still_reachable_for_plain_heatmap(self, tmp_path, basis_npz):
        from landmarking.training.engine import heatmap_family_flags

        cfg = _engine_config(tmp_path, basis_npz, 32, 128, "heatmap")
        cfg.model.heatmap_use_star = True

        assert heatmap_family_flags(cfg)["heatmap_use_star"] is True

    def test_star_never_enabled_for_shape_variant(self, tmp_path, basis_npz):
        # Belt and braces: the model constructor also rejects use_star, but the
        # engine must not request it in the first place.
        from landmarking.training.engine import heatmap_family_flags

        cfg = _engine_config(tmp_path, basis_npz, 32, 128, "heatmap_shape")
        cfg.model.heatmap_use_star = True

        assert heatmap_family_flags(cfg)["heatmap_use_star"] is False

    def test_wflw_is_rejected_loudly(self, tmp_path, basis_npz):
        # The WFLW heatmap path trains on pre-generated target heatmaps with a
        # pure heatmap MSE and never supervises decoded coordinates, so the
        # projection would receive no gradient. Silently training an
        # unconstrained model would be worse than failing.
        from landmarking.training.engine import heatmap_family_flags

        cfg = _engine_config(tmp_path, basis_npz, 32, 128, "heatmap_shape")
        cfg.dataset.name = "wflw"

        with pytest.raises(NotImplementedError, match="not wired for WFLW"):
            heatmap_family_flags(cfg)

    def test_train_and_validate(self, tmp_path, basis_npz):
        from pathlib import Path
        from torch.utils.data import DataLoader
        from landmarking.training.engine import TrainingEngine

        heatmap_size, input_size = 32, 128
        cfg = _engine_config(
            tmp_path, basis_npz, heatmap_size, input_size, "heatmap_shape"
        )
        engine = TrainingEngine(cfg)
        engine.output_dir = str(tmp_path / "out")
        Path(engine.output_dir).mkdir(parents=True, exist_ok=True)

        engine.model = get_model(
            "heatmap_shape",
            num_landmarks=NUM_LANDMARKS,
            pretrained=False,
            heatmap_size=heatmap_size,
            shape_basis_path=basis_npz,
            shape_components=K,
            shape_fit_iters=4,
        ).to(engine.device)

        engine._is_heatmap_model = False
        engine._is_graph_cond_heatmap = False
        engine._is_heatmap_on_coords = True
        engine._heatmap_use_star = False
        engine._is_coord_only_model = False
        engine._is_star_model = False
        engine._use_star_loss = False
        engine._is_graph_prior_fusion = False
        engine._is_pipnet = False
        engine._is_cascade = False
        engine.mean_shape = None
        engine.mean_shape_flipped = None
        engine.edge_index = None

        engine.optimizer = torch.optim.Adam(engine.model.parameters(), lr=1e-4)
        engine.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            engine.optimizer, [1]
        )

        ds = _SynthLizard(4, NUM_LANDMARKS, input_size)
        engine.train_loader = DataLoader(ds, batch_size=2)
        engine.val_loader = DataLoader(ds, batch_size=2)

        loss = engine._train_epoch(epoch=1)
        assert np.isfinite(loss)
        metrics = engine._validate(epoch=1)
        assert "val_loss" in metrics and np.isfinite(metrics["val_loss"])
        assert "val_px_err" in metrics

    def test_training_updates_backbone_through_the_projection(
        self, tmp_path, basis_npz
    ):
        # The whole point is that the coordinate loss reaches the encoder THROUGH
        # the projection. If the projection blocked gradients, the head would
        # never move and the variant would be a no-op wrapper.
        model = get_model(
            "heatmap_shape",
            num_landmarks=NUM_LANDMARKS,
            pretrained=False,
            heatmap_size=32,
            shape_basis_path=basis_npz,
            shape_components=K,
            shape_fit_iters=4,
        )
        model.train()
        target = torch.rand(2, NUM_LANDMARKS, 2)
        _, coords = model(torch.randn(2, 3, 64, 64))
        torch.nn.functional.mse_loss(coords, target).backward()

        head_grad = model.head[-1].weight.grad
        assert head_grad is not None
        assert torch.isfinite(head_grad).all()
        assert head_grad.abs().sum() > 0

    def test_eval_mode_zero_gradient_is_shared_with_the_baseline(self, basis_npz):
        # An untrained model in eval() mode has zero head gradient, because the
        # backbone's BatchNorm running stats are still at their defaults and
        # normalize the activations wrongly -- the same train/eval discrepancy
        # already documented in hrnet_heatmap.py. It is NOT caused by the
        # projection, and pinning it here stops it being misdiagnosed as such.
        # Training always runs in train() mode, where gradients are healthy.
        common = dict(num_landmarks=NUM_LANDMARKS, pretrained=False, heatmap_size=32)

        def head_grad_sum(variant, mode, **extra):
            model = get_model(variant, **common, **extra)
            getattr(model, mode)()
            _, coords = model(torch.randn(2, 3, 128, 128))
            torch.nn.functional.mse_loss(
                coords, torch.rand_like(coords)
            ).backward()
            return model.head[-1].weight.grad.abs().sum().item()

        shape_kw = dict(
            shape_basis_path=basis_npz, shape_components=K, shape_fit_iters=4
        )

        assert head_grad_sum("heatmap", "eval") == 0.0
        assert head_grad_sum("heatmap_shape", "eval", **shape_kw) == 0.0
        assert head_grad_sum("heatmap", "train") > 0.0
        assert head_grad_sum("heatmap_shape", "train", **shape_kw) > 0.0


class TestEvalIntegration:
    def test_build_model_kwargs_roundtrip(self, basis_npz):
        # A mismatch here evaluates a different geometry than was trained, and it
        # would be silent because the basis adds no parameters to the checkpoint.
        from landmarking.config.schema import LandmarkingConfig
        from landmarking.scripts.evaluate import build_model_kwargs

        cfg = LandmarkingConfig.from_dict({
            "dataset": {"name": "lizard", "num_landmarks": NUM_LANDMARKS},
            "model": {
                "variant": "heatmap_shape",
                "heatmap_size": 32,
                "shape_basis_path": basis_npz,
                "shape_components": K,
                "shape_n_sigma": 2.0,
                "shape_blend": 0.75,
                "shape_fit_iters": 5,
            },
        })

        kwargs = build_model_kwargs(cfg)
        assert kwargs["shape_basis_path"] == basis_npz
        assert kwargs["shape_components"] == K
        assert kwargs["shape_n_sigma"] == 2.0
        assert kwargs["shape_blend"] == 0.75
        assert kwargs["shape_fit_iters"] == 5

        trained = get_model("heatmap_shape", pretrained=False, **kwargs)
        rebuilt = get_model("heatmap_shape", pretrained=False, **kwargs)
        missing, unexpected = rebuilt.load_state_dict(
            trained.state_dict(), strict=True
        )
        assert not missing and not unexpected

        # The rebuilt model must reproduce the projection settings, not defaults.
        assert rebuilt.projection.num_components == K
        assert rebuilt.projection.n_sigma == 2.0
        assert rebuilt.projection.blend == 0.75
        assert rebuilt.projection.n_fit_iter == 5

    def test_lizard_dispatch_treats_it_as_heatmap_family(self):
        import inspect

        from landmarking.scripts import evaluate

        source = inspect.getsource(evaluate)
        # Both the lizard forward dispatch and the cephalometric is_heatmap flag
        # must include the new variant, else it falls through to a branch with a
        # different forward signature.
        assert 'in ("heatmap", "heatmap_shape")' in source


class TestExperimentConfigs:
    @pytest.mark.parametrize(
        "name,expected_variant",
        [
            ("lizard_heatmap_shape.json", "heatmap_shape"),
            ("lizard_heatmap_baseline.json", "heatmap"),
        ],
    )
    def test_config_loads(self, name, expected_variant):
        from pathlib import Path

        from landmarking.config.schema import LandmarkingConfig

        path = (
            Path(__file__).resolve().parents[2]
            / "config" / "experiments" / "shape_prior" / name
        )
        assert path.exists(), f"missing config: {path}"
        cfg = LandmarkingConfig.from_json(str(path))
        cfg.resolve_paths()
        cfg.validate()

        assert cfg.model.variant == expected_variant
        assert cfg.dataset.num_landmarks == NUM_LANDMARKS
        assert cfg.model.heatmap_size == 128

    def test_ab_pair_differs_only_in_the_variant(self):
        # The comparison is only interpretable if nothing else moved.
        import json
        from pathlib import Path

        base = (
            Path(__file__).resolve().parents[2]
            / "config" / "experiments" / "shape_prior"
        )
        with open(base / "lizard_heatmap_baseline.json") as f:
            control = json.load(f)
        with open(base / "lizard_heatmap_shape.json") as f:
            treatment = json.load(f)

        assert control["dataset"] == treatment["dataset"]
        assert control["training"] == treatment["training"]

        shape_only = {
            "shape_basis_path", "shape_basis_group", "shape_basis_variant",
            "shape_components", "shape_n_sigma", "shape_blend",
            "shape_fit_iters", "variant", "heatmap_use_star",
        }
        differing = {
            k for k in set(control["model"]) | set(treatment["model"])
            if control["model"].get(k) != treatment["model"].get(k)
        }
        assert differing <= shape_only, f"unexpected differences: {differing}"
