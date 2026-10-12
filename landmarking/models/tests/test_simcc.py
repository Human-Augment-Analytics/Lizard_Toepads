"""Tests for the SimCC variant: decoders, bin conventions, loss, and wiring.

CPU-only and backbone-free wherever possible. The decoder and loss tests build
synthetic logits so they assert on the MATH rather than on a trained model, in
the same spirit as the measured decoder table in docs/graph_prior_theory.md.
"""

import json
from pathlib import Path

import pytest
import torch

from landmarking.config.schema import LandmarkingConfig
from landmarking.models.hrnet_simcc import (
    _AxisUpsample,
    decode_simcc,
    hard_argmax_1d,
    soft_argmax_1d,
    windowed_soft_argmax_1d,
)
from landmarking.training.loss import _gaussian_target_1d, simcc_loss


def _peaked_logits(coords_norm, num_bins, scale=10.0, sigma_bins=6.0):
    """Gaussian-shaped logits peaked at the given normalized coordinates.

    Uses the same bin-centre convention as the decoders: coordinate c sits at
    bin index c * num_bins - 0.5.
    """
    coords = torch.as_tensor(coords_norm, dtype=torch.float32)
    centre = coords * num_bins - 0.5
    pos = torch.arange(num_bins, dtype=torch.float32)
    d = pos.view(*([1] * centre.dim()), -1) - centre.unsqueeze(-1)
    return scale * torch.exp(-(d ** 2) / (2 * sigma_bins ** 2))


class TestBinConvention:
    """The decoder and the loss target must agree on where a bin sits."""

    @pytest.mark.parametrize("num_bins", [64, 128, 512])
    def test_hard_argmax_recovers_bin_centre(self, num_bins):
        # A one-hot at bin i must decode to exactly (i + 0.5) / num_bins.
        for i in (0, 1, num_bins // 3, num_bins - 1):
            logits = torch.full((1, 1, num_bins), -10.0)
            logits[0, 0, i] = 10.0
            got = hard_argmax_1d(logits)[0, 0].item()
            assert got == pytest.approx((i + 0.5) / num_bins)

    def test_target_peak_matches_decode(self):
        """_gaussian_target_1d must peak at the bin hard_argmax_1d decodes.

        This is the invariant that keeps the loss and the readout consistent; a
        half-bin disagreement here would bias every prediction by 0.5px at the
        Lizard defaults.
        """
        num_bins = 512
        gt = torch.tensor([[0.1, 0.25, 0.5, 0.777, 0.99]])
        tgt = _gaussian_target_1d(gt, num_bins, sigma=2.0)
        # Treat the target itself as logits: its argmax bin should decode back
        # to within half a bin of the coordinate that generated it.
        decoded = hard_argmax_1d(torch.log(tgt.clamp_min(1e-20)))
        assert torch.allclose(decoded, gt, atol=0.5 / num_bins + 1e-6)

    def test_target_rows_sum_to_one(self):
        tgt = _gaussian_target_1d(torch.rand(4, 9), 256, sigma=6.0)
        assert torch.allclose(tgt.sum(dim=-1), torch.ones(4, 9), atol=1e-5)


class TestWindowedDecoderBeatsGlobal:
    """The centre-bias pathology documented for 2D soft-argmax applies in 1-D."""

    def test_global_is_centre_biased_at_realistic_scale(self):
        """At trained-logit scale the global readout collapses toward 0.5.

        docs/graph_prior_theory.md measured this in 2D (265px at canvas 1024).
        The same arithmetic holds per axis, which is why "windowed" is the
        default here rather than a plain softmax expectation.
        """
        num_bins = 512
        gt = torch.tensor([[0.15, 0.85]])
        logits = _peaked_logits(gt, num_bins, scale=1.0, sigma_bins=6.0)

        err_global = (soft_argmax_1d(logits) - gt).abs().mean().item()
        err_window = (windowed_soft_argmax_1d(logits, radius=8) - gt).abs().mean().item()

        # Global drags both points toward the centre; windowed does not.
        assert err_global > 0.2, f"expected centre collapse, got {err_global}"
        assert err_window < 0.01, f"windowed should be accurate, got {err_window}"

    @pytest.mark.parametrize("scale", [0.5, 1.0, 5.0, 10.0])
    def test_windowed_accurate_across_logit_scales(self, scale):
        """Windowed decode stays sub-pixel across the realistic logit range."""
        num_bins = 512
        gt = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        logits = _peaked_logits(gt, num_bins, scale=scale, sigma_bins=6.0)
        err_px = ((windowed_soft_argmax_1d(logits, radius=8) - gt).abs() * 512).mean()
        assert err_px.item() < 1.0, f"scale {scale}: {err_px.item():.3f}px"

    def test_windowed_is_differentiable_in_logit_values(self):
        logits = _peaked_logits(torch.tensor([[0.3]]), 256, scale=5.0)
        logits.requires_grad_(True)
        windowed_soft_argmax_1d(logits, radius=8).sum().backward()
        assert logits.grad is not None
        assert int((logits.grad != 0).sum()) > 0

    def test_decode_simcc_rejects_unknown_mode(self):
        lx = torch.zeros(1, 2, 32)
        with pytest.raises(ValueError, match="Unknown simcc decode mode"):
            decode_simcc(lx, lx, mode="bogus")

    @pytest.mark.parametrize("mode", ["windowed", "global", "hard"])
    def test_decode_simcc_shape_and_range(self, mode):
        lx = _peaked_logits(torch.tensor([[0.3, 0.7]]), 128, scale=5.0)
        ly = _peaked_logits(torch.tensor([[0.6, 0.2]]), 128, scale=5.0)
        coords = decode_simcc(lx, ly, mode=mode, radius=8)
        assert coords.shape == (1, 2, 2)
        assert coords.min() >= 0.0 and coords.max() <= 1.0


class TestAxisUpsample:
    """Bin-count arithmetic and the identity-ish initialization."""

    @pytest.mark.parametrize("up", [2, 4, 8])
    @pytest.mark.parametrize("length", [64, 128])
    def test_output_length_is_exactly_length_times_up(self, up, length):
        mod = _AxisUpsample(num_landmarks=9, up=up)
        out = mod(torch.randn(2, 9, length))
        assert out.shape == (2, 9, length * up)

    def test_preserves_peak_location_at_init(self):
        """At init the layer is a smooth upsample, so the peak must not move.

        This is the property that lets training start from the backbone's actual
        spatial evidence instead of from a uniform distribution.
        """
        up, length = 4, 128
        mod = _AxisUpsample(num_landmarks=1, up=up)
        marginal = _peaked_logits(torch.tensor([[0.3]]), length, scale=10.0, sigma_bins=2.0)
        with torch.no_grad():
            out = mod(marginal)
        before = hard_argmax_1d(marginal)[0, 0].item()
        after = hard_argmax_1d(out)[0, 0].item()
        # Within one coarse cell.
        assert abs(after - before) < 1.0 / length

    def test_landmarks_stay_independent(self):
        """groups=K means landmark j's marginal cannot affect landmark i's."""
        mod = _AxisUpsample(num_landmarks=3, up=4)
        a = torch.zeros(1, 3, 32)
        b = a.clone()
        b[0, 2, :] = 5.0  # perturb only landmark 2
        with torch.no_grad():
            out_a, out_b = mod(a), mod(b)
        assert torch.allclose(out_a[0, :2], out_b[0, :2], atol=1e-6)
        assert not torch.allclose(out_a[0, 2], out_b[0, 2], atol=1e-6)


class TestSimccLoss:
    def test_decreases_as_prediction_approaches_truth(self):
        num_bins = 512
        gt = torch.tensor([[0.3, 0.6]])
        gt_coords = torch.stack([gt, gt], dim=-1)  # (1, 2, 2)

        def loss_for(offset):
            pred = gt + offset
            lx = _peaked_logits(pred, num_bins, scale=5.0)
            ly = _peaked_logits(pred, num_bins, scale=5.0)
            coords = decode_simcc(lx, ly, mode="windowed", radius=8)
            return simcc_loss(lx, ly, coords, gt_coords, sigma=6.0)[0].item()

        assert loss_for(0.0) < loss_for(0.02) < loss_for(0.08)

    def test_components_are_finite_and_nonnegative(self):
        gt_coords = torch.rand(3, 9, 2)
        lx = torch.randn(3, 9, 256)
        ly = torch.randn(3, 9, 256)
        coords = decode_simcc(lx, ly, radius=8)
        total, ce_x, ce_y, coord = simcc_loss(lx, ly, coords, gt_coords)
        for t in (total, ce_x, ce_y, coord):
            assert torch.isfinite(t)
            assert t.item() >= 0.0

    def test_gradient_reaches_logits(self):
        gt_coords = torch.rand(2, 9, 2)
        lx = torch.randn(2, 9, 128, requires_grad=True)
        ly = torch.randn(2, 9, 128, requires_grad=True)
        coords = decode_simcc(lx, ly, radius=8)
        simcc_loss(lx, ly, coords, gt_coords)[0].backward()
        assert lx.grad is not None and int((lx.grad != 0).sum()) > 0
        assert ly.grad is not None and int((ly.grad != 0).sum()) > 0

    def test_ce_is_scale_appropriate_at_init(self):
        """Uniform logits should give CE near log(num_bins), not ~0.

        The 2D MSE heatmap term failed precisely because it started negligible
        (0.004% of the loss). This asserts the CE framing does not have that
        problem: an untrained head starts at the entropy of the bin axis.
        """
        num_bins = 512
        gt_coords = torch.rand(4, 9, 2)
        lx = torch.zeros(4, 9, num_bins)
        ly = torch.zeros(4, 9, num_bins)
        coords = decode_simcc(lx, ly, radius=8)
        _, ce_x, ce_y, _ = simcc_loss(lx, ly, coords, gt_coords)
        import math
        expected = math.log(num_bins)
        assert ce_x.item() == pytest.approx(expected, rel=0.01)
        assert ce_y.item() == pytest.approx(expected, rel=0.01)


class TestRegistryAndConfig:
    def test_simcc_is_registered(self):
        from landmarking.models.registry import MODEL_REGISTRY
        assert "simcc" in MODEL_REGISTRY

    def test_rejects_odd_upsample(self):
        from landmarking.models.registry import get_model
        with pytest.raises(ValueError, match="even integer"):
            get_model("simcc", num_landmarks=9, pretrained=False, simcc_upsample=3)

    @pytest.mark.parametrize("name", [
        "lizard_simcc", "lizard_simcc_up8", "lizard_simcc_native",
        "probe_heatmap_hm64", "probe_heatmap_hm256",
    ])
    def test_experiment_configs_load(self, name):
        path = (
            Path(__file__).resolve().parents[2]
            / "config" / "experiments" / "simcc" / f"{name}.json"
        )
        assert path.exists(), f"missing config: {path}"
        cfg = LandmarkingConfig.from_json(str(path))
        assert cfg.dataset.name == "lizard"
        assert cfg.dataset.num_landmarks == 9

    @pytest.mark.parametrize("name", ["lizard_simcc", "lizard_simcc_native"])
    def test_one_bin_per_canvas_pixel(self, name):
        """Headline claim of both 512-bin arms: one bin per canvas pixel."""
        path = (
            Path(__file__).resolve().parents[2]
            / "config" / "experiments" / "simcc" / f"{name}.json"
        )
        cfg = LandmarkingConfig.from_json(str(path))
        assert cfg.model.variant == "simcc"
        num_bins = cfg.model.heatmap_size * cfg.model.simcc_upsample
        assert num_bins == cfg.dataset.input_size

    def test_arms_hold_physical_window_and_sigma_constant(self):
        """Resolution arms must differ in bins ONLY, not in physical scales.

        If an arm changes bin count without rescaling simcc_decode_radius and
        simcc_sigma, it silently changes the decode window and target width too,
        and the comparison stops being single-variable.
        """
        base = Path(__file__).resolve().parents[2] / "config" / "experiments" / "simcc"
        physical = {}
        for name in ("lizard_simcc", "lizard_simcc_up8", "lizard_simcc_native"):
            cfg = LandmarkingConfig.from_json(str(base / f"{name}.json"))
            bins = cfg.model.heatmap_size * cfg.model.simcc_upsample
            px_per_bin = cfg.dataset.input_size / bins
            physical[name] = (
                cfg.model.simcc_decode_radius * px_per_bin,
                cfg.model.simcc_sigma * px_per_bin,
            )

        ref = physical["lizard_simcc"]
        assert ref == (8.0, 6.0), ref
        for name, got in physical.items():
            assert got == pytest.approx(ref), (
                f"{name} physical (window_px, sigma_px)={got} != {ref}"
            )

    def test_simcc_ab_control_differs_only_in_variant_and_simcc_keys(self):
        """Guard the single-variable claim in the config comment."""
        base = Path(__file__).resolve().parents[2] / "config" / "experiments"
        simcc = json.loads((base / "simcc" / "lizard_simcc.json").read_text())
        ctrl = json.loads((base / "shape_prior" / "lizard_heatmap_baseline.json").read_text())

        allowed_model = {"variant", "sigma", "decode_mode", "decode_radius",
                         "heatmap_use_star", "simcc_upsample",
                         "simcc_decode_mode", "simcc_decode_radius",
                         "simcc_sigma"}
        for key in set(simcc["model"]) | set(ctrl["model"]):
            if key in allowed_model:
                continue
            assert simcc["model"].get(key) == ctrl["model"].get(key), (
                f"model.{key} differs between the SimCC arm and its control"
            )

        allowed_training = {"heatmap_loss_mode", "simcc_coord_weight",
                            "flip_prob"}
        for key in set(simcc["training"]) | set(ctrl["training"]):
            if key in allowed_training:
                continue
            assert simcc["training"].get(key) == ctrl["training"].get(key), (
                f"training.{key} differs between the SimCC arm and its control"
            )

        assert simcc["dataset"] == ctrl["dataset"]


class TestEngineDispatch:
    def test_engine_builds_simcc_kwargs(self, monkeypatch):
        """setup()'s kwargs block must forward every simcc_* field.

        Asserted against get_model rather than by training, so the test stays
        CPU-only and needs no backbone download.
        """
        from landmarking.models import registry

        captured = {}

        def fake_get_model(variant, **kwargs):
            captured["variant"] = variant
            captured.update(kwargs)
            raise RuntimeError("stop-after-kwargs")

        monkeypatch.setattr(registry, "get_model", fake_get_model)
        monkeypatch.setattr("landmarking.training.engine.get_model", fake_get_model)

        cfg = LandmarkingConfig.from_dict({
            "dataset": {"name": "lizard", "num_landmarks": 9, "input_size": 512,
                        "data_dir": "/nonexistent"},
            "model": {"variant": "simcc", "heatmap_size": 128, "simcc_upsample": 4,
                      "simcc_decode_radius": 7, "bn_momentum": 0.1},
            "training": {"device": "cpu", "epochs": 1},
        })

        from landmarking.training.engine import TrainingEngine
        engine = TrainingEngine(cfg)
        with pytest.raises(RuntimeError, match="stop-after-kwargs"):
            engine.setup()

        assert captured["variant"] == "simcc"
        assert captured["num_landmarks"] == 9
        assert captured["heatmap_size"] == 128
        assert captured["simcc_upsample"] == 4
        assert captured["simcc_decode_radius"] == 7
        # The GCN-only kwargs must NOT be forwarded.
        assert "num_iters" not in captured
        assert "gnn_hidden" not in captured

    def test_eval_kwargs_mirror_engine_kwargs(self):
        """A mismatch here silently builds a different architecture than the ckpt."""
        from landmarking.scripts.evaluate import build_model_kwargs

        path = (
            Path(__file__).resolve().parents[2]
            / "config" / "experiments" / "simcc" / "lizard_simcc.json"
        )
        cfg = LandmarkingConfig.from_json(str(path))
        kwargs = build_model_kwargs(cfg)

        assert kwargs["heatmap_size"] == cfg.model.heatmap_size
        assert kwargs["simcc_upsample"] == cfg.model.simcc_upsample
        assert kwargs["simcc_decode_mode"] == cfg.model.simcc_decode_mode
        assert kwargs["simcc_decode_radius"] == cfg.model.simcc_decode_radius
        assert kwargs["pretrained"] is False
