"""Unit tests for evaluation metrics.

Tests known input/output pairs and mathematical properties for both
Lizard and WFLW metrics. No model loading or real data.
"""

import numpy as np
import pytest

from landmarking.evaluation.metrics_lizard import (
    compute_pixel_error,
    pixel_to_mm,
    back_project,
)
from landmarking.evaluation.metrics_wflw import (
    compute_nme,
    compute_fr,
    compute_auc,
)


class TestComputePixelError:
    """Test Lizard pixel error computation."""

    def test_zero_for_identical(self):
        """Pixel error is zero when pred == gt."""
        coords = np.random.rand(9, 2) * 512
        err = compute_pixel_error(coords, coords)
        np.testing.assert_array_almost_equal(err, 0.0)

    def test_known_distance(self):
        """Known Euclidean distance: (0,0) to (3,4) = 5."""
        pred = np.array([[3.0, 4.0]])
        gt = np.array([[0.0, 0.0]])
        err = compute_pixel_error(pred, gt)
        assert err[0] == pytest.approx(5.0)

    def test_symmetric(self):
        """Error is symmetric: d(pred, gt) == d(gt, pred)."""
        pred = np.random.rand(9, 2) * 512
        gt = np.random.rand(9, 2) * 512
        err1 = compute_pixel_error(pred, gt)
        err2 = compute_pixel_error(gt, pred)
        np.testing.assert_array_almost_equal(err1, err2)

    def test_non_negative(self):
        """Pixel error is always non-negative."""
        pred = np.random.rand(9, 2) * 512
        gt = np.random.rand(9, 2) * 512
        err = compute_pixel_error(pred, gt)
        assert np.all(err >= 0)


class TestPixelToMm:
    """Test pixel-to-mm conversion."""

    def test_known_conversion(self):
        """10px error with ruler_px=100 and ruler_mm=10 → 1mm."""
        px_err = np.array([10.0])
        mm_err = pixel_to_mm(px_err, ruler_px=100.0, ruler_mm=10.0)
        assert mm_err[0] == pytest.approx(1.0)

    def test_zero_error_stays_zero(self):
        """Zero pixel error maps to zero mm."""
        px_err = np.array([0.0, 0.0])
        mm_err = pixel_to_mm(px_err, ruler_px=50.0, ruler_mm=10.0)
        np.testing.assert_array_almost_equal(mm_err, 0.0)

    def test_raises_on_zero_ruler(self):
        """Should raise ValueError for zero ruler_px."""
        with pytest.raises(ValueError):
            pixel_to_mm(np.array([5.0]), ruler_px=0.0)

    def test_raises_on_negative_ruler(self):
        """Should raise ValueError for negative ruler_px."""
        with pytest.raises(ValueError):
            pixel_to_mm(np.array([5.0]), ruler_px=-10.0)


class TestBackProject:
    """Test back-projection from 512 canvas to original space."""

    def test_identity_transform(self):
        """Identity M + no padding/scaling → unchanged coordinates."""
        coords = np.array([[100.0, 200.0], [300.0, 400.0]])
        M = np.eye(3)
        result = back_project(coords, M, scale=1.0, pad_x=0.0, pad_y=0.0)
        np.testing.assert_array_almost_equal(result, coords)

    def test_undo_padding(self):
        """Padding offset is correctly removed."""
        coords = np.array([[110.0, 215.0]])
        M = np.eye(3)
        result = back_project(coords, M, scale=1.0, pad_x=10.0, pad_y=15.0)
        expected = np.array([[100.0, 200.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_undo_scaling(self):
        """Scale factor is correctly inverted."""
        coords = np.array([[200.0, 200.0]])
        M = np.eye(3)
        result = back_project(coords, M, scale=2.0, pad_x=0.0, pad_y=0.0)
        expected = np.array([[100.0, 100.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeNME:
    """Test WFLW NME computation."""

    def test_zero_for_identical(self):
        """NME is zero when pred == gt."""
        gt = np.random.rand(98, 2) * 512
        # Ensure IOD > 0 by setting landmarks 60 and 72 apart
        gt[60] = [100, 256]
        gt[72] = [400, 256]
        nme = compute_nme(gt, gt)
        assert nme == pytest.approx(0.0)

    def test_returns_none_for_zero_iod(self):
        """NME returns None when IOD is zero."""
        gt = np.random.rand(98, 2) * 512
        gt[60] = gt[72]  # Same position → IOD = 0
        pred = np.random.rand(98, 2) * 512
        nme = compute_nme(pred, gt)
        assert nme is None

    def test_non_negative(self):
        """NME is non-negative for any valid input."""
        gt = np.random.rand(98, 2) * 512
        gt[60] = [100, 256]
        gt[72] = [400, 256]
        pred = np.random.rand(98, 2) * 512
        nme = compute_nme(pred, gt)
        assert nme >= 0.0

    def test_known_value(self):
        """Known NME: all landmarks off by 30px, IOD=300 → NME=0.1."""
        gt = np.zeros((98, 2))
        gt[60] = [0, 0]
        gt[72] = [300, 0]
        pred = gt.copy()
        pred += 30.0  # All landmarks off by 30px in both x and y
        # Per-landmark distance = sqrt(30^2 + 30^2) = 30*sqrt(2) ≈ 42.43
        # NME = mean(42.43) / 300 ≈ 0.1414
        nme = compute_nme(pred, gt)
        expected = 30.0 * np.sqrt(2) / 300.0
        assert nme == pytest.approx(expected, rel=1e-4)


class TestComputeFR:
    """Test WFLW failure rate."""

    def test_all_pass(self):
        """FR = 0 when all NME values below threshold."""
        nme_list = [0.01, 0.02, 0.05, 0.09]
        fr = compute_fr(nme_list, threshold=0.10)
        assert fr == pytest.approx(0.0)

    def test_all_fail(self):
        """FR = 1 when all NME values above threshold."""
        nme_list = [0.11, 0.15, 0.20, 0.30]
        fr = compute_fr(nme_list, threshold=0.10)
        assert fr == pytest.approx(1.0)

    def test_half_fail(self):
        """FR = 0.5 when half fail."""
        nme_list = [0.05, 0.15, 0.03, 0.20]
        fr = compute_fr(nme_list, threshold=0.10)
        assert fr == pytest.approx(0.5)

    def test_in_range(self):
        """FR is always in [0, 1]."""
        nme_list = list(np.random.rand(100) * 0.2)
        fr = compute_fr(nme_list, threshold=0.10)
        assert 0.0 <= fr <= 1.0

    def test_empty_list(self):
        """FR = 0 for empty list."""
        assert compute_fr([]) == 0.0


class TestComputeAUC:
    """Test WFLW AUC computation."""

    def test_perfect_predictions(self):
        """AUC = 1 when all NME = 0."""
        nme_list = [0.0] * 100
        auc = compute_auc(nme_list, threshold=0.10)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_all_failures(self):
        """AUC = 0 when all NME > threshold."""
        nme_list = [0.5] * 100
        auc = compute_auc(nme_list, threshold=0.10)
        assert auc == pytest.approx(0.0)

    def test_in_range(self):
        """AUC is always in [0, 1]."""
        nme_list = list(np.random.rand(100) * 0.2)
        auc = compute_auc(nme_list, threshold=0.10)
        assert 0.0 <= auc <= 1.0

    def test_empty_list(self):
        """AUC = 0 for empty list."""
        assert compute_auc([]) == 0.0

    def test_monotonic_with_quality(self):
        """Better predictions → higher AUC."""
        good = [0.02] * 100
        bad = [0.08] * 100
        auc_good = compute_auc(good, threshold=0.10)
        auc_bad = compute_auc(bad, threshold=0.10)
        assert auc_good > auc_bad


# ---------------------------------------------------------------------------
# Cephalometric metrics tests
# ---------------------------------------------------------------------------

from hypothesis import given, settings, strategies as st

from landmarking.evaluation.metrics_cephalometric import (
    to_original_pixels,
    compute_radial_error_mm,
    compute_mre_sdr,
)
from landmarking.evaluation.engine import evaluate_cephalometric


# Feature: cephalometric-dataset, Property 2: Coordinate normalize / map-back round trip
class TestCephalometricNormalizeRoundTrip:
    """Property 2: normalize by (W, H) then map back recovers pixels.

    Validates: Requirements 2.3, 7.2
    """

    @settings(max_examples=200)
    @given(
        h=st.integers(min_value=2, max_value=4000),
        w=st.integers(min_value=2, max_value=4000),
        data=st.data(),
    )
    def test_normalize_mapback_round_trip(self, h, w, data):
        n = data.draw(st.integers(min_value=1, max_value=19))
        xs = data.draw(
            st.lists(
                st.floats(min_value=0.0, max_value=float(w),
                          allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            )
        )
        ys = data.draw(
            st.lists(
                st.floats(min_value=0.0, max_value=float(h),
                          allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            )
        )
        pixels = np.array(list(zip(xs, ys)), dtype=np.float64)

        # Normalize by width (x) and height (y)
        coords_norm = np.empty_like(pixels)
        coords_norm[:, 0] = pixels[:, 0] / w
        coords_norm[:, 1] = pixels[:, 1] / h

        # Normalized values are within [0, 1]
        assert np.all(coords_norm >= 0.0 - 1e-9)
        assert np.all(coords_norm <= 1.0 + 1e-9)

        # Map back via orig_size = [H, W]
        recovered = to_original_pixels(coords_norm, orig_size=[h, w])
        np.testing.assert_allclose(recovered, pixels, rtol=1e-6, atol=1e-6)


# Feature: cephalometric-dataset, Property 9: MRE / SDR metric invariants
class TestCephalometricMREInvariants:
    """Property 9: MRE / SDR invariants.

    Validates: Requirements 7.2, 7.3, 7.4, 9.4
    """

    @settings(max_examples=200, deadline=None)
    @given(
        n=st.integers(min_value=5, max_value=19),
        h=st.integers(min_value=10, max_value=3000),
        w=st.integers(min_value=10, max_value=3000),
        pixel_spacing=st.floats(min_value=0.01, max_value=5.0,
                                allow_nan=False, allow_infinity=False),
        k=st.floats(min_value=0.1, max_value=10.0,
                    allow_nan=False, allow_infinity=False),
        data=st.data(),
    )
    def test_mre_sdr_invariants(self, n, h, w, pixel_spacing, k, data):
        coord = st.floats(min_value=0.0, max_value=1.0,
                          allow_nan=False, allow_infinity=False)
        pred = np.array(
            data.draw(st.lists(st.tuples(coord, coord), min_size=n, max_size=n)),
            dtype=np.float64,
        )
        gt = np.array(
            data.draw(st.lists(st.tuples(coord, coord), min_size=n, max_size=n)),
            dtype=np.float64,
        )
        orig_size = [h, w]

        errs = compute_radial_error_mm(pred, gt, orig_size, pixel_spacing)

        # Every per-landmark error is non-negative
        assert np.all(errs >= 0.0)

        # MRE is zero when pred == gt
        zero_errs = compute_radial_error_mm(gt, gt, orig_size, pixel_spacing)
        res_zero = compute_mre_sdr(zero_errs)
        assert res_zero["mre"] == pytest.approx(0.0, abs=1e-9)

        # Scaling pixel_spacing by k scales MRE by k
        res = compute_mre_sdr(errs)
        errs_scaled = compute_radial_error_mm(
            pred, gt, orig_size, pixel_spacing * k
        )
        res_scaled = compute_mre_sdr(errs_scaled)
        assert res_scaled["mre"] == pytest.approx(res["mre"] * k, rel=1e-6, abs=1e-9)

        # SDR non-decreasing across increasing thresholds, each in [0, 100]
        sdr = res["sdr"]
        ordered = [sdr["2.0mm"], sdr["2.5mm"], sdr["3.0mm"], sdr["4.0mm"]]
        for val in ordered:
            assert 0.0 <= val <= 100.0
        for a, b in zip(ordered, ordered[1:]):
            assert a <= b + 1e-9

        # Landmark subset behaves consistently: metric normalizes by physical
        # pixel_spacing, not a landmark pair. Any subset's errors are the same
        # subset of the full per-landmark errors.
        subset = data.draw(
            st.lists(st.integers(min_value=0, max_value=n - 1),
                     min_size=1, max_size=n, unique=True)
        )
        subset_errs = compute_radial_error_mm(
            pred[subset], gt[subset], orig_size, pixel_spacing
        )
        np.testing.assert_allclose(subset_errs, errs[subset], rtol=1e-6, atol=1e-6)


class TestEvaluateCephalometricNoNME:
    """Unit test: cephalometric metric result has no 'nme' key.

    Validates: Requirement 7.5
    """

    def test_result_has_no_nme_but_has_mre_and_sdr(self):
        gt = [
            np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float64),
            np.array([[0.2, 0.2], [0.4, 0.6], [0.7, 0.3]], dtype=np.float64),
        ]
        pred = [
            g + 0.001 for g in gt
        ]
        metadata = [
            {"orig_size": np.array([800.0, 640.0]), "pixel_spacing": 0.1,
             "split": "test1"},
            {"orig_size": np.array([1000.0, 900.0]), "pixel_spacing": 0.1,
             "split": "test1"},
        ]

        result = evaluate_cephalometric(pred, gt, metadata)

        assert "nme" not in result
        assert "mre" in result
        assert "sdr" in result
        assert result["mre"] is not None
        assert result["n_evaluated"] == 2

    def test_skips_none_predictions(self):
        gt = [np.array([[0.1, 0.1], [0.5, 0.5]], dtype=np.float64)]
        pred = [None]
        metadata = [{"orig_size": np.array([800.0, 640.0]),
                     "pixel_spacing": 0.1, "split": "test1"}]
        result = evaluate_cephalometric(pred, gt, metadata)
        assert result["n_evaluated"] == 0
        assert result["mre"] is None
        assert "nme" not in result
