"""Unit tests for dataset adapters.

Tests use mock .pt files to verify return shapes without loading real data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from landmarking.datasets.cephalometric.compute_mean_shape import compute_mean_shape
from landmarking.datasets.cephalometric.dataset import CephalometricDataset
from landmarking.datasets.cephalometric.preprocess import (
    average_annotators,
    parse_annotation_file,
    to_three_channel,
)
from landmarking.datasets.lizard.dataset import LizardDataset
from landmarking.datasets.wflw.dataset import WFLWDataset


@pytest.fixture
def lizard_pt_dir(tmp_path):
    """Create mock Lizard .pt files for testing."""
    for i in range(5):
        img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        # Pixel-space coordinates in [0, 512)
        tps = torch.rand(9, 2) * 500 + 5  # keep away from edges
        data = {
            "image": img,
            "tps": tps.float(),
            "orig_size": torch.tensor([300.0, 200.0]),
            "M": torch.eye(3, dtype=torch.float64),
            "scale": torch.tensor(0.8),
            "pad": torch.tensor([10.0, 15.0]),
            "class_name": "finger",
            "ruler_px": torch.tensor(100.0),
        }
        torch.save(data, str(tmp_path / f"sample_{i}.pt"))
    return tmp_path


@pytest.fixture
def wflw_pt_dir(tmp_path):
    """Create mock WFLW .pt files for testing."""
    for i in range(5):
        img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        # Normalized coordinates in [0, 1]
        tps = torch.rand(98, 2)
        attrs = torch.zeros(6, dtype=torch.uint8)
        data = {
            "image": img,
            "tps": tps.float(),
            "attrs": attrs,
            "orig_size": torch.tensor([400, 400], dtype=torch.int32),
        }
        torch.save(data, str(tmp_path / f"face_{i}.pt"))
    return tmp_path


def _write_cephalometric_pt(path, orig_h=1935, orig_w=2400, tps=None):
    """Write one synthetic cephalometric .pt file mirroring the preprocessor."""
    img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    if tps is None:
        tps = torch.rand(19, 2)
    data = {
        "image": img,
        "tps": tps.float(),
        "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
        "pixel_spacing": torch.tensor(0.1, dtype=torch.float32),
        "split": "train",
    }
    torch.save(data, str(path))


@pytest.fixture
def cephalometric_pt_dir(tmp_path):
    """Create mock cephalometric .pt files for testing."""
    for i in range(5):
        _write_cephalometric_pt(tmp_path / f"ceph_{i}.pt")
    return tmp_path


class TestLizardDataset:
    """Test LizardDataset in coord and heatmap modes."""

    def test_coord_mode_shapes(self, lizard_pt_dir):
        """Verify coord mode returns correct shapes."""
        paths = sorted([str(p) for p in lizard_pt_dir.glob("*.pt")])
        ds = LizardDataset(paths, augment=False, mode="coord")

        assert len(ds) == 5

        img, coords, metadata = ds[0]
        assert img.shape == (3, 512, 512)
        assert img.dtype == torch.float32
        assert coords.shape == (9, 2)
        assert coords.dtype == torch.float32
        assert "orig_size" in metadata

    def test_coord_values_normalized(self, lizard_pt_dir):
        """Verify coordinates are in [0, 1] range."""
        paths = sorted([str(p) for p in lizard_pt_dir.glob("*.pt")])
        ds = LizardDataset(paths, augment=False, mode="coord")

        _, coords, _ = ds[0]
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

    def test_heatmap_mode_shapes(self, lizard_pt_dir):
        """Verify heatmap mode returns correct shapes."""
        paths = sorted([str(p) for p in lizard_pt_dir.glob("*.pt")])
        heatmap_size = 128
        ds = LizardDataset(
            paths, augment=False, mode="heatmap", heatmap_size=heatmap_size
        )

        img, coords, heatmaps, metadata = ds[0]
        assert img.shape == (3, 512, 512)
        assert coords.shape == (9, 2)
        assert heatmaps.shape == (9, heatmap_size, heatmap_size)
        assert heatmaps.dtype == torch.float32

    def test_metadata_contains_expected_keys(self, lizard_pt_dir):
        """Verify metadata dict has the expected fields."""
        paths = sorted([str(p) for p in lizard_pt_dir.glob("*.pt")])
        ds = LizardDataset(paths, augment=False, mode="coord")

        _, _, metadata = ds[0]
        assert "orig_size" in metadata
        assert "M" in metadata
        assert "scale" in metadata
        assert "pad" in metadata
        assert "class_name" in metadata
        assert "ruler_px" in metadata

    def test_augmented_output_shapes(self, lizard_pt_dir):
        """Verify augmentation doesn't break output shapes."""
        paths = sorted([str(p) for p in lizard_pt_dir.glob("*.pt")])
        ds = LizardDataset(paths, augment=True, mode="coord")

        img, coords, metadata = ds[0]
        assert img.shape == (3, 512, 512)
        assert coords.shape == (9, 2)


class TestWFLWDataset:
    """Test WFLWDataset with flip-aware augmentation."""

    def test_shapes_no_augment(self, wflw_pt_dir):
        """Verify no-augment mode returns correct shapes."""
        paths = sorted([str(p) for p in wflw_pt_dir.glob("*.pt")])
        ds = WFLWDataset(paths, augment=False)

        assert len(ds) == 5

        img, coords, metadata = ds[0]
        assert img.shape == (3, 512, 512)
        assert img.dtype == torch.float32
        assert coords.shape == (98, 2)
        assert coords.dtype == torch.float32
        assert "orig_size" in metadata
        assert "was_flipped" in metadata
        assert "rot_angle" in metadata

    def test_coord_values_in_range(self, wflw_pt_dir):
        """Verify coordinates remain in [0, 1] after processing."""
        paths = sorted([str(p) for p in wflw_pt_dir.glob("*.pt")])
        ds = WFLWDataset(paths, augment=False)

        _, coords, _ = ds[0]
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

    def test_augmented_output_shapes(self, wflw_pt_dir):
        """Verify augmentation preserves output shapes."""
        paths = sorted([str(p) for p in wflw_pt_dir.glob("*.pt")])
        ds = WFLWDataset(paths, augment=True, flip_prob=0.5)

        img, coords, metadata = ds[0]
        assert img.shape == (3, 512, 512)
        assert coords.shape == (98, 2)

    def test_rotation_augmentation(self, wflw_pt_dir):
        """Verify rotation augmentation preserves shapes."""
        paths = sorted([str(p) for p in wflw_pt_dir.glob("*.pt")])
        ds = WFLWDataset(paths, augment=True, rot_factor=30)

        img, coords, metadata = ds[0]
        assert img.shape == (3, 512, 512)
        assert coords.shape == (98, 2)
        assert "rot_angle" in metadata

    def test_attrs_in_metadata(self, wflw_pt_dir):
        """Verify attributes are passed through in metadata."""
        paths = sorted([str(p) for p in wflw_pt_dir.glob("*.pt")])
        ds = WFLWDataset(paths, augment=False)

        _, _, metadata = ds[0]
        assert "attrs" in metadata
        assert metadata["attrs"].shape == (6,)


# Bounded finite floats suitable for text round-trip at %.6f precision.
_coord_floats = st.floats(
    min_value=-5000.0,
    max_value=5000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _coords_19():
    """Strategy producing a (19, 2) list of bounded (x, y) float pairs."""
    return st.lists(
        st.tuples(_coord_floats, _coord_floats),
        min_size=19,
        max_size=19,
    )


class TestCephalometricPreprocess:
    """Preprocessing helper tests for the cephalometric dataset."""

    # Feature: cephalometric-dataset, Property 3: Annotation parse round-trip and annotator averaging
    @settings(max_examples=150)
    @given(a=_coords_19(), b=_coords_19())
    def test_parse_round_trip_and_averaging(self, a, b):
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        # Write annotator A in the ISBI "x,y" per-line format with enough
        # precision to survive a text round-trip.
        with tempfile.TemporaryDirectory() as tmp:
            ann_path = Path(tmp) / "annotator_a.txt"
            with open(ann_path, "w") as f:
                for x, y in a_arr:
                    f.write(f"{x:.6f},{y:.6f}\n")

            recovered = parse_annotation_file(ann_path)
        assert recovered.shape == (19, 2)
        # Coordinates recovered in the same order within text-precision tolerance.
        np.testing.assert_allclose(recovered, a_arr, atol=1e-4)

        # Annotator averaging is the element-wise mean.
        avg = average_annotators(a_arr, b_arr)
        assert avg.shape == (19, 2)
        np.testing.assert_allclose(avg, (a_arr + b_arr) / 2.0, atol=1e-9)

    # Feature: cephalometric-dataset, Property 4: Grayscale to three-channel replication
    @settings(max_examples=150)
    @given(
        h=st.integers(min_value=1, max_value=16),
        w=st.integers(min_value=1, max_value=16),
        data=st.data(),
    )
    def test_grayscale_to_three_channel(self, h, w, data):
        gray = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=255),
                min_size=h * w,
                max_size=h * w,
            )
        )
        gray_hw = np.asarray(gray, dtype=np.uint8).reshape(h, w)

        out = to_three_channel(gray_hw)
        assert out.shape == (3, h, w)
        assert out.dtype == np.uint8
        for c in range(3):
            np.testing.assert_array_equal(out[c], gray_hw)

    def test_parse_wrong_count_raises_with_file_path(self, tmp_path):
        # 18 rows instead of 19.
        short_path = tmp_path / "too_few.txt"
        with open(short_path, "w") as f:
            for i in range(18):
                f.write(f"{i}.0,{i}.0\n")

        with pytest.raises(ValueError) as excinfo:
            parse_annotation_file(short_path)
        assert str(short_path) in str(excinfo.value)

        # Garbage content.
        garbage_path = tmp_path / "garbage.txt"
        with open(garbage_path, "w") as f:
            f.write("not a coordinate\nhello world\n")

        with pytest.raises(ValueError) as excinfo:
            parse_annotation_file(garbage_path)
        assert str(garbage_path) in str(excinfo.value)


# Feature: cephalometric-dataset, Property 1: Dataset output contract
@settings(max_examples=100, deadline=None)
@given(
    input_size=st.sampled_from([256, 384, 512]),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_cephalometric_output_contract(input_size, seed):
    """Property 1: __getitem__ returns the BaseDataset output contract.

    Validates: Requirements 1.2, 1.4, 1.5, 1.6, 3.1, 3.2, 3.3
    """
    rng = np.random.default_rng(seed)
    tps = torch.from_numpy(rng.random((19, 2)).astype(np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = Path(tmp) / "sample.pt"
        _write_cephalometric_pt(pt_path, tps=tps)
        ds = CephalometricDataset(
            [str(pt_path)], input_size=input_size, augment=False, mode="coord"
        )
        img, coords, metadata = ds[0]

    assert img.shape == (3, input_size, input_size)
    assert img.dtype == torch.float32

    assert coords.shape == (19, 2)
    assert coords.dtype == torch.float32
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0

    assert "orig_size" in metadata
    assert "pixel_spacing" in metadata
    assert "split" in metadata
    # No visibility / occlusion keys for a dataset where all landmarks exist.
    assert "visibility" not in metadata
    assert "visible" not in metadata
    assert "occlusion" not in metadata


# Feature: cephalometric-dataset, Property 8: Coordinate subset slicing
@settings(max_examples=100, deadline=None)
@given(
    subset=st.lists(
        st.integers(min_value=0, max_value=18),
        min_size=1,
        max_size=19,
        unique=True,
    ),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_cephalometric_coordinate_subset_slicing(subset, seed):
    """Property 8: subset coords equal full coords indexed by the subset.

    Matches WFLW `coords[landmark_indices]` semantics.

    Validates: Requirements 9.1
    """
    rng = np.random.default_rng(seed)
    tps = torch.from_numpy(rng.random((19, 2)).astype(np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = Path(tmp) / "sample.pt"
        _write_cephalometric_pt(pt_path, tps=tps)

        ds_full = CephalometricDataset(
            [str(pt_path)], augment=False, mode="coord"
        )
        ds_subset = CephalometricDataset(
            [str(pt_path)], augment=False, mode="coord",
            landmark_indices=subset,
        )
        _, coords_full, _ = ds_full[0]
        _, coords_subset, _ = ds_subset[0]

    assert coords_subset.shape == (len(subset), 2)
    torch.testing.assert_close(coords_subset, coords_full[subset])


class TestCephalometricDataset:
    """Shape/mode unit tests for CephalometricDataset."""

    def test_coord_mode_arity_and_shapes(self, cephalometric_pt_dir):
        """Coord mode returns a 3-tuple with correct shapes/dtypes/range."""
        paths = sorted([str(p) for p in cephalometric_pt_dir.glob("*.pt")])
        ds = CephalometricDataset(paths, augment=False, mode="coord")

        assert len(ds) == 5

        out = ds[0]
        assert len(out) == 3
        img, coords, metadata = out
        assert img.shape == (3, 512, 512)
        assert img.dtype == torch.float32
        assert coords.shape == (19, 2)
        assert coords.dtype == torch.float32
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0
        assert "orig_size" in metadata
        assert "pixel_spacing" in metadata
        assert "split" in metadata

    def test_heatmap_mode_arity_and_shapes(self, cephalometric_pt_dir):
        """Heatmap mode returns a 4-tuple with correct heatmap shape."""
        paths = sorted([str(p) for p in cephalometric_pt_dir.glob("*.pt")])
        heatmap_size = 128
        ds = CephalometricDataset(
            paths, augment=False, mode="heatmap", heatmap_size=heatmap_size
        )

        out = ds[0]
        assert len(out) == 4
        img, coords, heatmaps, metadata = out
        assert img.shape == (3, 512, 512)
        assert img.dtype == torch.float32
        assert coords.shape == (19, 2)
        assert coords.dtype == torch.float32
        assert heatmaps.shape == (19, heatmap_size, heatmap_size)
        assert heatmaps.dtype == torch.float32

    def test_pixel_spacing_uses_stored_value(self, tmp_path):
        """pixel_spacing metadata uses the stored .pt value when present."""
        pt_path = tmp_path / "with_spacing.pt"
        img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        data = {
            "image": img,
            "tps": torch.rand(19, 2).float(),
            "orig_size": torch.tensor([1935.0, 2400.0]),
            "pixel_spacing": torch.tensor(0.25, dtype=torch.float32),
            "split": "test1",
        }
        torch.save(data, str(pt_path))

        ds = CephalometricDataset([str(pt_path)], augment=False, mode="coord")
        _, _, metadata = ds[0]
        assert float(metadata["pixel_spacing"]) == pytest.approx(0.25)
        assert metadata["split"] == "test1"

    def test_pixel_spacing_defaults_when_absent(self, tmp_path):
        """pixel_spacing defaults to ISBI 0.1 when the .pt omits it."""
        pt_path = tmp_path / "no_spacing.pt"
        img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        data = {
            "image": img,
            "tps": torch.rand(19, 2).float(),
            "orig_size": torch.tensor([1935.0, 2400.0]),
            "split": "train",
        }
        torch.save(data, str(pt_path))

        ds = CephalometricDataset([str(pt_path)], augment=False, mode="coord")
        _, _, metadata = ds[0]
        assert float(metadata["pixel_spacing"]) == pytest.approx(0.1)


# Feature: cephalometric-dataset, Property 10: Mean-shape element-wise average
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_cephalometric_mean_shape_elementwise_average(n, seed):
    """Property 10: mean shape equals the element-wise mean across samples.

    Validates: Requirements 8.1, 8.2, 8.3
    """
    rng = np.random.default_rng(seed)
    tps_list = [
        torch.from_numpy(rng.random((19, 2)).astype(np.float32)) for _ in range(n)
    ]

    # Use TemporaryDirectory (not the tmp_path fixture) to avoid the Hypothesis
    # function-scoped-fixture health check.
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = Path(tmp) / "train"
        train_dir.mkdir()
        for i, tps in enumerate(tps_list):
            _write_cephalometric_pt(train_dir / f"ceph_{i}.pt", tps=tps)

        out_path = Path(tmp) / "mean_shapes" / "mean_shape_cephalometric.pt"
        mean_shape = compute_mean_shape(str(train_dir), str(out_path))

    assert mean_shape.shape == (19, 2)
    expected = torch.stack(tps_list, dim=0).mean(dim=0)
    torch.testing.assert_close(mean_shape, expected, atol=1e-5, rtol=1e-5)
