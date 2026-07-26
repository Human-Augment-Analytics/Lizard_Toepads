"""Unit tests for dataset adapters.

Tests use mock .pt files to verify return shapes without loading real data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

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
