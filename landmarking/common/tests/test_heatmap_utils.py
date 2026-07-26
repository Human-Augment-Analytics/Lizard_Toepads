"""Unit tests for heatmap utilities."""

import numpy as np
import torch
import pytest

from landmarking.common.heatmap_utils import (
    generate_gaussian_heatmap,
    soft_argmax,
    hard_argmax,
)


class TestGenerateGaussianHeatmap:
    def test_output_shape(self):
        coords = np.array([[0.5, 0.5], [0.2, 0.8]])
        hm = generate_gaussian_heatmap(coords, heatmap_size=64, sigma=2.0)
        assert hm.shape == (2, 64, 64)

    def test_values_in_range(self):
        coords = np.array([[0.5, 0.5]])
        hm = generate_gaussian_heatmap(coords, heatmap_size=32, sigma=3.0)
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0

    def test_peak_near_target(self):
        coords = np.array([[0.5, 0.5]])
        hm = generate_gaussian_heatmap(coords, heatmap_size=64, sigma=2.0)
        peak_idx = np.unravel_index(hm[0].argmax(), hm[0].shape)
        # Peak should be within 1 pixel of target (0.5*63 ≈ 31.5)
        expected_y = 0.5 * 63
        expected_x = 0.5 * 63
        assert abs(peak_idx[0] - expected_y) <= 1
        assert abs(peak_idx[1] - expected_x) <= 1

    def test_center_peak_is_max(self):
        coords = np.array([[0.3, 0.7]])
        hm = generate_gaussian_heatmap(coords, heatmap_size=64, sigma=2.0)
        # The max value should be approximately 1.0
        assert hm[0].max() > 0.9


class TestSoftArgmax:
    def test_output_shape(self):
        heatmaps = torch.randn(2, 5, 32, 32)
        coords = soft_argmax(heatmaps)
        assert coords.shape == (2, 5, 2)

    def test_output_in_unit_range(self):
        heatmaps = torch.randn(1, 3, 16, 16)
        coords = soft_argmax(heatmaps)
        assert (coords >= 0).all()
        assert (coords <= 1).all()

    def test_single_peak_recovery(self):
        # Create a heatmap with a strong peak at center
        heatmaps = torch.zeros(1, 1, 32, 32)
        heatmaps[0, 0, 16, 16] = 100.0  # Strong peak at center
        coords = soft_argmax(heatmaps)
        # Should be close to (0.5, 0.5)
        assert abs(coords[0, 0, 0].item() - 16 / 31) < 0.05
        assert abs(coords[0, 0, 1].item() - 16 / 31) < 0.05


class TestHardArgmax:
    def test_output_shape(self):
        heatmaps = torch.randn(2, 5, 32, 32)
        coords = hard_argmax(heatmaps)
        assert coords.shape == (2, 5, 2)

    def test_output_in_unit_range(self):
        heatmaps = torch.randn(1, 3, 16, 16)
        coords = hard_argmax(heatmaps)
        assert (coords >= 0).all()
        assert (coords <= 1).all()


class TestConstants:
    def test_imagenet_mean(self):
        from landmarking.common.constants import IMAGENET_MEAN
        assert len(IMAGENET_MEAN) == 3
        assert all(0 < v < 1 for v in IMAGENET_MEAN)

    def test_imagenet_std(self):
        from landmarking.common.constants import IMAGENET_STD
        assert len(IMAGENET_STD) == 3
        assert all(0 < v < 1 for v in IMAGENET_STD)

    def test_classmap(self):
        from landmarking.common.constants import CLASSMAP, TRAIN_CLASSES
        assert CLASSMAP[0] == "finger"
        assert CLASSMAP[1] == "toe"
        assert TRAIN_CLASSES == [0, 1]
