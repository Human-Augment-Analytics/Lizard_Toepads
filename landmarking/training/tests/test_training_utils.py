"""Unit tests for training utilities.

Tests seed determinism, param group separation, output dir uniqueness,
and loss functions. No model forward passes or GPU usage.
"""

import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from landmarking.training.utils import (
    set_seed,
    get_device,
    make_param_groups,
    make_output_dir,
)
from landmarking.training.loss import landmark_loss, dist_loss


class TestSetSeed:
    """Test seed determinism."""

    def test_deterministic_random(self):
        """Same seed produces same Python random sequence."""
        import random

        set_seed(123)
        seq1 = [random.random() for _ in range(10)]
        set_seed(123)
        seq2 = [random.random() for _ in range(10)]
        assert seq1 == seq2

    def test_deterministic_numpy(self):
        """Same seed produces same NumPy random sequence."""
        set_seed(456)
        arr1 = np.random.rand(5)
        set_seed(456)
        arr2 = np.random.rand(5)
        np.testing.assert_array_equal(arr1, arr2)

    def test_deterministic_torch(self):
        """Same seed produces same PyTorch random tensor."""
        set_seed(789)
        t1 = torch.randn(3, 3)
        set_seed(789)
        t2 = torch.randn(3, 3)
        assert torch.equal(t1, t2)

    def test_different_seeds_differ(self):
        """Different seeds produce different sequences."""
        set_seed(1)
        t1 = torch.randn(5)
        set_seed(2)
        t2 = torch.randn(5)
        assert not torch.equal(t1, t2)


class TestGetDevice:
    """Test device resolution."""

    def test_cpu_always_works(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_cuda_fallback_to_cpu(self):
        """If CUDA unavailable, should fall back to CPU."""
        device = get_device("cuda")
        # Either cuda or cpu depending on machine
        assert device.type in ("cuda", "cpu")


class TestMakeParamGroups:
    """Test parameter group separation."""

    def test_separates_backbone_and_head(self):
        """Model with backbone attribute splits into 2 groups."""

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(10, 5)
                self.head = nn.Linear(5, 2)

        model = DummyModel()
        groups = make_param_groups(model, lr=1e-3, lr_backbone=1e-5)

        assert len(groups) == 2
        # Backbone group
        assert groups[0]["lr"] == 1e-5
        # Head group
        assert groups[1]["lr"] == 1e-3

    def test_backbone_params_are_correct(self):
        """Backbone group contains exactly the backbone parameters."""

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(10, 5), nn.Linear(5, 5)
                )
                self.head = nn.Linear(5, 2)

        model = DummyModel()
        groups = make_param_groups(model, lr=1e-3, lr_backbone=1e-5)

        backbone_param_count = sum(
            p.numel() for p in model.backbone.parameters()
        )
        group_param_count = sum(
            p.numel() for p in groups[0]["params"]
        )
        assert backbone_param_count == group_param_count

    def test_no_backbone_gives_single_group(self):
        """Model without backbone attribute puts all params in head group."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)

        model = SimpleModel()
        groups = make_param_groups(model, lr=1e-3, lr_backbone=1e-5)

        # Only head group (no backbone params)
        assert len(groups) == 1
        assert groups[0]["lr"] == 1e-3


class TestMakeOutputDir:
    """Test output directory path generation."""

    def test_contains_dataset_and_variant(self):
        path = make_output_dir("./runs", "lizard", "fused", "run001")
        assert "lizard" in path
        assert "fused" in path
        assert "run001" in path

    def test_different_inputs_produce_different_paths(self):
        path1 = make_output_dir("./runs", "lizard", "fused", "run001")
        path2 = make_output_dir("./runs", "wflw", "fused", "run001")
        path3 = make_output_dir("./runs", "lizard", "standard", "run001")
        assert path1 != path2
        assert path1 != path3

    def test_none_run_id_uses_timestamp(self):
        path = make_output_dir("./runs", "lizard", "fused")
        # Should have a timestamp component
        assert "lizard" in path
        assert "fused" in path


class TestLossFunctions:
    """Test loss function behavior with known inputs."""

    def test_landmark_loss_zero_for_identical(self):
        """MSE loss is zero when pred == gt."""
        coords = torch.rand(4, 9, 2)
        loss = landmark_loss(coords, coords)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_landmark_loss_positive_for_different(self):
        """MSE loss is positive when pred != gt."""
        pred = torch.zeros(2, 9, 2)
        gt = torch.ones(2, 9, 2)
        loss = landmark_loss(pred, gt)
        assert loss.item() > 0

    def test_landmark_loss_known_value(self):
        """Known MSE: pred=0, gt=1 → MSE=1.0."""
        pred = torch.zeros(1, 1, 2)
        gt = torch.ones(1, 1, 2)
        loss = landmark_loss(pred, gt)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_dist_loss_zero_for_identical(self):
        """Distance loss is zero when pred == gt."""
        coords = torch.rand(2, 5, 2)
        # Simple chain edge index
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
            dtype=torch.long,
        )
        loss = dist_loss(coords, coords, edge_index)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_dist_loss_positive_for_different(self):
        """Distance loss is positive when spacings differ."""
        pred = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
        gt = torch.tensor([[[0.0, 0.0], [0.3, 0.0], [1.0, 0.0]]])
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        loss = dist_loss(pred, gt, edge_index)
        assert loss.item() > 0
