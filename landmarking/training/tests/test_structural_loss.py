"""Tests for the pose-invariant structural loss (edge-length + turning-angle)."""

import math

import pytest
import torch

from landmarking.training.loss import (
    turning_angle_loss, structural_loss, chain_triples, dist_loss,
)
from landmarking.common.graph_topologies import make_chain_edge_index


def _rotate(coords, deg, center=0.5):
    """Rotate (B, N, 2) coords about `center` by `deg` degrees."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    R = torch.tensor([[c, -s], [s, c]], dtype=coords.dtype)
    x = coords - center
    return (x @ R.T) + center


def test_chain_triples_shape():
    tr = chain_triples(9)
    assert tr.shape == (7, 3)  # interior vertices 1..7
    assert tr[0].tolist() == [0, 1, 2]
    assert tr[-1].tolist() == [6, 7, 8]
    # N < 3 => empty
    assert chain_triples(2).shape == (0, 3)


def test_turning_angle_zero_when_identical():
    torch.manual_seed(0)
    coords = torch.rand(2, 9, 2)
    tr = chain_triples(9)
    assert turning_angle_loss(coords, coords, tr).item() == pytest.approx(0.0, abs=1e-7)


def test_turning_angle_is_rotation_invariant():
    """THE key property: a rotated version of the GT shape must incur ~0 angle
    loss (turning angles are pose-invariant), so a correctly-shaped but rotated
    toe-pad is NOT penalized."""
    torch.manual_seed(1)
    gt = torch.rand(1, 9, 2)
    tr = chain_triples(9)
    for deg in (10.0, 30.0, 90.0, 180.0):
        pred = _rotate(gt, deg)
        loss = turning_angle_loss(pred, gt, tr)
        assert loss.item() == pytest.approx(0.0, abs=1e-5), f"deg={deg}"


def test_turning_angle_penalizes_bending_change():
    """A shape that bends differently at a vertex incurs a positive angle loss."""
    tr = chain_triples(3)
    gt = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])  # straight line
    pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]])  # bent at vertex 1
    assert turning_angle_loss(pred, gt, tr).item() > 0.1


def test_dist_loss_is_rotation_invariant():
    """Edge-length loss is also pose-invariant (distances are rotation-invariant)."""
    torch.manual_seed(2)
    gt = torch.rand(1, 9, 2)
    ei = make_chain_edge_index(9)
    pred = _rotate(gt, 45.0)
    assert dist_loss(pred, gt, ei).item() == pytest.approx(0.0, abs=1e-5)


def test_structural_loss_zero_on_rotated_correct_shape():
    """The combined structural loss must be ~0 for a rotated correct shape —
    the whole point: it does not penalize pose, only shape deviation."""
    torch.manual_seed(3)
    gt = torch.rand(1, 9, 2)
    ei = make_chain_edge_index(9)
    tr = chain_triples(9)
    pred = _rotate(gt, 37.0)
    loss = structural_loss(pred, gt, ei, tr)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_structural_loss_positive_on_wrong_shape():
    torch.manual_seed(4)
    gt = torch.rand(1, 9, 2)
    pred = torch.rand(1, 9, 2)  # unrelated shape
    ei = make_chain_edge_index(9)
    tr = chain_triples(9)
    assert structural_loss(pred, gt, ei, tr).item() > 0.0


def test_turning_angle_empty_triples_is_zero():
    coords = torch.rand(1, 2, 2)
    empty = chain_triples(2)  # (0,3)
    assert turning_angle_loss(coords, coords, empty).item() == 0.0


def test_structural_loss_gradients_flow():
    gt = torch.rand(1, 9, 2)
    pred = torch.rand(1, 9, 2, requires_grad=True)
    ei = make_chain_edge_index(9)
    tr = chain_triples(9)
    structural_loss(pred, gt, ei, tr).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
