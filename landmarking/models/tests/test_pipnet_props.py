"""Property-based tests for the PIPNet variant.

Feature: pipnet. Each property runs a minimum of 100 examples. Model-constructing
properties use reduced settings (resnet18, small input, no pretrained weights)
to stay fast.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from landmarking.config.schema import LandmarkingConfig
from landmarking.models.pipnet import PIPNet, decode_pip, get_meanface_indices
from landmarking.training.loss import pipnet_loss


# --------------------------------------------------------------------------- #
# Property 1: Construction consistency across hyperparameters
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    num_lms=st.integers(min_value=2, max_value=20),
    net_stride=st.sampled_from([32, 64]),
    num_nb=st.integers(min_value=1, max_value=8),
    input_size=st.sampled_from([128, 256]),
)
def test_p1_construction_consistency(num_lms, net_stride, num_nb, input_size):
    num_nb = min(num_nb, num_lms - 1)
    model = PIPNet(
        num_landmarks=num_lms, backbone="resnet18", pretrained=False,
        input_size=input_size, net_stride=net_stride, num_nb=num_nb,
    )
    assert model.cls_layer.out_channels == num_lms
    assert model.x_layer.out_channels == num_lms
    assert model.y_layer.out_channels == num_lms
    assert model.nb_x_layer.out_channels == num_nb * num_lms
    assert model.nb_y_layer.out_channels == num_nb * num_lms
    assert model.grid_h == input_size // net_stride
    assert model.grid_w == input_size // net_stride


# --------------------------------------------------------------------------- #
# Property 2: Forward output shape invariant
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=3),
    num_lms=st.integers(min_value=2, max_value=12),
    num_nb=st.integers(min_value=1, max_value=6),
    input_size=st.sampled_from([128, 256]),
)
def test_p2_forward_shape_invariant(b, num_lms, num_nb, input_size):
    num_nb = min(num_nb, num_lms - 1)
    stride = 32
    grid = input_size // stride
    model = PIPNet(
        num_landmarks=num_lms, backbone="resnet18", pretrained=False,
        input_size=input_size, net_stride=stride, num_nb=num_nb,
    )
    model.eval()
    with torch.no_grad():
        cls, ox, oy, nbx, nby = model(torch.randn(b, 3, input_size, input_size))
    assert cls.shape == (b, num_lms, grid, grid)
    assert ox.shape == (b, num_lms, grid, grid)
    assert oy.shape == (b, num_lms, grid, grid)
    assert nbx.shape == (b, num_nb * num_lms, grid, grid)
    assert nby.shape == (b, num_nb * num_lms, grid, grid)


# --------------------------------------------------------------------------- #
# Property 3: Decode matches the reference formula
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=3),
    n=st.integers(min_value=1, max_value=8),
    grid=st.sampled_from([4, 8, 16]),
    data=st.data(),
)
def test_p3_decode_matches_formula(b, n, grid, data):
    input_size, net_stride = grid * 32, 32
    cls = torch.full((b, n, grid, grid), -5.0)
    off_x = torch.zeros(b, n, grid, grid)
    off_y = torch.zeros(b, n, grid, grid)
    exp = torch.zeros(b, n, 2)
    for bi in range(b):
        for i in range(n):
            r = data.draw(st.integers(0, grid - 1))
            c = data.draw(st.integers(0, grid - 1))
            ox = data.draw(st.floats(0.0, 0.99))
            oy = data.draw(st.floats(0.0, 0.99))
            cls[bi, i, r, c] = 5.0
            off_x[bi, i, r, c] = ox
            off_y[bi, i, r, c] = oy
            exp[bi, i, 0] = (c + ox) / grid
            exp[bi, i, 1] = (r + oy) / grid
    coords = decode_pip(cls, off_x, off_y, input_size, net_stride)
    assert torch.allclose(coords, exp, atol=1e-5)


# --------------------------------------------------------------------------- #
# Property 4: Meanface index validity and reference equivalence
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=40),
    num_nb=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p4_meanface_validity(n, num_nb, seed):
    num_nb = min(num_nb, n - 1)
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    # ensure distinct points to avoid ambiguous ties
    pts = pts + np.arange(n).reshape(-1, 1) * 1e-3
    idx, _, _, _ = get_meanface_indices(torch.tensor(pts), num_nb)
    assert idx.shape == (n, num_nb)
    for i in range(n):
        row = idx[i].tolist()
        assert len(set(row)) == num_nb  # distinct
        assert i not in row  # excludes self
        assert all(0 <= v < n for v in row)
        # ranked ascending by squared distance
        d = np.sum((pts[i] - pts) ** 2, axis=1)
        expected = np.argsort(d)[1:1 + num_nb]
        assert row == expected.tolist()


# --------------------------------------------------------------------------- #
# Property 5: Loss targets = gen_target_pip (matched preds → 0)
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=3),
    n=st.integers(min_value=2, max_value=8),
    num_nb=st.integers(min_value=1, max_value=4),
    grid=st.sampled_from([4, 8]),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p5_loss_zero_on_matched_targets(b, n, num_nb, grid, seed):
    num_nb = min(num_nb, n - 1)
    g = torch.Generator().manual_seed(seed)
    cols = torch.randint(0, grid, (b, n), generator=g)
    rows = torch.randint(0, grid, (b, n), generator=g)
    fx = torch.rand(b, n, generator=g) * 0.8 + 0.1
    fy = torch.rand(b, n, generator=g) * 0.8 + 0.1
    coords = torch.stack([(cols + fx) / grid, (rows + fy) / grid], dim=-1)

    mf = torch.stack([torch.arange(n).float(), torch.zeros(n)], dim=-1)
    idx, _, _, _ = get_meanface_indices(mf, num_nb)

    cls = torch.zeros(b, n, grid, grid)
    off_x = torch.zeros(b, n, grid, grid)
    off_y = torch.zeros(b, n, grid, grid)
    nb_x = torch.zeros(b, num_nb * n, grid, grid)
    nb_y = torch.zeros(b, num_nb * n, grid, grid)
    for bi in range(b):
        for i in range(n):
            r, c = rows[bi, i].item(), cols[bi, i].item()
            cls[bi, i, r, c] = 1.0
            off_x[bi, i, r, c] = fx[bi, i]
            off_y[bi, i, r, c] = fy[bi, i]
            for j in range(num_nb):
                nj = idx[i, j].item()
                nb_x[bi, num_nb * i + j, r, c] = coords[bi, nj, 0] * grid - c
                nb_y[bi, num_nb * i + j, r, c] = coords[bi, nj, 1] * grid - r

    total, lm, lx, ly, lnx, lny = pipnet_loss(cls, off_x, off_y, nb_x, nb_y, coords, idx)
    for term in (lm, lx, ly, lnx, lny):
        assert term.item() == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Property 6: Loss composition
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=3),
    n=st.integers(min_value=2, max_value=8),
    num_nb=st.integers(min_value=1, max_value=4),
    grid=st.sampled_from([4, 8]),
    cls_w=st.floats(0.5, 20.0),
    reg_w=st.floats(0.1, 5.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p6_loss_composition(b, n, num_nb, grid, cls_w, reg_w, seed):
    num_nb = min(num_nb, n - 1)
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(b, n, 2, generator=g)
    mf = torch.stack([torch.arange(n).float(), torch.zeros(n)], dim=-1)
    idx, _, _, _ = get_meanface_indices(mf, num_nb)
    cls = torch.randn(b, n, grid, grid, generator=g)
    off_x = torch.randn(b, n, grid, grid, generator=g)
    off_y = torch.randn(b, n, grid, grid, generator=g)
    nb_x = torch.randn(b, num_nb * n, grid, grid, generator=g)
    nb_y = torch.randn(b, num_nb * n, grid, grid, generator=g)
    total, lm, lx, ly, lnx, lny = pipnet_loss(
        cls, off_x, off_y, nb_x, nb_y, coords, idx,
        cls_loss_weight=cls_w, reg_loss_weight=reg_w,
    )
    expected = cls_w * lm + reg_w * (lx + ly + lnx + lny)
    assert total.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Property 7: Valid configuration round-trip
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    backbone=st.sampled_from(["resnet18", "resnet34", "resnet50"]),
    net_stride=st.sampled_from([16, 32, 64, 128]),
    num_nb=st.integers(min_value=1, max_value=30),
    cls_w=st.floats(0.1, 50.0),
    reg_w=st.floats(0.1, 10.0),
)
def test_p7_config_round_trip(backbone, net_stride, num_nb, cls_w, reg_w):
    cfg = LandmarkingConfig.from_dict({
        "model": {"variant": "pipnet", "backbone": backbone,
                  "net_stride": net_stride, "num_nb": num_nb},
        "training": {"pipnet_cls_loss_weight": cls_w,
                     "pipnet_reg_loss_weight": reg_w},
    })
    cfg2 = LandmarkingConfig.from_dict(cfg.to_dict())
    assert cfg2.model.backbone == backbone
    assert cfg2.model.net_stride == net_stride
    assert cfg2.model.num_nb == num_nb
    assert cfg2.training.pipnet_cls_loss_weight == cls_w
    assert cfg2.training.pipnet_reg_loss_weight == reg_w
