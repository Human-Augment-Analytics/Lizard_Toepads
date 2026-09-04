"""Property-based tests for the hrnet_cascade variant.

Feature: hrnet_cascade. Each property runs a minimum of 100 examples. Model-
constructing properties use reduced settings (input 256, small cascade_width,
no pretrained weights) to stay fast.
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st

from landmarking.config.schema import LandmarkingConfig
from landmarking.models.hrnet_cascade import HRNetCascade
from landmarking.models.hrnet_heatmap import decode_coords
from landmarking.training.loss import cascade_heatmap_loss, heatmap_loss


def _distinct_stage_param_ids(model):
    """Count distinct RefineStage parameter sets (1 if shared, else K)."""
    if model.shared_weights:
        return 1
    return len(model.stages)


# --------------------------------------------------------------------------- #
# Property 1: Construction consistency
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    num_lms=st.integers(min_value=1, max_value=20),
    num_stages=st.integers(min_value=1, max_value=5),
    shared=st.booleans(),
    heatmap_size=st.sampled_from([64, 128]),
    cascade_width=st.sampled_from([64, 128]),
)
def test_p1_construction_consistency(num_lms, num_stages, shared, heatmap_size, cascade_width):
    m = HRNetCascade(
        num_landmarks=num_lms, num_stages=num_stages, shared_weights=shared,
        pretrained=False, heatmap_size=heatmap_size, cascade_width=cascade_width,
    )
    assert m.num_stages == num_stages
    expected_sets = 1 if shared else num_stages
    assert _distinct_stage_param_ids(m) == expected_sets


# --------------------------------------------------------------------------- #
# Property 2: Forward output shape invariant
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=3),
    num_lms=st.integers(min_value=1, max_value=12),
    num_stages=st.integers(min_value=1, max_value=4),
    s=st.sampled_from([256]),
)
def test_p2_forward_shapes(b, num_lms, num_stages, s):
    hs = 64
    m = HRNetCascade(num_landmarks=num_lms, num_stages=num_stages, pretrained=False,
                     heatmap_size=hs, cascade_width=64)
    m.eval()
    with torch.no_grad():
        stage_hms, coords = m(torch.randn(b, 3, s, s))
    assert len(stage_hms) == num_stages
    for hm in stage_hms:
        assert hm.shape == (b, num_lms, hs, hs)
    assert coords.shape == (b, num_lms, 2)


# --------------------------------------------------------------------------- #
# Property 3: Single-stage => no merge invoked
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    num_lms=st.integers(min_value=1, max_value=12),
    shared=st.booleans(),
)
def test_p3_single_stage_no_merge(num_lms, shared):
    m = HRNetCascade(num_landmarks=num_lms, num_stages=1, shared_weights=shared,
                     pretrained=False, heatmap_size=64, cascade_width=64)
    # No merge modules exist for a single-stage cascade.
    if shared:
        assert m.merge is None
    else:
        assert len(m.merges) == 0
    m.eval()
    with torch.no_grad():
        stage_hms, _ = m(torch.randn(1, 3, 256, 256))
    assert len(stage_hms) == 1


# --------------------------------------------------------------------------- #
# Property 4: Loss composition equals weighted mean of per-stage losses
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=3),
    n=st.integers(min_value=1, max_value=9),
    k=st.integers(min_value=1, max_value=4),
    hs=st.sampled_from([16, 32]),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p4_loss_composition(b, n, k, hs, seed):
    g = torch.Generator().manual_seed(seed)
    stage_hms = [torch.randn(b, n, hs, hs, generator=g) for _ in range(k)]
    gt = torch.rand(b, n, 2, generator=g)
    from landmarking.training.loss import _global_soft_argmax
    total, per_stage = cascade_heatmap_loss(stage_hms, None, gt, hs, mode="ce")
    manual = [
        heatmap_loss(hm, _global_soft_argmax(hm), gt, hs, coord_weight=1.0, mode="ce")
        for hm in stage_hms
    ]
    expected = sum(manual) / k
    assert total.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Property 5: Coordinates come from the final stage
# --------------------------------------------------------------------------- #

@settings(max_examples=50, deadline=None)
@given(
    num_lms=st.integers(min_value=1, max_value=9),
    num_stages=st.integers(min_value=1, max_value=3),
)
def test_p5_final_stage_coords(num_lms, num_stages):
    m = HRNetCascade(num_landmarks=num_lms, num_stages=num_stages, pretrained=False,
                     heatmap_size=64, cascade_width=64)
    m.eval()
    with torch.no_grad():
        stage_hms, coords = m(torch.randn(1, 3, 256, 256))
        expected = decode_coords(stage_hms[-1], mode="windowed", radius=5)
    assert torch.allclose(coords, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Property 6: Config round-trip
# --------------------------------------------------------------------------- #

@settings(max_examples=100, deadline=None)
@given(
    num_stages=st.integers(min_value=1, max_value=6),
    shared=st.booleans(),
    heatmap_size=st.sampled_from([64, 128]),
    cascade_width=st.sampled_from([128, 256]),
)
def test_p6_config_round_trip(num_stages, shared, heatmap_size, cascade_width):
    cfg = LandmarkingConfig.from_dict({
        "model": {"variant": "hrnet_cascade", "num_stages": num_stages,
                  "shared_weights": shared, "heatmap_size": heatmap_size,
                  "cascade_width": cascade_width},
    })
    cfg2 = LandmarkingConfig.from_dict(cfg.to_dict())
    assert cfg2.model.num_stages == num_stages
    assert cfg2.model.shared_weights == shared
    assert cfg2.model.heatmap_size == heatmap_size
    assert cfg2.model.cascade_width == cascade_width
