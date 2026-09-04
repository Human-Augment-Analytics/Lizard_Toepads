"""Tests for STAR loss on the HRNet heatmap variant (Option A)."""

import pytest
import torch

from landmarking.models.hrnet_heatmap import HRNetHeatmap
from landmarking.training.loss import heatmap_loss, heatmap_star_loss


def test_star_disabled_by_default():
    m = HRNetHeatmap(num_landmarks=9, pretrained=False, heatmap_size=32)
    assert m.sigma_head is None
    with pytest.raises(RuntimeError):
        m.forward_star(torch.randn(1, 3, 128, 128))


def test_forward_star_shapes():
    n, hs = 9, 32
    m = HRNetHeatmap(num_landmarks=n, pretrained=False, heatmap_size=hs, use_star=True)
    m.eval()
    with torch.no_grad():
        hm, coords, sigma = m.forward_star(torch.randn(2, 3, 128, 128))
    assert hm.shape == (2, n, hs, hs)
    assert coords.shape == (2, n, 2)
    assert sigma.shape == (2, 3 * n, hs, hs)


def test_star_weight_zero_recovers_plain_heatmap_loss():
    n, hs = 9, 32
    m = HRNetHeatmap(num_landmarks=n, pretrained=False, heatmap_size=hs, use_star=True)
    m.eval()
    imgs = torch.randn(2, 3, 128, 128)
    gt = torch.rand(2, n, 2)
    with torch.no_grad():
        hm, coords, sigma = m.forward_star(imgs)
        plain = heatmap_loss(hm, coords, gt, hs, mode="ce")
        star0 = heatmap_star_loss(
            hm, coords, gt, sigma, hs, mode="ce", star_weight=0.0,
        )[0]
    assert star0.item() == pytest.approx(plain.item(), rel=1e-6)


def test_star_gradients_flow_to_sigma_and_head():
    n, hs = 9, 32
    m = HRNetHeatmap(num_landmarks=n, pretrained=False, heatmap_size=hs, use_star=True)
    m.train()
    imgs = torch.randn(2, 3, 128, 128)
    gt = torch.rand(2, n, 2)
    hm, coords, sigma = m.forward_star(imgs)
    total = heatmap_star_loss(
        hm, coords, gt, sigma, hs, mode="ce", star_weight=1.0,
    )[0]
    total.backward()
    assert m.sigma_head.weight.grad is not None
    assert m.sigma_head.weight.grad.abs().sum() > 0
    # heatmap head also receives gradient (via the heatmap + coord terms)
    assert m.head[-1].weight.grad is not None
    assert m.head[-1].weight.grad.abs().sum() > 0


def test_star_zero_init_isotropic_finite():
    """Zero-init sigma head => finite STAR term at start (isotropic Sigma≈I)."""
    n, hs = 9, 32
    m = HRNetHeatmap(num_landmarks=n, pretrained=False, heatmap_size=hs, use_star=True)
    m.eval()
    imgs = torch.randn(2, 3, 128, 128)
    gt = torch.rand(2, n, 2)
    with torch.no_grad():
        hm, coords, sigma = m.forward_star(imgs)
        total, hm_coord, lstar = heatmap_star_loss(hm, coords, gt, sigma, hs, mode="ce")
    assert torch.isfinite(total)
    assert torch.isfinite(lstar)


def test_eval_build_model_kwargs_star_roundtrip():
    """The eval script must build a STAR heatmap model that loads a STAR
    checkpoint (sigma_head weights present on both sides)."""
    from landmarking.config.schema import LandmarkingConfig
    from landmarking.scripts.evaluate import build_model_kwargs
    from landmarking.models.registry import get_model

    n = 9
    cfg = LandmarkingConfig.from_dict({
        "dataset": {"name": "lizard", "num_landmarks": n, "input_size": 256},
        "model": {"variant": "heatmap", "heatmap_size": 64, "bn_momentum": 0.1,
                  "heatmap_use_star": True},
    })
    kwargs = build_model_kwargs(cfg)
    assert kwargs["use_star"] is True

    trained = get_model("heatmap", num_landmarks=n, pretrained=False,
                        heatmap_size=64, use_star=True)
    state = trained.state_dict()
    # kwargs already contains num_landmarks (build_model_kwargs adds it), exactly
    # as the eval script calls get_model(variant, **kwargs).
    eval_model = get_model("heatmap", **{**kwargs, "pretrained": False})
    missing, unexpected = eval_model.load_state_dict(state, strict=False)
    assert not any("sigma_head" in k for k in missing)
    assert not any("sigma_head" in k for k in unexpected)


def test_eval_decode_dispatch_heatmap_returns_coords():
    """The heatmap eval dispatch takes coords (element [1]), not heatmaps."""
    m = HRNetHeatmap(num_landmarks=9, pretrained=False, heatmap_size=64, use_star=True)
    m.eval()
    with torch.no_grad():
        _, pred = m(torch.randn(2, 3, 256, 256))  # mirrors eval `_, pred = model(imgs)`
    assert pred.shape == (2, 9, 2)  # coords, not heatmaps
    assert float(pred.min()) >= 0.0 - 1e-4
