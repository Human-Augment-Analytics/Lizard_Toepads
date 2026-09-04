"""Unit + integration tests for the hrnet_cascade variant (tasks 2-4)."""

import numpy as np
import pytest
import torch

from landmarking.config.schema import LandmarkingConfig
from landmarking.models.hrnet_cascade import HRNetCascade, _RefineStage
from landmarking.models.hrnet_heatmap import decode_coords
from landmarking.models.registry import MODEL_REGISTRY, get_model
from landmarking.training.loss import cascade_heatmap_loss, heatmap_loss


# --------------------------------------------------------------------------- #
# Task 2: model
# --------------------------------------------------------------------------- #

def test_registered():
    import landmarking.models.hrnet_cascade  # noqa: F401
    assert "hrnet_cascade" in MODEL_REGISTRY


def test_forward_returns_k_heatmaps_and_coords():
    n, k, hs = 9, 3, 64
    m = HRNetCascade(num_landmarks=n, num_stages=k, pretrained=False,
                     heatmap_size=hs, cascade_width=64)
    m.eval()
    with torch.no_grad():
        stage_hms, coords = m(torch.randn(2, 3, 256, 256))
    assert isinstance(stage_hms, list)
    assert len(stage_hms) == k
    for hm in stage_hms:
        assert hm.shape == (2, n, hs, hs)
    assert coords.shape == (2, n, 2)


def test_num_stages_one_no_merge():
    m = HRNetCascade(num_landmarks=9, num_stages=1, pretrained=False,
                     heatmap_size=64, cascade_width=64, shared_weights=True)
    # No merge module when K == 1 (shared mode).
    assert m.merge is None
    m.eval()
    with torch.no_grad():
        stage_hms, coords = m(torch.randn(1, 3, 256, 256))
    assert len(stage_hms) == 1


def test_shared_vs_independent_param_count():
    kw = dict(num_landmarks=9, num_stages=3, pretrained=False,
              heatmap_size=64, cascade_width=64)
    shared = HRNetCascade(shared_weights=True, **kw)
    indep = HRNetCascade(shared_weights=False, **kw)

    def refine_params(model):
        # Count only the refinement (non-backbone) params.
        total = sum(p.numel() for p in model.parameters())
        backbone = sum(p.numel() for p in model.backbone.parameters())
        return total - backbone

    assert refine_params(indep) > refine_params(shared)


def test_coords_from_final_stage():
    m = HRNetCascade(num_landmarks=9, num_stages=3, pretrained=False,
                     heatmap_size=64, cascade_width=64)
    m.eval()
    with torch.no_grad():
        stage_hms, coords = m(torch.randn(2, 3, 256, 256))
        expected = decode_coords(stage_hms[-1], mode="windowed", radius=5)
    assert torch.allclose(coords, expected, atol=1e-6)


def test_default_construction():
    m = HRNetCascade(num_landmarks=9, pretrained=False)
    assert m.num_stages == 3
    assert m.shared_weights is True


def test_uses_timm_backbone_not_scratch_stem():
    """Must build the timm HRNet backbone, not the stacked-hourglass stem."""
    import landmarking.models.hrnet_cascade as mod
    src = mod.__file__
    text = open(src).read()
    assert "torch_geometric" not in "".join(
        ln for ln in text.splitlines() if ln.strip().startswith(("import ", "from "))
    )
    assert "hrnet_w18" in text  # builds timm HRNet


def test_bad_num_stages_rejected():
    with pytest.raises(ValueError):
        HRNetCascade(num_landmarks=9, num_stages=0, pretrained=False)


# --------------------------------------------------------------------------- #
# Task 3: loss
# --------------------------------------------------------------------------- #

def test_cascade_loss_equals_mean_of_per_stage():
    b, n, hs, k = 2, 9, 32, 3
    torch.manual_seed(0)
    stage_hms = [torch.randn(b, n, hs, hs) for _ in range(k)]
    gt = torch.rand(b, n, 2)

    # The cascade loss decodes its own coord term via global soft-argmax and
    # defaults coord_weight=1.0; the manual comparison must match that.
    from landmarking.training.loss import _global_soft_argmax
    total, per_stage = cascade_heatmap_loss(stage_hms, None, gt, hs, mode="ce")
    manual = [
        heatmap_loss(hm, _global_soft_argmax(hm), gt, hs, coord_weight=1.0, mode="ce")
        for hm in stage_hms
    ]
    expected = sum(manual) / k
    assert total.item() == pytest.approx(expected.item(), rel=1e-6)


def test_cascade_loss_single_stage_equals_plain_heatmap_loss():
    b, n, hs = 2, 9, 32
    torch.manual_seed(1)
    hm = torch.randn(b, n, hs, hs)
    gt = torch.rand(b, n, 2)
    from landmarking.training.loss import _global_soft_argmax
    total, _ = cascade_heatmap_loss([hm], None, gt, hs, mode="ce")
    plain = heatmap_loss(
        hm, _global_soft_argmax(hm), gt, hs, coord_weight=1.0, mode="ce"
    )
    assert total.item() == pytest.approx(plain.item(), rel=1e-6)


def test_cascade_loss_stage_weights_length_mismatch_raises():
    hm = torch.randn(1, 9, 32, 32)
    co = torch.rand(1, 9, 2)
    gt = torch.rand(1, 9, 2)
    with pytest.raises(ValueError):
        cascade_heatmap_loss([hm, hm], [co, co], gt, 32, stage_weights=[1.0])


# --------------------------------------------------------------------------- #
# Task 1: schema round-trip
# --------------------------------------------------------------------------- #

def test_config_round_trip():
    cfg = LandmarkingConfig.from_dict({
        "dataset": {"name": "lizard", "num_landmarks": 9, "input_size": 512},
        "model": {"variant": "hrnet_cascade", "num_stages": 3,
                  "shared_weights": True, "heatmap_size": 128, "cascade_width": 256},
    })
    cfg2 = LandmarkingConfig.from_dict(cfg.to_dict())
    assert cfg2.model.variant == "hrnet_cascade"
    assert cfg2.model.num_stages == 3
    assert cfg2.model.shared_weights is True
    assert cfg2.model.cascade_width == 256


def test_all_registry_variants_still_constructible():
    import landmarking.models  # noqa: F401
    for name in ("heatmap", "hrnet_coord", "stacked_hourglass", "vit", "pipnet",
                 "hrnet_cascade"):
        assert name in MODEL_REGISTRY


# --------------------------------------------------------------------------- #
# Task 4: engine integration
# --------------------------------------------------------------------------- #

class _SynthLizard(torch.utils.data.Dataset):
    def __init__(self, n_items, num_lms, input_size):
        g = torch.Generator().manual_seed(7)
        self.imgs = torch.randn(n_items, 3, input_size, input_size, generator=g)
        self.coords = torch.rand(n_items, num_lms, 2, generator=g)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        return self.imgs[i], self.coords[i], {"orig_size": torch.tensor([512.0, 512.0])}


def test_engine_train_and_validate_cascade(tmp_path):
    from pathlib import Path
    from torch.utils.data import DataLoader
    from landmarking.training.engine import TrainingEngine

    n, hs, s = 9, 32, 128
    cfg = LandmarkingConfig.from_dict({
        "paths": {"output_root": str(tmp_path / "runs")},
        "dataset": {"name": "lizard", "num_landmarks": n, "input_size": s,
                    "graph_topology": "chain"},
        "model": {"variant": "hrnet_cascade", "num_stages": 2,
                  "shared_weights": True, "heatmap_size": hs, "cascade_width": 64,
                  "sigma": 1.5},
        "training": {"epochs": 1, "batch_size": 2, "val_batch_size": 2,
                     "lr": 1e-4, "lr_backbone": 1e-4, "device": "cpu",
                     "heatmap_loss_mode": "ce"},
    })
    cfg.resolve_paths()

    engine = TrainingEngine(cfg)
    engine.output_dir = str(tmp_path / "out")
    Path(engine.output_dir).mkdir(parents=True, exist_ok=True)

    engine.model = get_model(
        "hrnet_cascade", num_landmarks=n, num_stages=2, shared_weights=True,
        pretrained=False, heatmap_size=hs, cascade_width=64,
    ).to(engine.device)

    # Flags the train/val branches check.
    engine._is_heatmap_model = False
    engine._is_graph_cond_heatmap = False
    engine._is_heatmap_on_coords = False
    engine._heatmap_use_star = False
    engine._is_coord_only_model = False
    engine._is_star_model = False
    engine._use_star_loss = False
    engine._is_graph_prior_fusion = False
    engine._is_pipnet = False
    engine._is_cascade = True
    engine.mean_shape = None
    engine.mean_shape_flipped = None
    engine.edge_index = None

    engine.optimizer = torch.optim.Adam(engine.model.parameters(), lr=1e-4)
    engine.scheduler = torch.optim.lr_scheduler.MultiStepLR(engine.optimizer, [1])

    ds = _SynthLizard(4, n, s)
    engine.train_loader = DataLoader(ds, batch_size=2)
    engine.val_loader = DataLoader(ds, batch_size=2)

    loss = engine._train_epoch(epoch=1)
    assert np.isfinite(loss)
    metrics = engine._validate(epoch=1)
    assert "val_loss" in metrics and np.isfinite(metrics["val_loss"])
    assert "val_px_err" in metrics


# --------------------------------------------------------------------------- #
# Task 6: experiment config
# --------------------------------------------------------------------------- #

def test_experiment_config_loads():
    from pathlib import Path
    cfg_path = (
        Path(__file__).resolve().parents[2]
        / "config" / "experiments" / "hrnet_cascade" / "lizard.json"
    )
    assert cfg_path.exists(), f"missing config: {cfg_path}"
    cfg = LandmarkingConfig.from_json(str(cfg_path))
    cfg.resolve_paths()
    cfg.validate()
    assert cfg.model.variant == "hrnet_cascade"
    assert cfg.model.num_stages == 3
    assert cfg.model.shared_weights is True
    assert cfg.dataset.num_landmarks == 9


# --------------------------------------------------------------------------- #
# Task 5: eval-script integration
# --------------------------------------------------------------------------- #

def test_eval_build_model_kwargs_cascade_roundtrip():
    """The eval script must build a cascade model that loads a cascade
    checkpoint (stage/merge params match)."""
    from landmarking.scripts.evaluate import build_model_kwargs

    n = 9
    cfg = LandmarkingConfig.from_dict({
        "dataset": {"name": "lizard", "num_landmarks": n, "input_size": 256},
        "model": {"variant": "hrnet_cascade", "num_stages": 3,
                  "shared_weights": False, "heatmap_size": 64, "cascade_width": 64},
    })
    kwargs = build_model_kwargs(cfg)
    assert kwargs["num_stages"] == 3
    assert kwargs["shared_weights"] is False

    trained = get_model("hrnet_cascade", num_landmarks=n, num_stages=3,
                        shared_weights=False, pretrained=False, heatmap_size=64,
                        cascade_width=64)
    state = trained.state_dict()
    eval_model = get_model("hrnet_cascade", **{**kwargs, "pretrained": False})
    missing, unexpected = eval_model.load_state_dict(state, strict=True)
    assert not missing and not unexpected


def test_eval_decode_dispatch_cascade_returns_coords():
    """The cascade eval dispatch (`_, pred = model(imgs)`) yields coords."""
    m = HRNetCascade(num_landmarks=9, num_stages=2, pretrained=False,
                     heatmap_size=64, cascade_width=64)
    m.eval()
    with torch.no_grad():
        _, pred = m(torch.randn(2, 3, 256, 256))
    assert pred.shape == (2, 9, 2)


# --------------------------------------------------------------------------- #
# Corner-trap fix: the cascade coord gradient must pull a corner prediction
# toward the true location (global soft-argmax, not windowed argmax trap).
# --------------------------------------------------------------------------- #

def test_cascade_coord_gradient_escapes_corner():
    """A heatmap whose argmax is at the (0,0) corner but whose GT is elsewhere
    must receive a gradient that INCREASES mass away from the corner."""
    from landmarking.training.loss import cascade_heatmap_loss

    b, n, hs = 1, 1, 16
    # Logits with the peak forced at the top-left corner cell (0,0).
    hm = torch.full((b, n, hs, hs), -5.0, requires_grad=True)
    with torch.no_grad():
        hm[0, 0, 0, 0] = 5.0  # argmax at corner
    gt = torch.tensor([[[0.8, 0.8]]])  # true location is bottom-right

    total, _ = cascade_heatmap_loss([hm], None, gt, hs, mode="ce", coord_weight=100.0)
    total.backward()
    grad = hm.grad[0, 0]
    # Descending the loss should push probability mass toward the GT (bottom-right)
    # cell relative to the corner: the negative gradient at a bottom-right cell
    # should exceed that at the corner (i.e. loss decreases by raising BR logits).
    br = grad[hs - 1, hs - 1].item()
    corner = grad[0, 0].item()
    # Gradient wrt logits: more-negative grad => raising that logit lowers loss.
    assert br < corner  # bottom-right is favored over the corner
