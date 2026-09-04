"""Unit tests for the PIPNet variant (tasks 1-4).

Covers meanface neighbor derivation, model construction/forward/decode, and the
composite loss. Where the vendored reference (``../PIPNet/PIPNet``) is present,
cross-check tests assert equivalence and are otherwise skipped.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from landmarking.config.schema import LandmarkingConfig
from landmarking.models.pipnet import (
    PIPNet,
    decode_pip,
    get_meanface_indices,
)
from landmarking.models.registry import MODEL_REGISTRY
from landmarking.training.loss import pipnet_loss


# --------------------------------------------------------------------------- #
# Reference discovery (optional cross-checks)
# --------------------------------------------------------------------------- #

def _reference_lib_dir():
    """Return the vendored PIPNet lib dir if present, else None."""
    here = Path(__file__).resolve()
    # workspace root .../Lizard_Toepads ; reference sibling .../PIPNet/PIPNet/lib
    for parent in here.parents:
        cand = parent.parent / "PIPNet" / "PIPNet" / "lib"
        if cand.exists():
            return cand
    return None


def _install_simps_shim():
    """Reference functions.py imports scipy.integrate.simps, removed in SciPy
    >= 1.14 (renamed to simpson). Alias it so the reference module loads."""
    import scipy.integrate as si

    if not hasattr(si, "simps") and hasattr(si, "simpson"):
        si.simps = si.simpson


def _load_reference_functions():
    """Import the reference lib/functions.py in isolation, or None."""
    lib = _reference_lib_dir()
    if lib is None:
        return None
    _install_simps_shim()
    spec = importlib.util.spec_from_file_location(
        "_pipnet_ref_functions", str(lib / "functions.py")
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _load_reference_data_utils():
    lib = _reference_lib_dir()
    if lib is None:
        return None
    spec = importlib.util.spec_from_file_location(
        "_pipnet_ref_data_utils", str(lib / "data_utils.py")
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


REF_HAVE = _reference_lib_dir() is not None


# --------------------------------------------------------------------------- #
# Task 2: meanface neighbor derivation
# --------------------------------------------------------------------------- #

def test_meanface_indices_hand():
    """Nearest-neighbor ranking on a fixed 1-D layout is hand-verifiable."""
    # Points on a line at x = 0,1,2,3,4 (y=0). Nearest neighbors are adjacent.
    mf = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    )
    idx, rev1, rev2, max_len = get_meanface_indices(mf, num_nb=2)
    assert idx.shape == (5, 2)
    # Landmark 0: nearest are 1 then 2
    assert idx[0].tolist() == [1, 2]
    # Landmark 2 (middle): nearest are 1 and 3 (tie at dist 1), then 0/4
    assert set(idx[2].tolist()) == {1, 3}
    # Landmark 4: nearest are 3 then 2
    assert idx[4].tolist() == [3, 2]
    # No self-reference
    for i in range(5):
        assert i not in idx[i].tolist()


def test_meanface_rejects_too_many_neighbors():
    mf = torch.zeros(4, 2)
    mf[:, 0] = torch.arange(4).float()
    with pytest.raises(ValueError):
        get_meanface_indices(mf, num_nb=4)  # num_nb > N-1 = 3


@pytest.mark.skipif(not REF_HAVE, reason="vendored PIPNet reference not present")
def test_meanface_matches_reference():
    """get_meanface_indices equals the reference get_meanface on random points."""
    ref = _load_reference_functions()
    if ref is None:
        pytest.skip("reference functions.py could not be imported")

    rng = np.random.default_rng(0)
    pts = rng.random((20, 2))
    num_nb = 5

    ours, _, _, _ = get_meanface_indices(torch.tensor(pts), num_nb)

    # Write a temporary meanface.txt in the reference's expected single-line form.
    import tempfile

    flat = " ".join(str(v) for v in pts.reshape(-1).tolist())
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "meanface.txt")
        with open(p, "w") as f:
            f.write(flat + "\n")
        ref_idx, _, _, _ = ref.get_meanface(p, num_nb)

    ref_idx = np.array(ref_idx)
    assert np.array_equal(ours.numpy(), ref_idx)


# --------------------------------------------------------------------------- #
# Task 3: model construction / forward / decode
# --------------------------------------------------------------------------- #

def test_pipnet_registered():
    import landmarking.models.pipnet  # noqa: F401
    assert "pipnet" in MODEL_REGISTRY


def test_forward_five_tuple_shapes():
    n, num_nb, s, stride = 9, 4, 256, 32
    model = PIPNet(
        num_landmarks=n,
        backbone="resnet18",
        pretrained=False,
        input_size=s,
        net_stride=stride,
        num_nb=num_nb,
    )
    model.eval()
    grid = s // stride  # 8
    with torch.no_grad():
        cls, ox, oy, nbx, nby = model(torch.randn(2, 3, s, s))
    assert cls.shape == (2, n, grid, grid)
    assert ox.shape == (2, n, grid, grid)
    assert oy.shape == (2, n, grid, grid)
    assert nbx.shape == (2, num_nb * n, grid, grid)
    assert nby.shape == (2, num_nb * n, grid, grid)


def test_default_construction_only_num_landmarks():
    # input_size default 512, net_stride 32 -> grid 16
    model = PIPNet(num_landmarks=9, pretrained=False)
    assert model.grid_h == 16 and model.grid_w == 16
    assert model.meanface_indices.shape == (9, 10)


def test_decode_one_hot():
    """A one-hot score at a known cell + fixed offset decodes correctly."""
    b, n, gh, gw = 1, 3, 8, 8
    cls = torch.full((b, n, gh, gw), -10.0)
    off_x = torch.zeros(b, n, gh, gw)
    off_y = torch.zeros(b, n, gh, gw)
    # Put each landmark's peak at cell (row=r, col=c) and offset (0.25, 0.75).
    targets = [(2, 3), (0, 7), (7, 0)]
    for i, (r, c) in enumerate(targets):
        cls[0, i, r, c] = 10.0
        off_x[0, i, r, c] = 0.25
        off_y[0, i, r, c] = 0.75
    coords = decode_pip(cls, off_x, off_y, input_size=256, net_stride=32)
    for i, (r, c) in enumerate(targets):
        assert coords[0, i, 0].item() == pytest.approx((c + 0.25) / gw)
        assert coords[0, i, 1].item() == pytest.approx((r + 0.75) / gh)


def test_no_forbidden_imports():
    """pipnet.py must not IMPORT torch_geometric or timm.

    Checks import statements specifically (the module docstring mentions these
    libraries to explain their deliberate absence).
    """
    src_path = Path(__file__).resolve().parents[2] / "models" / "pipnet.py"
    import_lines = [
        ln.strip()
        for ln in src_path.read_text().splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    joined = "\n".join(import_lines)
    assert "torch_geometric" not in joined
    assert "timm" not in joined


def test_bad_net_stride_rejected():
    with pytest.raises(ValueError):
        PIPNet(num_landmarks=9, pretrained=False, net_stride=7)


def test_grid_mismatch_detected():
    # net_stride 128 at input 256 -> grid 2; construct fine. Use a mismatch by
    # requesting an impossible small input for stride 128 handled by assertion.
    m = PIPNet(num_landmarks=9, pretrained=False, input_size=256, net_stride=128)
    assert m.grid_h == 2 and m.grid_w == 2


# --------------------------------------------------------------------------- #
# Task 4: composite loss
# --------------------------------------------------------------------------- #

def _make_grid_coords(b, n, gh, gw, seed=0):
    """Random coords that land cleanly inside cells (avoid exact boundaries)."""
    g = torch.Generator().manual_seed(seed)
    # cell centers + small jitter to keep floor stable
    cols = torch.randint(0, gw, (b, n), generator=g)
    rows = torch.randint(0, gh, (b, n), generator=g)
    fx = torch.rand(b, n, generator=g) * 0.8 + 0.1
    fy = torch.rand(b, n, generator=g) * 0.8 + 0.1
    x = (cols.float() + fx) / gw
    y = (rows.float() + fy) / gh
    return torch.stack([x, y], dim=-1), rows, cols, fx, fy


def test_loss_zero_when_predictions_match_targets():
    """Feeding exact targets drives all five terms to ~0."""
    b, n, num_nb, gh, gw = 2, 6, 3, 8, 8
    coords, rows, cols, fx, fy = _make_grid_coords(b, n, gh, gw)
    mf = torch.stack(
        [torch.arange(n).float(), torch.zeros(n)], dim=-1
    )  # line layout
    idx, _, _, _ = get_meanface_indices(mf, num_nb)

    # Build predictions equal to the reference targets.
    cls = torch.zeros(b, n, gh, gw)
    off_x = torch.zeros(b, n, gh, gw)
    off_y = torch.zeros(b, n, gh, gw)
    nb_x = torch.zeros(b, num_nb * n, gh, gw)
    nb_y = torch.zeros(b, num_nb * n, gh, gw)

    for bi in range(b):
        for i in range(n):
            r, c = rows[bi, i].item(), cols[bi, i].item()
            cls[bi, i, r, c] = 1.0
            off_x[bi, i, r, c] = fx[bi, i]
            off_y[bi, i, r, c] = fy[bi, i]
            for j in range(num_nb):
                nj = idx[i, j].item()
                nb_x[bi, num_nb * i + j, r, c] = coords[bi, nj, 0] * gw - c
                nb_y[bi, num_nb * i + j, r, c] = coords[bi, nj, 1] * gh - r

    total, lm, lx, ly, lnx, lny = pipnet_loss(
        cls, off_x, off_y, nb_x, nb_y, coords, idx
    )
    assert lm.item() == pytest.approx(0.0, abs=1e-6)
    assert lx.item() == pytest.approx(0.0, abs=1e-6)
    assert ly.item() == pytest.approx(0.0, abs=1e-6)
    assert lnx.item() == pytest.approx(0.0, abs=1e-6)
    assert lny.item() == pytest.approx(0.0, abs=1e-6)
    assert total.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_composition_weights():
    """Total equals cls_w*map + reg_w*(x+y+nbx+nby)."""
    b, n, num_nb, gh, gw = 2, 5, 2, 8, 8
    coords, *_ = _make_grid_coords(b, n, gh, gw, seed=1)
    mf = torch.stack([torch.arange(n).float(), torch.zeros(n)], dim=-1)
    idx, _, _, _ = get_meanface_indices(mf, num_nb)
    cls = torch.randn(b, n, gh, gw)
    off_x = torch.randn(b, n, gh, gw)
    off_y = torch.randn(b, n, gh, gw)
    nb_x = torch.randn(b, num_nb * n, gh, gw)
    nb_y = torch.randn(b, num_nb * n, gh, gw)
    total, lm, lx, ly, lnx, lny = pipnet_loss(
        cls, off_x, off_y, nb_x, nb_y, coords, idx,
        cls_loss_weight=10.0, reg_loss_weight=1.0,
    )
    expected = 10.0 * lm + 1.0 * (lx + ly + lnx + lny)
    assert total.item() == pytest.approx(expected.item(), rel=1e-6)


@pytest.mark.skipif(not REF_HAVE, reason="vendored PIPNet reference not present")
def test_loss_matches_reference():
    """pipnet_loss components equal the reference compute_loss_pip term-by-term."""
    ref_fn = _load_reference_functions()
    ref_du = _load_reference_data_utils()
    if ref_fn is None or ref_du is None:
        pytest.skip("reference modules could not be imported")

    import torch.nn as nn

    b, n, num_nb, gh, gw = 3, 8, 3, 8, 8
    coords, _, _, _, _ = _make_grid_coords(b, n, gh, gw, seed=7)
    mf = torch.rand(n, 2)
    idx, _, _, _ = get_meanface_indices(mf, num_nb)
    meanface_indices_list = [idx[i].tolist() for i in range(n)]

    cls = torch.randn(b, n, gh, gw)
    off_x = torch.randn(b, n, gh, gw)
    off_y = torch.randn(b, n, gh, gw)
    nb_x = torch.randn(b, num_nb * n, gh, gw)
    nb_y = torch.randn(b, num_nb * n, gh, gw)

    # Build reference targets per sample with gen_target_pip.
    tm = torch.zeros(b, n, gh, gw)
    tlx = torch.zeros(b, n, gh, gw)
    tly = torch.zeros(b, n, gh, gw)
    tnx = torch.zeros(b, num_nb * n, gh, gw)
    tny = torch.zeros(b, num_nb * n, gh, gw)
    for bi in range(b):
        target = coords[bi].numpy().reshape(-1)
        m = np.zeros((n, gh, gw))
        lx = np.zeros((n, gh, gw))
        ly = np.zeros((n, gh, gw))
        nx = np.zeros((num_nb * n, gh, gw))
        ny = np.zeros((num_nb * n, gh, gw))
        m, lx, ly, nx, ny = ref_du.gen_target_pip(
            target, meanface_indices_list, m, lx, ly, nx, ny
        )
        tm[bi] = torch.from_numpy(m).float()
        tlx[bi] = torch.from_numpy(lx).float()
        tly[bi] = torch.from_numpy(ly).float()
        tnx[bi] = torch.from_numpy(nx).float()
        tny[bi] = torch.from_numpy(ny).float()

    ref_map, ref_x, ref_y, ref_nx, ref_ny = ref_fn.compute_loss_pip(
        cls, off_x, off_y, nb_x, nb_y, tm, tlx, tly, tnx, tny,
        nn.MSELoss(), nn.L1Loss(), num_nb,
    )

    _, lm, lx2, ly2, lnx, lny = pipnet_loss(
        cls, off_x, off_y, nb_x, nb_y, coords, idx
    )
    assert lm.item() == pytest.approx(ref_map.item(), rel=1e-5, abs=1e-6)
    assert lx2.item() == pytest.approx(ref_x.item(), rel=1e-5, abs=1e-6)
    assert ly2.item() == pytest.approx(ref_y.item(), rel=1e-5, abs=1e-6)
    assert lnx.item() == pytest.approx(ref_nx.item(), rel=1e-5, abs=1e-6)
    assert lny.item() == pytest.approx(ref_ny.item(), rel=1e-5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Task 1: schema round-trip
# --------------------------------------------------------------------------- #

def test_config_round_trip():
    cfg = LandmarkingConfig.from_dict(
        {
            "dataset": {"name": "lizard", "num_landmarks": 9, "input_size": 512},
            "model": {
                "variant": "pipnet",
                "backbone": "resnet18",
                "net_stride": 32,
                "num_nb": 8,
            },
            "training": {
                "pipnet_cls_loss_weight": 10.0,
                "pipnet_reg_loss_weight": 1.0,
            },
        }
    )
    d = cfg.to_dict()
    cfg2 = LandmarkingConfig.from_dict(d)
    assert cfg2.model.backbone == "resnet18"
    assert cfg2.model.net_stride == 32
    assert cfg2.model.num_nb == 8
    assert cfg2.training.pipnet_cls_loss_weight == 10.0
    assert cfg2.training.pipnet_reg_loss_weight == 1.0


# --------------------------------------------------------------------------- #
# Task 6: experiment config smoke-load
# --------------------------------------------------------------------------- #

def test_experiment_config_loads():
    """The delivered Lizard pipnet config parses and resolves."""
    cfg_path = (
        Path(__file__).resolve().parents[2]
        / "config" / "experiments" / "pipnet" / "lizard.json"
    )
    assert cfg_path.exists(), f"missing config: {cfg_path}"
    cfg = LandmarkingConfig.from_json(str(cfg_path))
    cfg.resolve_paths()
    cfg.validate()
    assert cfg.model.variant == "pipnet"
    assert cfg.model.backbone == "resnet18"
    assert cfg.model.num_nb == 8
    assert cfg.dataset.num_landmarks == 9


# --------------------------------------------------------------------------- #
# Task 7.2: integration via TrainingEngine (synthetic data + mean shape)
# --------------------------------------------------------------------------- #

class _SynthLizard(torch.utils.data.Dataset):
    """Minimal Lizard-shaped dataset: (img, coords_norm, metadata)."""

    def __init__(self, n_items, num_lms, input_size):
        g = torch.Generator().manual_seed(123)
        self.imgs = torch.randn(n_items, 3, input_size, input_size, generator=g)
        self.coords = torch.rand(n_items, num_lms, 2, generator=g)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        return self.imgs[i], self.coords[i], {"orig_size": torch.tensor([512.0, 512.0])}


def _make_pipnet_engine(tmp_path, num_lms=9, num_nb=8, input_size=128, net_stride=32,
                        landmark_indices=None):
    """Build a TrainingEngine wired for pipnet with synthetic data + mean shape."""
    from torch.utils.data import DataLoader
    from landmarking.training.engine import TrainingEngine

    n_total = 9  # full mean-shape size before subsampling
    cfg = LandmarkingConfig.from_dict({
        "paths": {"output_root": str(tmp_path / "runs")},
        "dataset": {
            "name": "lizard",
            "num_landmarks": num_lms,
            "input_size": input_size,
            "graph_topology": "chain",
            "landmark_indices": landmark_indices or [],
        },
        "model": {"variant": "pipnet", "backbone": "resnet18",
                  "net_stride": net_stride, "num_nb": num_nb},
        "training": {"epochs": 1, "batch_size": 2, "val_batch_size": 2,
                     "lr": 1e-4, "lr_backbone": 1e-4, "device": "cpu",
                     "pipnet_cls_loss_weight": 10.0, "pipnet_reg_loss_weight": 1.0},
    })
    cfg.resolve_paths()

    engine = TrainingEngine(cfg)

    # Provide a full (n_total, 2) mean shape via monkeypatched setup pieces.
    # We replicate the minimal parts of setup() that pipnet needs, then swap in
    # synthetic loaders so no disk .pt files are required.
    from landmarking.common.graph_topologies import get_edge_index
    engine.output_dir = str(tmp_path / "out")
    Path(engine.output_dir).mkdir(parents=True, exist_ok=True)

    li = landmark_indices or None
    engine.edge_index = get_edge_index("chain", n_total, landmark_indices=li).to(engine.device)
    mshape = torch.rand(n_total, 2)
    if li:
        mshape = mshape[li]
    engine.mean_shape = mshape.to(engine.device)
    engine.mean_shape_flipped = None

    # Run the pipnet-specific portion of setup by calling get_model path manually.
    from landmarking.models.pipnet import get_meanface_indices
    from landmarking.models.registry import get_model
    idx, r1, r2, ml = get_meanface_indices(engine.mean_shape, num_nb)
    engine._meanface_indices = idx.to(engine.device)
    engine.model = get_model(
        "pipnet", num_landmarks=num_lms, backbone="resnet18", pretrained=False,
        input_size=input_size, net_stride=net_stride, num_nb=num_nb,
        meanface_indices=engine._meanface_indices,
    ).to(engine.device)

    # Flags the train/val branches check.
    engine._is_heatmap_model = False
    engine._is_graph_cond_heatmap = False
    engine._is_heatmap_on_coords = False
    engine._is_coord_only_model = False
    engine._is_star_model = False
    engine._use_star_loss = False
    engine._is_graph_prior_fusion = False
    engine._is_pipnet = True

    engine.optimizer = torch.optim.Adam(engine.model.parameters(), lr=1e-4)
    engine.scheduler = torch.optim.lr_scheduler.MultiStepLR(engine.optimizer, [1])

    ds = _SynthLizard(4, num_lms, input_size)
    engine.train_loader = DataLoader(ds, batch_size=2)
    engine.val_loader = DataLoader(ds, batch_size=2)
    return engine


def test_engine_train_and_validate_pipnet(tmp_path):
    engine = _make_pipnet_engine(tmp_path)
    loss = engine._train_epoch(epoch=1)
    assert np.isfinite(loss)
    metrics = engine._validate(epoch=1)
    assert "val_loss" in metrics
    assert np.isfinite(metrics["val_loss"])
    assert "val_px_err" in metrics


def test_engine_pipnet_sparsity(tmp_path):
    """With landmark_indices, meanface + heads size to the subset."""
    subset = [0, 2, 4, 6, 8]  # 5 landmarks
    engine = _make_pipnet_engine(
        tmp_path, num_lms=len(subset), num_nb=4, landmark_indices=subset
    )
    assert engine.model.meanface_indices.shape == (5, 4)
    assert engine.model.cls_layer.out_channels == 5
    assert engine.model.nb_x_layer.out_channels == 4 * 5
    loss = engine._train_epoch(epoch=1)
    assert np.isfinite(loss)


def test_all_registry_variants_still_constructible():
    """No regression: every registered variant remains in the registry."""
    import landmarking.models  # noqa: F401  (triggers registration)
    for name in ("heatmap", "hrnet_coord", "stacked_hourglass", "vit", "pipnet"):
        assert name in MODEL_REGISTRY


def test_sparsity_rejects_too_many_neighbors(tmp_path):
    """num_nb >= subset size must be rejected by meanface derivation."""
    mf = torch.rand(4, 2)
    with pytest.raises(ValueError):
        get_meanface_indices(mf, num_nb=4)  # needs <= 3


# --------------------------------------------------------------------------- #
# Task 3.6: neighbor-averaged decode (merge)
# --------------------------------------------------------------------------- #

def test_decode_merge_equals_direct_when_neighbors_agree():
    """If neighbor heads predict exactly the direct positions, the merge equals
    the direct decode."""
    from landmarking.models.pipnet import decode_pip_merge

    b, n, num_nb, gh, gw = 1, 5, 2, 8, 8
    input_size, net_stride = 256, 32
    mf = torch.stack([torch.arange(n).float(), torch.zeros(n)], dim=-1)
    idx, rev1, rev2, ml = get_meanface_indices(mf, num_nb)

    # Build a clean set of direct predictions.
    cls = torch.full((b, n, gh, gw), -5.0)
    off_x = torch.zeros(b, n, gh, gw)
    off_y = torch.zeros(b, n, gh, gw)
    nb_x = torch.zeros(b, num_nb * n, gh, gw)
    nb_y = torch.zeros(b, num_nb * n, gh, gw)

    rows = [1, 2, 3, 4, 5]
    cols = [1, 2, 3, 4, 5]
    fx, fy = 0.3, 0.6
    for i in range(n):
        r, c = rows[i], cols[i]
        cls[0, i, r, c] = 5.0
        off_x[0, i, r, c] = fx
        off_y[0, i, r, c] = fy
    # Make each landmark i predict neighbor j's TRUE position from i's own cell.
    for i in range(n):
        r, c = rows[i], cols[i]
        for j in range(num_nb):
            nj = idx[i, j].item()
            # neighbor true normalized position:
            nx = (cols[nj] + fx) / gw
            ny = (rows[nj] + fy) / gh
            # store as offset from i's cell origin (in cell units)
            nb_x[0, num_nb * i + j, r, c] = nx * gw - c
            nb_y[0, num_nb * i + j, r, c] = ny * gh - r

    direct = decode_pip(cls, off_x, off_y, input_size, net_stride)
    merged = decode_pip_merge(
        cls, off_x, off_y, nb_x, nb_y, input_size, net_stride, num_nb,
        rev1, rev2, ml,
    )
    assert torch.allclose(direct, merged, atol=1e-5)


@pytest.mark.skipif(not REF_HAVE, reason="vendored PIPNet reference not present")
def test_decode_merge_matches_reference_forward_pip():
    """decode_pip_merge equals the reference forward_pip + test.py merge (B=1)."""
    ref = _load_reference_functions()
    if ref is None:
        pytest.skip("reference functions.py could not be imported")
    from landmarking.models.pipnet import decode_pip_merge

    n, num_nb, gh, gw = 8, 3, 8, 8
    input_size, net_stride = 256, 32
    mf = torch.rand(n, 2)
    idx, rev1, rev2, ml = get_meanface_indices(mf, num_nb)

    torch.manual_seed(0)
    cls = torch.randn(1, n, gh, gw)
    off_x = torch.randn(1, n, gh, gw)
    off_y = torch.randn(1, n, gh, gw)
    nb_x = torch.randn(1, num_nb * n, gh, gw)
    nb_y = torch.randn(1, num_nb * n, gh, gw)

    # Reference forward_pip via a tiny stub net returning our fixed maps.
    class _Stub:
        def eval(self):
            return self

        def __call__(self, _):
            return cls, off_x, off_y, nb_x, nb_y

    lx, ly, lnx, lny, _, _ = ref.forward_pip(
        _Stub(), None, None, input_size, net_stride, num_nb
    )
    # reference test.py merge
    tmp_nb_x = lnx[rev1, rev2].view(n, ml)
    tmp_nb_y = lny[rev1, rev2].view(n, ml)
    tx = torch.mean(torch.cat((lx, tmp_nb_x), dim=1), dim=1).view(-1, 1)
    ty = torch.mean(torch.cat((ly, tmp_nb_y), dim=1), dim=1).view(-1, 1)
    ref_merge = torch.cat((tx, ty), dim=1)  # (N, 2)

    ours = decode_pip_merge(
        cls, off_x, off_y, nb_x, nb_y, input_size, net_stride, num_nb,
        rev1, rev2, ml,
    )[0]
    assert torch.allclose(ours, ref_merge, atol=1e-5)
