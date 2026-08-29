"""Training-engine dispatch tests for the cephalometric branch.

CPU-only integration tests using synthetic ``.pt`` fixtures (no GPU, no
network, no real ISBI data). Verifies that ``TrainingEngine.setup`` builds
``CephalometricDataset`` dataloaders and the ``cephalometric`` edge index for
``dataset.name == "cephalometric"`` (Requirements 6.1, 6.2), and that the
existing Lizard dispatch path is unchanged (Requirement 12.5).
"""

import torch

from landmarking.config.schema import LandmarkingConfig
from landmarking.common.graph_topologies import get_edge_index
from landmarking.training.engine import TrainingEngine
from landmarking.datasets.cephalometric.dataset import CephalometricDataset
from landmarking.datasets.lizard.dataset import LizardDataset


def _write_ceph_pt(path, num_landmarks=19):
    """Write a single synthetic cephalometric .pt sample to ``path``."""
    torch.save(
        {
            "image": (torch.rand(3, 512, 512) * 255).to(torch.uint8),
            "tps": torch.rand(num_landmarks, 2).to(torch.float32),
            "orig_size": torch.tensor([600.0, 500.0], dtype=torch.float32),
            "pixel_spacing": torch.tensor(0.1, dtype=torch.float32),
            "split": "train",
        },
        str(path),
    )


def _write_lizard_pt(path, num_landmarks=9):
    """Write a single synthetic Lizard-style .pt sample (pixel-space tps)."""
    torch.save(
        {
            "image": (torch.rand(3, 512, 512) * 255).to(torch.uint8),
            # Lizard tps are in 512-pixel space.
            "tps": (torch.rand(num_landmarks, 2) * 511.0).to(torch.float32),
            "orig_size": torch.tensor([512.0, 512.0], dtype=torch.float32),
        },
        str(path),
    )


def _make_ceph_data_dir(tmp_path, n=6):
    """Create <data_dir>/train/ with n synthetic cephalometric .pt files."""
    data_dir = tmp_path / "Cephalometric_data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    for i in range(n):
        _write_ceph_pt(train_dir / f"sample_{i:03d}.pt")
    return data_dir


def _make_lizard_data_dir(tmp_path, n=6):
    """Create <data_dir>/train/ with n synthetic Lizard .pt files."""
    data_dir = tmp_path / "Lizard_data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    for i in range(n):
        _write_lizard_pt(train_dir / f"sample_{i:03d}.pt")
    return data_dir


def _ceph_config(data_dir):
    return LandmarkingConfig.from_dict(
        {
            "paths": {"data_root": "", "output_root": str(data_dir / "runs")},
            "dataset": {
                "name": "cephalometric",
                "num_landmarks": 19,
                "graph_topology": "cephalometric",
                "input_size": 512,
                "data_dir": str(data_dir),
                "mean_shape_path": "",
                "pixel_spacing": 0.1,
            },
            "model": {"variant": "fused_global"},
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "val_batch_size": 2,
                "epochs": 1,
                "seed": 0,
            },
        }
    ).resolve_paths()


def _lizard_config(data_dir):
    return LandmarkingConfig.from_dict(
        {
            "paths": {"data_root": "", "output_root": str(data_dir / "runs")},
            "dataset": {
                "name": "lizard",
                "num_landmarks": 9,
                "graph_topology": "chain",
                "input_size": 512,
                "data_dir": str(data_dir),
                "mean_shape_path": "",
            },
            "model": {"variant": "fused"},
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "val_batch_size": 2,
                "epochs": 1,
                "seed": 0,
            },
        }
    ).resolve_paths()


class TestCephalometricDispatch:
    def test_engine_builds_cephalometric_dataloaders(self, tmp_path):
        """Req 6.1: setup() builds CephalometricDataset train/val loaders."""
        data_dir = _make_ceph_data_dir(tmp_path)
        engine = TrainingEngine(_ceph_config(data_dir))
        engine.setup()

        assert isinstance(engine.train_loader.dataset, CephalometricDataset)
        assert isinstance(engine.val_loader.dataset, CephalometricDataset)

    def test_engine_builds_cephalometric_edge_index(self, tmp_path):
        """Req 6.2: edge_index equals the cephalometric topology."""
        data_dir = _make_ceph_data_dir(tmp_path)
        engine = TrainingEngine(_ceph_config(data_dir))
        engine.setup()

        expected = get_edge_index("cephalometric")
        assert engine.edge_index is not None
        assert torch.equal(engine.edge_index.cpu(), expected)

    def test_cephalometric_sample_shapes(self, tmp_path):
        """A dataset sample yields image (3,512,512) and coords (19,2)."""
        data_dir = _make_ceph_data_dir(tmp_path)
        engine = TrainingEngine(_ceph_config(data_dir))
        engine.setup()

        # Read directly from the dataset to avoid DataLoader multiprocessing.
        img, coords, *_ = engine.train_loader.dataset[0]
        assert img.shape == (3, 512, 512)
        assert coords.shape == (19, 2)
        assert img.dtype == torch.float32
        assert coords.dtype == torch.float32


class TestLizardDispatchUnchanged:
    def test_engine_still_builds_lizard_dataset(self, tmp_path):
        """Req 12.5: dataset.name == "lizard" still routes to LizardDataset."""
        data_dir = _make_lizard_data_dir(tmp_path)
        engine = TrainingEngine(_lizard_config(data_dir))
        engine.setup()

        assert isinstance(engine.train_loader.dataset, LizardDataset)
        assert isinstance(engine.val_loader.dataset, LizardDataset)
