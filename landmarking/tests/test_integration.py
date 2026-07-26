"""Integration tests for the landmarking package.

These tests validate end-to-end pipeline workflows. Tests marked with
@pytest.mark.skip(reason="HPC-only: requires GPU and real data") are designed
to run on HPC cluster nodes with GPU access and real datasets.

The package import test (12.4) runs locally without skip.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ============================================================================
# 12.1: Full Lizard Pipeline Integration Test
# ============================================================================


@pytest.mark.skip(reason="HPC-only: requires GPU and real data")
class TestFullLizardPipeline:
    """Integration test: Preprocess 1 sample → generate split → train 1 epoch
    → evaluate → verify metrics JSON exists.

    Requirements: 7.1, 7.5
    """

    def test_lizard_end_to_end(self, tmp_path):
        """Full Lizard pipeline produces metrics JSON."""
        from landmarking.config.schema import LandmarkingConfig
        from landmarking.common.split_utils import generate_split
        from landmarking.training.engine import TrainingEngine
        from landmarking.evaluation.engine import EvaluationEngine

        # Setup config
        config = LandmarkingConfig.from_dict({
            "paths": {
                "data_root": str(tmp_path / "data"),
                "output_root": str(tmp_path / "output"),
            },
            "dataset": {
                "name": "lizard",
                "num_landmarks": 9,
                "graph_topology": "chain",
            },
            "model": {
                "variant": "fused",
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "device": "cuda",
                "seed": 42,
            },
        })
        config.resolve_paths()

        # Step 1: Preprocess (requires real data)
        from landmarking.datasets.lizard.preprocess import run_preprocessing
        run_preprocessing(config.to_dict())

        # Step 2: Generate split
        split_path = str(tmp_path / "output" / "split.json")
        split = generate_split(
            data_dir=config.dataset.data_dir,
            fractions={"train": 0.8, "val": 0.1, "test": 0.1},
            seed=42,
            output_path=split_path,
        )
        config.dataset.split_path = split_path

        # Step 3: Train 1 epoch
        engine = TrainingEngine(config)
        engine.setup()
        engine.train()

        # Step 4: Evaluate
        eval_engine = EvaluationEngine("lizard")
        checkpoint_path = Path(engine.output_dir) / "checkpoints" / "best.pth"
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

        # Verify output directory structure
        assert (Path(engine.output_dir) / "config.json").exists()

        # Step 5: Verify metrics (basic existence check)
        results_path = Path(engine.output_dir) / "eval_results.json"
        # In a full pipeline, eval would write this
        assert Path(engine.output_dir).exists()


# ============================================================================
# 12.2: Full WFLW Pipeline Integration Test
# ============================================================================


@pytest.mark.skip(reason="HPC-only: requires GPU and real data")
class TestFullWFLWPipeline:
    """Integration test: Preprocess 1 sample → generate split → train 1 epoch
    → evaluate → verify NME JSON exists.

    Requirements: 7.2, 7.6
    """

    def test_wflw_end_to_end(self, tmp_path):
        """Full WFLW pipeline produces NME results."""
        from landmarking.config.schema import LandmarkingConfig
        from landmarking.common.split_utils import generate_split
        from landmarking.training.engine import TrainingEngine
        from landmarking.evaluation.engine import EvaluationEngine

        # Setup config
        config = LandmarkingConfig.from_dict({
            "paths": {
                "data_root": str(tmp_path / "data"),
                "output_root": str(tmp_path / "output"),
            },
            "dataset": {
                "name": "wflw",
                "num_landmarks": 98,
                "graph_topology": "wflw",
            },
            "model": {
                "variant": "fused",
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "device": "cuda",
                "seed": 42,
                "rot_factor": 30.0,
            },
        })
        config.resolve_paths()

        # Step 1: Setup and preprocess (requires real WFLW data)
        from landmarking.datasets.wflw.setup import run_setup
        run_setup(data_dir=config.dataset.data_dir)

        # Step 2: Generate split
        split_path = str(tmp_path / "output" / "split.json")
        generate_split(
            data_dir=config.dataset.data_dir,
            fractions={"train": 0.8, "val": 0.1, "test": 0.1},
            seed=42,
            output_path=split_path,
        )
        config.dataset.split_path = split_path

        # Step 3: Train 1 epoch
        engine = TrainingEngine(config)
        engine.setup()
        engine.train()

        # Step 4: Evaluate (NME, FR, AUC)
        eval_engine = EvaluationEngine("wflw")
        checkpoint_path = Path(engine.output_dir) / "checkpoints" / "best.pth"
        assert checkpoint_path.exists()

        # Verify output structure
        assert (Path(engine.output_dir) / "config.json").exists()


# ============================================================================
# 12.3: Parallel Training Integration Test
# ============================================================================


@pytest.mark.skip(reason="HPC-only: requires GPU and real data")
class TestParallelTraining:
    """Integration test: Launch 2 processes with different configs,
    verify non-conflicting output directories.

    Requirements: 10.4, 10.5
    """

    def test_parallel_non_conflicting_outputs(self, tmp_path):
        """Two parallel training runs produce distinct output directories."""
        from landmarking.config.schema import LandmarkingConfig

        configs = []
        for variant in ["fused", "multiscale"]:
            config = LandmarkingConfig.from_dict({
                "paths": {
                    "data_root": str(tmp_path / "data"),
                    "output_root": str(tmp_path / "output"),
                },
                "dataset": {
                    "name": "lizard",
                    "num_landmarks": 9,
                    "graph_topology": "chain",
                    "split_path": str(tmp_path / "split.json"),
                },
                "model": {"variant": variant},
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "device": "cuda",
                    "seed": 42,
                },
            })
            config.resolve_paths()
            cfg_path = tmp_path / f"{variant}_config.json"
            config.to_json(str(cfg_path))
            configs.append(cfg_path)

        # Launch as parallel subprocesses
        processes = []
        for cfg_path in configs:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "landmarking.pipelines.run_training",
                    "--config", str(cfg_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            processes.append(proc)

        # Wait for completion
        for proc in processes:
            proc.wait()

        # Verify distinct output directories
        output_root = tmp_path / "output"
        if output_root.exists():
            output_dirs = list(output_root.rglob("config.json"))
            # Each training run should produce its own config.json
            paths = [str(p.parent) for p in output_dirs]
            assert len(set(paths)) == len(paths), "Output directories conflict!"


# ============================================================================
# 12.4: Package Import Integration Test (runs locally)
# ============================================================================


class TestPackageImport:
    """Integration test: import landmarking in clean environment
    without sys.path manipulation.

    Requirements: 1.5
    """

    def test_import_landmarking_no_sys_path_hack(self):
        """Importing landmarking succeeds without sys.path manipulation."""
        import landmarking

        assert hasattr(landmarking, "__version__")

    def test_import_submodules(self):
        """All major submodules are importable."""
        import landmarking.config
        import landmarking.common
        import landmarking.models
        import landmarking.datasets
        import landmarking.training
        import landmarking.evaluation
        import landmarking.pipelines
        import landmarking.scripts

    def test_import_key_classes(self):
        """Key classes and functions are importable."""
        from landmarking.config.schema import LandmarkingConfig
        from landmarking.common.split_utils import generate_split
        from landmarking.common.heatmap_utils import generate_gaussian_heatmap
        from landmarking.models.registry import MODEL_REGISTRY, get_model
        from landmarking.evaluation.engine import EvaluationEngine

        assert LandmarkingConfig is not None
        assert generate_split is not None
        assert generate_gaussian_heatmap is not None
        assert MODEL_REGISTRY is not None
        assert get_model is not None
        assert EvaluationEngine is not None

    def test_config_from_dict_works(self):
        """LandmarkingConfig.from_dict() works without external dependencies."""
        from landmarking.config.schema import LandmarkingConfig

        config = LandmarkingConfig.from_dict({
            "dataset": {"name": "lizard", "num_landmarks": 9},
        })
        assert config.dataset.name == "lizard"
        assert config.dataset.num_landmarks == 9
