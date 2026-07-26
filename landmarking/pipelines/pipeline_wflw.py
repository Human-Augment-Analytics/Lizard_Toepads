"""WFLW pipeline: setup → generate split → train → evaluate (NME, FR, AUC).

Orchestrates the full WFLW facial landmark detection workflow.

Usage:
    python -m landmarking.pipelines.pipeline_wflw --config path/to/wflw.json
    python -m landmarking.pipelines.pipeline_wflw --config path/to/wflw.json --skip-setup
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.schema import LandmarkingConfig
from ..common.split_utils import generate_split
from ..evaluation.engine import EvaluationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full WFLW landmark detection pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to WFLW JSON configuration file.",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip WFLW dataset setup (use existing preprocessed files).",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip split generation (use existing split JSON).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Override model variant.",
    )
    return parser.parse_args(argv)


def run_setup(config: LandmarkingConfig) -> None:
    """Run WFLW dataset setup (download, extract, preprocess)."""
    logger.info("Running WFLW dataset setup...")
    from ..datasets.wflw.setup import run_setup as _run_setup

    _run_setup(data_dir=config.dataset.data_dir)
    logger.info("WFLW setup complete.")


def run_split_generation(config: LandmarkingConfig) -> str:
    """Generate WFLW train/val/test split.

    WFLW has a predefined test set, so we use predefined_test parameter.

    Returns:
        Path to the generated split JSON file.
    """
    logger.info("Generating WFLW train/val/test split...")
    split_path = str(
        Path(config.paths.output_root) / config.dataset.name / "split.json"
    )

    # WFLW uses predefined test set from annotations
    # For now, use fraction-based split (predefined test can be passed via config)
    generate_split(
        data_dir=config.dataset.data_dir,
        fractions={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=config.training.seed,
        output_path=split_path,
    )
    logger.info(f"Split saved to: {split_path}")
    return split_path


def run_training(config: LandmarkingConfig, split_path: str) -> Optional[str]:
    """Run WFLW training.

    Returns:
        Output directory path on success, None on failure.
    """
    from ..training.engine import TrainingEngine

    config.dataset.split_path = split_path

    try:
        engine = TrainingEngine(config)
        engine.setup()
        engine.train()
        return engine.output_dir
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None


def run_evaluation(config: LandmarkingConfig, output_dir: str) -> dict:
    """Run WFLW evaluation (NME, FR, AUC).

    Returns:
        Evaluation results dict.
    """
    logger.info("Running WFLW evaluation...")
    engine = EvaluationEngine(config.dataset.name)

    ckpt_path = Path(output_dir) / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        logger.warning(f"No best checkpoint found at {ckpt_path}")
        return {"status": "no_checkpoint"}

    # Note: full evaluation requires dataloader setup (HPC only)
    results = {"checkpoint": str(ckpt_path), "status": "pending"}

    # Save results
    results_path = str(Path(output_dir) / "eval_results.json")
    engine.save_results(results, results_path)
    logger.info(f"Evaluation results saved to: {results_path}")

    return results


def main(argv=None):
    """Main entry point for the WFLW pipeline."""
    args = parse_args(argv)

    # Load config
    config = LandmarkingConfig.from_json(args.config)
    config.resolve_paths()

    # Apply CLI overrides
    if args.device is not None:
        config.training.device = args.device
    if args.variant is not None:
        config.model.variant = args.variant

    # Step 1: Setup
    if not args.skip_setup:
        run_setup(config)

    # Step 2: Generate split
    if args.skip_split and config.dataset.split_path:
        split_path = config.dataset.split_path
    else:
        split_path = run_split_generation(config)

    # Step 3: Train
    logger.info(
        f"Training: variant={config.model.variant}, "
        f"epochs={config.training.epochs}"
    )
    output_dir = run_training(config, split_path)

    if output_dir is None:
        logger.error("Training failed. Exiting.")
        sys.exit(1)

    # Step 4: Evaluate (NME, FR, AUC)
    results = run_evaluation(config, output_dir)

    logger.info("WFLW pipeline complete.")
    if results.get("nme"):
        logger.info(f"  NME (full): {results['nme'].get('full', 'N/A')}")
        logger.info(f"  FR (full):  {results['fr'].get('full', 'N/A')}")
        logger.info(f"  AUC (full): {results['auc'].get('full', 'N/A')}")


if __name__ == "__main__":
    main()
