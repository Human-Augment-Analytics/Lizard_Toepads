"""Lizard benchmark pipeline: preprocess → split → train all variants → evaluate → report.

Orchestrates the full Lizard landmark detection benchmark. Supports parallel
training of multiple model variants via subprocess-based GPU execution.

Usage:
    python -m landmarking.pipelines.benchmark_lizard --config path/to/lizard.json
    python -m landmarking.pipelines.benchmark_lizard --config path/to/lizard.json --parallel
    python -m landmarking.pipelines.benchmark_lizard --config path/to/lizard.json --variants fused multiscale
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from ..config.schema import LandmarkingConfig
from ..common.split_utils import generate_split
from ..evaluation.engine import EvaluationEngine
from ..evaluation.report import generate_html_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default model variants to benchmark
DEFAULT_VARIANTS = [
    "standard",
    "multiscale",
    "coord",
    "fused",
    "fused_global",
    "hinit",
    "heatmap",
    "hrnet_coord",
    "stacked_hourglass",
    "vit",
]


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full Lizard landmark detection benchmark."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base Lizard JSON configuration file.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Model variants to benchmark (default: all registered).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run training jobs in parallel via subprocesses.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing step (use existing .pt files).",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip split generation (use existing split JSON).",
    )
    return parser.parse_args(argv)


def run_preprocessing(config: LandmarkingConfig) -> None:
    """Run Lizard preprocessing pipeline."""
    logger.info("Running Lizard preprocessing...")
    from ..datasets.lizard.preprocess import run_preprocessing as _run_preprocess

    _run_preprocess(config.to_dict())
    logger.info("Preprocessing complete.")


def run_split_generation(config: LandmarkingConfig) -> str:
    """Generate deterministic train/val/test split.

    Returns:
        Path to the generated split JSON file.
    """
    logger.info("Generating train/val/test split...")
    split_path = str(
        Path(config.paths.output_root) / config.dataset.name / "split.json"
    )
    generate_split(
        data_dir=config.dataset.data_dir,
        fractions={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=config.training.seed,
        output_path=split_path,
    )
    logger.info(f"Split saved to: {split_path}")
    return split_path


def run_training_subprocess(config_path: str, variant: str, device: str = "cuda") -> subprocess.Popen:
    """Launch a training subprocess for a single variant.

    Args:
        config_path: Path to the variant-specific config JSON.
        variant: Model variant name.
        device: CUDA device string.

    Returns:
        The subprocess.Popen handle.
    """
    cmd = [
        sys.executable, "-m", "landmarking.pipelines.run_training",
        "--config", config_path,
        "--device", device,
    ]
    logger.info(f"Launching training subprocess: {variant} on {device}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def train_variant_sequential(config: LandmarkingConfig, variant: str, split_path: str) -> Optional[str]:
    """Train a single model variant sequentially (in-process).

    Args:
        config: Base configuration.
        variant: Model variant key.
        split_path: Path to split JSON.

    Returns:
        Output directory path on success, None on failure.
    """
    from ..training.engine import TrainingEngine

    # Create variant-specific config
    variant_config = LandmarkingConfig.from_dict(config.to_dict())
    variant_config.model.variant = variant
    variant_config.dataset.split_path = split_path
    variant_config.resolve_paths()

    try:
        engine = TrainingEngine(variant_config)
        engine.setup()
        engine.train()
        return engine.output_dir
    except Exception as e:
        logger.error(f"Training failed for variant '{variant}': {e}")
        return None


def train_all_parallel(config: LandmarkingConfig, variants: List[str], split_path: str) -> List[str]:
    """Train all variants in parallel using subprocesses.

    Args:
        config: Base configuration.
        variants: List of model variant keys.
        split_path: Path to split JSON.

    Returns:
        List of output directory paths for successful runs.
    """
    tmp_dir = Path(config.paths.output_root) / config.dataset.name / "_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    config_paths = []

    for variant in variants:
        variant_config = LandmarkingConfig.from_dict(config.to_dict())
        variant_config.model.variant = variant
        variant_config.dataset.split_path = split_path
        variant_config.resolve_paths()

        cfg_path = str(tmp_dir / f"{variant}.json")
        variant_config.to_json(cfg_path)
        config_paths.append(cfg_path)

        proc = run_training_subprocess(cfg_path, variant)
        processes.append((variant, proc))

    # Wait for all to finish
    output_dirs = []
    for variant, proc in processes:
        stdout, stderr = proc.communicate()
        if proc.returncode == 0:
            logger.info(f"Training complete for variant: {variant}")
            # Infer output dir from config
            out_dir = str(
                Path(config.paths.output_root) / config.dataset.name / variant
            )
            output_dirs.append(out_dir)
        else:
            logger.error(
                f"Training FAILED for variant '{variant}' "
                f"(exit code {proc.returncode}): {stderr.decode()[:500]}"
            )

    return output_dirs


def train_all_sequential(config: LandmarkingConfig, variants: List[str], split_path: str) -> List[str]:
    """Train all variants sequentially.

    Returns:
        List of output directory paths for successful runs.
    """
    output_dirs = []
    for variant in variants:
        logger.info(f"Training variant: {variant}")
        out_dir = train_variant_sequential(config, variant, split_path)
        if out_dir:
            output_dirs.append(out_dir)
    return output_dirs


def run_evaluation(config: LandmarkingConfig, output_dirs: List[str]) -> dict:
    """Run evaluation on all trained models.

    Returns:
        Dict mapping variant → evaluation results.
    """
    logger.info("Running evaluation...")
    engine = EvaluationEngine(config.dataset.name)
    all_results = {}

    for out_dir in output_dirs:
        variant = Path(out_dir).name
        ckpt_path = Path(out_dir) / "checkpoints" / "best.pth"
        if ckpt_path.exists():
            logger.info(f"Evaluating: {variant}")
            # Note: full evaluation requires dataloader setup (HPC only)
            all_results[variant] = {"checkpoint": str(ckpt_path), "status": "pending"}
        else:
            logger.warning(f"No best checkpoint found for: {variant}")

    return all_results


def main(argv=None):
    """Main entry point for the Lizard benchmark pipeline."""
    args = parse_args(argv)

    # Load config
    config = LandmarkingConfig.from_json(args.config)
    config.resolve_paths()

    variants = args.variants or DEFAULT_VARIANTS

    # Step 1: Preprocess
    if not args.skip_preprocess:
        run_preprocessing(config)

    # Step 2: Generate split
    if args.skip_split and config.dataset.split_path:
        split_path = config.dataset.split_path
    else:
        split_path = run_split_generation(config)

    # Step 3: Train all variants
    if args.parallel:
        output_dirs = train_all_parallel(config, variants, split_path)
    else:
        output_dirs = train_all_sequential(config, variants, split_path)

    # Step 4: Evaluate
    results = run_evaluation(config, output_dirs)

    # Step 5: Generate HTML report
    report_path = str(
        Path(config.paths.output_root) / config.dataset.name / "benchmark_report.html"
    )
    generate_html_report(results, report_path)
    logger.info(f"Benchmark report saved to: {report_path}")

    logger.info("Benchmark pipeline complete.")


if __name__ == "__main__":
    main()
