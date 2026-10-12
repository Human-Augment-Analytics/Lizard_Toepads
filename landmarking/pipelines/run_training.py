"""CLI entry point for a single training run.

Usage:
    python -m landmarking.pipelines.run_training --config path/to/config.json
    python -m landmarking.pipelines.run_training --config path/to/config.json --device cuda:1
"""

import argparse
import logging
import sys

from ..config.schema import LandmarkingConfig
from ..training.engine import TrainingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a single landmark detection training session."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override training (and validation) batch size. Exists because the "
            "experiment configs are sized for a 48GB cluster GPU: the HRNet "
            "4-branch fusion is 960 channels at 256x256 for a 512px input, "
            "which is ~252MB of activations PER SAMPLE, so batch 32 needs tens "
            "of GB. Lower this to run the same config interactively on a "
            "desktop GPU without editing the config and desynchronizing an arm "
            "from its controls."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Override model variant (e.g., 'fused', 'heatmap', 'multiscale').",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point for single training run CLI."""
    args = parse_args(argv)

    # Load and resolve config
    logger.info(f"Loading config from: {args.config}")
    config = LandmarkingConfig.from_json(args.config)
    config.resolve_paths()

    # Apply CLI overrides
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.variant is not None:
        config.model.variant = args.variant
    if args.batch_size is not None:
        # Applied to BOTH loaders. The experiment configs set batch_size ==
        # val_batch_size, and leaving validation at the config's larger value
        # would reintroduce the memory spike this override exists to avoid --
        # validation forwards the same backbone at the same resolution.
        config.training.batch_size = args.batch_size
        config.training.val_batch_size = args.batch_size

    logger.info(
        f"Training: dataset={config.dataset.name}, "
        f"variant={config.model.variant}, "
        f"epochs={config.training.epochs}, "
        f"batch_size={config.training.batch_size}, "
        f"device={config.training.device}"
    )

    # Instantiate and run training engine
    engine = TrainingEngine(config)
    engine.setup()
    engine.train()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
