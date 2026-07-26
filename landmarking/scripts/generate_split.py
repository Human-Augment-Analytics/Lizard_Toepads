"""CLI tool for standalone deterministic split generation.

Usage:
    python -m landmarking.scripts.generate_split --data-dir /path/to/data --output split.json
    python -m landmarking.scripts.generate_split --data-dir /path/to/data --output split.json --seed 123
    python -m landmarking.scripts.generate_split --data-dir /path/to/data --output split.json --train 0.7 --val 0.15 --test 0.15
"""

import argparse
import logging
import sys

from ..common.split_utils import generate_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a deterministic train/val/test split JSON file."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data files to split.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the split JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Training fraction (default: 0.8).",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation fraction (default: 0.1).",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test fraction (default: 0.1).",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="*.pt",
        help="Glob pattern for finding data files (default: '*.pt').",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point for split generation CLI."""
    args = parse_args(argv)

    fractions = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }

    total = sum(fractions.values())
    if total > 1.0:
        logger.error(f"Split fractions sum to {total:.3f}, must be <= 1.0")
        sys.exit(1)

    logger.info(f"Generating split from: {args.data_dir}")
    logger.info(f"Fractions: train={args.train}, val={args.val}, test={args.test}")
    logger.info(f"Seed: {args.seed}, Pattern: {args.glob_pattern}")

    split = generate_split(
        data_dir=args.data_dir,
        fractions=fractions,
        seed=args.seed,
        output_path=args.output,
        glob_pattern=args.glob_pattern,
    )

    logger.info(
        f"Split generated: "
        f"train={len(split['train'])}, "
        f"val={len(split['val'])}, "
        f"test={len(split['test'])}"
    )
    logger.info(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
