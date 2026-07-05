"""
WFLW fraction-aware split generator.

Reads preprocessed .pt crops from pt_crops/train/ and pt_crops/test/,
samples a training fraction, and writes a split JSON file compatible
with the existing LizardDataset and all training scripts.

Usage:
    python generate_split.py \\
        --data-dir alternative-datasets/wflw/ \\
        --fraction 0.25 \\
        --seed 42 \\
        --output splits/wflw_0.25_seed42.json

    # Val fraction (default 0.1) is taken from the remaining train pool
    # Test set is always the full official WFLW test split (never subsampled)
"""
import argparse
import sys
from pathlib import Path

# Allow import of split_utils from sibling common/ package
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.split_utils import sample_fraction, write_split


def main():
    parser = argparse.ArgumentParser(
        description="Generate a WFLW train/val/test split at a given training fraction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Parent directory containing pt_crops/train/ and pt_crops/test/",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of the training pool to use for training (0 < fraction <= 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the split JSON file",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the remaining pool (after train) used for val (default: 0.2)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_pool_dir = data_dir / "pt_crops" / "train"
    test_dir = data_dir / "pt_crops" / "test"

    # Validate directories
    if not train_pool_dir.exists():
        print(
            f"ERROR: training .pt directory not found: {train_pool_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not test_dir.exists():
        print(
            f"ERROR: test .pt directory not found: {test_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Collect files
    train_pool = sorted([str(p) for p in train_pool_dir.glob("*.pt")])
    test_files = sorted([str(p) for p in test_dir.glob("*.pt")])

    if not train_pool:
        print(
            f"ERROR: no .pt files found in {train_pool_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not test_files:
        print(
            f"ERROR: no .pt files found in {test_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sample training fraction
    try:
        train_files = sample_fraction(train_pool, args.fraction, args.seed)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Val is sampled from the pool before train when fraction == 1.0,
    # otherwise from the remaining files after train is removed.
    # This avoids an empty val set when fraction=1.0.
    if args.fraction >= 1.0 and args.val_fraction > 0:
        # Reserve val first, then use the rest for train
        try:
            val_files = sample_fraction(train_pool, args.val_fraction, args.seed + 1)
        except ValueError as e:
            print(f"ERROR sampling val: {e}", file=sys.stderr)
            sys.exit(1)
        val_set = set(val_files)
        train_files = [p for p in train_pool if p not in val_set]
    else:
        # Val is sampled from what remains after train
        train_set = set(train_files)
        remaining = [p for p in train_pool if p not in train_set]
        if remaining and args.val_fraction > 0:
            try:
                val_files = sample_fraction(remaining, args.val_fraction, args.seed + 1)
            except ValueError as e:
                print(f"ERROR sampling val: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            val_files = []

    write_split(train_files, val_files, test_files, args.output)

    print(
        f"Split written to {args.output}\n"
        f"  Train: {len(train_files)} / {len(train_pool)} "
        f"({args.fraction * 100:.0f}%)\n"
        f"  Val:   {len(val_files)}\n"
        f"  Test:  {len(test_files)} (full official test set)"
    )


if __name__ == "__main__":
    main()
