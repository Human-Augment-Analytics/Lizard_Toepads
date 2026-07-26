"""WFLW one-shot setup script.

Runs all preprocessing steps in order:
  1. Preprocess train annotations → pt_crops/train/
  2. Preprocess test annotations  → pt_crops/test/
  3. Generate split               → splits/wflw_{fraction}_seed{seed}.json
  4. Compute mean shape           → mean_shapes/mean_shape_wflw_{fraction}_seed{seed}.pt

All paths are parameterized — no hardcoded absolute paths.

Usage:
    python -m landmarking.datasets.wflw.setup \\
        --data-dir /path/to/wflw_data/ \\
        --output-dir /path/to/output/

    # Quick smoke test at 10% data:
    python -m landmarking.datasets.wflw.setup \\
        --data-dir /path/to/wflw_data/ \\
        --fraction 0.1
"""

import argparse
import json
import sys
from pathlib import Path

from .preprocess import preprocess_wflw
from .compute_mean_shape import compute_mean_shape
from ...common.split_utils import generate_split

# Default annotation/image subdirectory names (standard WFLW download structure)
DEFAULT_TRAIN_ANNOTATION = (
    "WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
)
DEFAULT_TEST_ANNOTATION = (
    "WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
)
DEFAULT_IMAGE_ROOT = "WFLW_images"


def run_setup(
    data_dir: str,
    output_dir: str = None,
    annotation_root: str = None,
    image_root: str = None,
    fraction: float = 0.8,
    seed: int = 42,
):
    """Run the full WFLW setup pipeline.

    Args:
        data_dir: Root directory containing raw WFLW data.
        output_dir: Directory for outputs (splits, mean shapes). Defaults to data_dir.
        annotation_root: Override path to WFLW_annotations/ directory.
        image_root: Override path to WFLW_images/ directory.
        fraction: Training fraction for split generation.
        seed: Random seed.
    """
    data_dir = Path(data_dir).resolve()
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir).resolve()

    # Resolve annotation and image paths
    if annotation_root:
        ann_root = Path(annotation_root).resolve()
        train_annotation = (
            ann_root / "list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
        )
        test_annotation = (
            ann_root / "list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
        )
    else:
        train_annotation = data_dir / DEFAULT_TRAIN_ANNOTATION
        test_annotation = data_dir / DEFAULT_TEST_ANNOTATION

    if image_root:
        img_root = Path(image_root).resolve()
    else:
        img_root = data_dir / DEFAULT_IMAGE_ROOT

    # Output paths
    train_pt_dir = output_dir / "pt_crops" / "train"
    test_pt_dir = output_dir / "pt_crops" / "test"
    split_name = f"wflw_{fraction}_seed{seed}.json"
    split_path = output_dir / "splits" / split_name
    mean_shape_name = f"mean_shape_wflw_{fraction}_seed{seed}.pt"
    mean_shape_path = output_dir / "mean_shapes" / mean_shape_name

    # Validate inputs
    for path, label in [
        (train_annotation, "train annotation file"),
        (test_annotation, "test annotation file"),
        (img_root, "image root directory"),
    ]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"\nWFLW Setup")
    print(f"  data_dir:      {data_dir}")
    print(f"  image_root:    {img_root}")
    print(f"  fraction:      {fraction}")
    print(f"  seed:          {seed}")
    print(f"  split output:  {split_path}")
    print(f"  mean shape:    {mean_shape_path}")

    # Step 1 — preprocess train
    print("\n[1/4] Preprocessing train split...")
    preprocess_wflw(str(train_annotation), str(img_root), str(train_pt_dir))

    # Step 2 — preprocess test
    print("\n[2/4] Preprocessing test split...")
    preprocess_wflw(str(test_annotation), str(img_root), str(test_pt_dir))

    # Step 3 — generate split
    print(f"\n[3/4] Generating split at fraction={fraction}...")
    # Use predefined test set from test_pt_dir
    test_files = sorted([str(p) for p in test_pt_dir.glob("*.pt")])
    train_data_dir = str(train_pt_dir)

    fractions = {"train": fraction, "val": 1.0 - fraction, "test": 0.0}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    generate_split(
        data_dir=train_data_dir,
        fractions=fractions,
        seed=seed,
        output_path=str(split_path),
        predefined_test=test_files,
    )

    # Step 4 — compute mean shape
    print("\n[4/4] Computing mean shape...")
    compute_mean_shape(str(split_path), str(mean_shape_path))

    print(f"\nSetup complete.")
    print(f"  Split: {split_path}")
    print(f"  Mean shape: {mean_shape_path}")


def main():
    parser = argparse.ArgumentParser(
        description="One-shot WFLW setup: preprocess → split → mean shape"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--annotation-root", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_setup(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        annotation_root=args.annotation_root,
        image_root=args.image_root,
        fraction=args.fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
