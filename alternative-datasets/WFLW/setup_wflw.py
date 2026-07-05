"""
WFLW one-shot setup script.

Runs all preprocessing steps in order from a single --data-dir argument:
  1. Preprocess train annotations → pt_crops/train/
  2. Preprocess test annotations  → pt_crops/test/
  3. Generate split at --fraction  → splits/wflw_{fraction}_seed{seed}.json
  4. Compute mean shape            → mean_shapes/mean_shape_wflw_{fraction}_seed{seed}.pt
  5. Update wflw-config.json with the resolved mean_shape_path

After this script completes, run training directly:
    python alternative-datasets/wflw/run_wflw.py \\
        --split alternative-datasets/wflw/splits/wflw_1.0_seed42.json

Usage:
    python alternative-datasets/wflw/setup_wflw.py \\
        --data-dir /storage/wflw/

    # With non-default annotation/image locations:
    python alternative-datasets/wflw/setup_wflw.py \\
        --data-dir /storage/wflw/ \\
        --annotation-root /storage/wflw/WFLW_annotations/ \\
        --image-root /storage/wflw/WFLW_images/

    # Quick smoke test at 10% data:
    python alternative-datasets/wflw/setup_wflw.py \\
        --data-dir /storage/wflw/ \\
        --fraction 0.1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "configs" / "wflw-config.json"

# Default annotation/image subdirectory names (standard WFLW download structure)
DEFAULT_TRAIN_ANNOTATION = "WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
DEFAULT_TEST_ANNOTATION  = "WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
DEFAULT_IMAGE_ROOT       = "WFLW_images"


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd], check=False)
    if result.returncode != 0:
        print(f"\nERROR: step failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="One-shot WFLW setup: preprocess → split → mean shape → config update"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Root directory containing raw WFLW data. "
            "Expected to contain WFLW_annotations/ and WFLW_images/ subdirs "
            "(override with --annotation-root and --image-root if different)."
        ),
    )
    parser.add_argument(
        "--annotation-root",
        type=str,
        default=None,
        help="Override path to WFLW_annotations/ directory",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Override path to WFLW_images/ directory",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Training fraction to generate split at (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split sampling (default: 42)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    # Resolve annotation and image paths
    ann_root = Path(args.annotation_root).resolve() if args.annotation_root else data_dir
    img_root = Path(args.image_root).resolve() if args.image_root else data_dir / DEFAULT_IMAGE_ROOT

    train_annotation = ann_root / DEFAULT_TRAIN_ANNOTATION if not args.annotation_root else Path(args.annotation_root) / DEFAULT_TRAIN_ANNOTATION
    test_annotation  = ann_root / DEFAULT_TEST_ANNOTATION  if not args.annotation_root else Path(args.annotation_root) / DEFAULT_TEST_ANNOTATION

    # If annotation_root was given as a full path ending in WFLW_annotations, handle that
    if args.annotation_root:
        ann_root = Path(args.annotation_root).resolve()
        train_annotation = ann_root / "list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
        test_annotation  = ann_root / "list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
    else:
        train_annotation = data_dir / DEFAULT_TRAIN_ANNOTATION
        test_annotation  = data_dir / DEFAULT_TEST_ANNOTATION

    if not args.image_root:
        img_root = data_dir / DEFAULT_IMAGE_ROOT

    # Derived output paths — all under data_dir
    train_pt_dir  = data_dir / "pt_crops" / "train"
    test_pt_dir   = data_dir / "pt_crops" / "test"
    split_name    = f"wflw_{args.fraction}_seed{args.seed}.json"
    split_path    = SCRIPT_DIR / "splits" / split_name
    mean_shape_name = f"mean_shape_wflw_{args.fraction}_seed{args.seed}.pt"
    mean_shape_path = SCRIPT_DIR / "mean_shapes" / mean_shape_name

    # Validate inputs before running anything
    for path, label in [
        (train_annotation, "train annotation file"),
        (test_annotation,  "test annotation file"),
        (img_root,         "image root directory"),
    ]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"\nWFLW Setup")
    print(f"  data_dir:      {data_dir}")
    print(f"  image_root:    {img_root}")
    print(f"  fraction:      {args.fraction}")
    print(f"  seed:          {args.seed}")
    print(f"  split output:  {split_path}")
    print(f"  mean shape:    {mean_shape_path}")

    py = sys.executable
    preprocess = SCRIPT_DIR / "preprocess.py"
    generate_split = SCRIPT_DIR / "generate_split.py"
    compute_mean = SCRIPT_DIR / "compute_mean_shape.py"

    # Step 1 — preprocess train
    run(
        [py, preprocess,
         "--annotation-file", train_annotation,
         "--image-root", img_root,
         "--output-dir", train_pt_dir],
        "Step 1/4 — Preprocessing train split"
    )

    # Step 2 — preprocess test
    run(
        [py, preprocess,
         "--annotation-file", test_annotation,
         "--image-root", img_root,
         "--output-dir", test_pt_dir],
        "Step 2/4 — Preprocessing test split"
    )

    # Step 3 — generate split
    (SCRIPT_DIR / "splits").mkdir(parents=True, exist_ok=True)
    run(
        [py, generate_split,
         "--data-dir", data_dir,
         "--fraction", args.fraction,
         "--seed", args.seed,
         "--output", split_path],
        f"Step 3/4 — Generating split at fraction={args.fraction}"
    )

    # Step 4 — compute mean shape
    (SCRIPT_DIR / "mean_shapes").mkdir(parents=True, exist_ok=True)
    run(
        [py, compute_mean,
         "--split", split_path,
         "--output", mean_shape_path],
        "Step 4/4 — Computing mean shape"
    )

    # Update wflw-config.json with resolved paths
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        config["training_data_path"] = str(data_dir / "pt_crops")
        config["mean_shape_path"] = str(mean_shape_path)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
        print(f"\nUpdated {CONFIG_PATH}:")
        print(f"  training_data_path → {config['training_data_path']}")
        print(f"  mean_shape_path    → {config['mean_shape_path']}")

    print(f"\n{'='*60}")
    print("  Setup complete. To train:")
    print(f"  python {SCRIPT_DIR}/run_wflw.py \\")
    print(f"      --split {split_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
