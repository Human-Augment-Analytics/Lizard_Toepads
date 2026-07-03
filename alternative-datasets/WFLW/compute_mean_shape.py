"""
Compute per-landmark mean shape from a training split.

Reads the "tps" tensor from each training .pt file and computes
the element-wise mean across all samples, producing a (N, 2) float32
tensor saved as a .pt file.

Must be run separately for each training fraction to avoid data leakage
from non-training samples.

Usage:
    python compute_mean_shape.py \\
        --split splits/wflw_1.0_seed42.json \\
        --output mean_shapes/mean_shape_wflw_1.0_seed42.pt
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-landmark mean shape from training split"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to split JSON file (must have a 'train' key)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for mean_shape.pt",
    )
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        sys.exit(1)

    with open(split_path) as f:
        split_data = json.load(f)

    train_paths = split_data.get("train", [])
    if not train_paths:
        raise ValueError(
            f"The 'train' list in {split_path} is empty — "
            "cannot compute mean shape from zero samples."
        )

    logging.info(f"Computing mean shape from {len(train_paths)} training samples...")

    tps_list = []
    skipped = 0
    expected_shape = None

    for pt_path in train_paths:
        try:
            data = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            logging.warning(f"Failed to load {pt_path}: {e}")
            skipped += 1
            continue

        tps = data.get("tps")
        if tps is None:
            logging.warning(f"Missing 'tps' key in {pt_path}")
            skipped += 1
            continue

        if expected_shape is None:
            expected_shape = tps.shape
        elif tps.shape != expected_shape:
            logging.warning(
                f"Unexpected 'tps' shape {tps.shape} (expected {expected_shape}) "
                f"in {pt_path} — skipping"
            )
            skipped += 1
            continue

        tps_list.append(tps.float())

    skip_pct = skipped / len(train_paths) * 100
    if skip_pct > 5.0:
        logging.warning(
            f"More than 5% of training files were skipped "
            f"({skipped}/{len(train_paths)}, {skip_pct:.1f}%) — "
            "check data integrity"
        )

    if not tps_list:
        raise ValueError(
            "No valid 'tps' tensors found in training split — "
            "cannot compute mean shape."
        )

    # Stack and compute mean: (N, landmarks, 2) → (landmarks, 2)
    stacked = torch.stack(tps_list, dim=0)  # (N, landmarks, 2)
    mean_shape = stacked.mean(dim=0)        # (landmarks, 2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_shape, str(output_path))

    logging.info(
        f"Mean shape saved to {output_path} "
        f"(shape: {mean_shape.shape}, "
        f"from {len(tps_list)} samples, {skipped} skipped)"
    )


if __name__ == "__main__":
    main()
