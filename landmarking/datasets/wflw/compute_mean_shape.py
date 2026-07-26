"""Compute per-landmark mean shape from a training split.

Reads the "tps" tensor from each training .pt file and computes
the element-wise mean across all samples, producing a (N, 2) float32
tensor saved as a .pt file.

Usage:
    python -m landmarking.datasets.wflw.compute_mean_shape \\
        --split splits/wflw_1.0_seed42.json \\
        --output mean_shapes/mean_shape_wflw.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def compute_mean_shape(split_path: str, output_path: str) -> torch.Tensor:
    """Compute mean shape from training split .pt files.

    Args:
        split_path: Path to split JSON with 'train' key.
        output_path: Path to save the mean shape .pt file.

    Returns:
        (num_landmarks, 2) mean shape tensor.
    """
    with open(split_path) as f:
        split_data = json.load(f)

    train_paths = split_data.get("train", [])
    if not train_paths:
        raise ValueError(f"'train' list in {split_path} is empty")

    logging.info(f"Computing mean shape from {len(train_paths)} training samples...")

    tps_list = []
    skipped = 0
    expected_shape = None

    for pt_path in train_paths:
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
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
            logging.warning(f"Unexpected shape {tps.shape} in {pt_path}")
            skipped += 1
            continue

        tps_list.append(tps.float())

    if not tps_list:
        raise ValueError("No valid 'tps' tensors found in training split")

    stacked = torch.stack(tps_list, dim=0)
    mean_shape = stacked.mean(dim=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_shape, str(output_path))

    logging.info(
        f"Mean shape saved to {output_path} "
        f"(shape: {mean_shape.shape}, from {len(tps_list)} samples, {skipped} skipped)"
    )
    return mean_shape


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-landmark mean shape from training split"
    )
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if not Path(args.split).exists():
        print(f"ERROR: split file not found: {args.split}", file=sys.stderr)
        sys.exit(1)

    compute_mean_shape(args.split, args.output)


if __name__ == "__main__":
    main()
