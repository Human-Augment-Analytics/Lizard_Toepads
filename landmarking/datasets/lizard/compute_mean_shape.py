"""Compute per-landmark mean shape for the Lizard dataset.

Reads 'tps' tensors from .pt files, normalizes to [0,1], and computes
the element-wise mean across all samples.

Usage:
    python -m landmarking.datasets.lizard.compute_mean_shape \
        --data-dir /path/to/Lizard_data/lizard/train \
        --output /path/to/mean_shape_lizard.pt \
        --input-size 512
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def compute_mean_shape(data_dir: str, output_path: str, input_size: int = 512) -> torch.Tensor:
    """Compute mean shape from a directory of .pt files.

    Normalizes pixel coordinates by input_size to produce [0,1] coords.

    Args:
        data_dir: Directory containing .pt files with 'tps' key.
        output_path: Path to save the mean shape .pt file.
        input_size: Canvas size for normalization (default 512).

    Returns:
        (num_landmarks, 2) mean shape tensor in [0, 1].
    """
    data_path = Path(data_dir)
    pt_files = sorted(data_path.glob("*.pt"))
    if not pt_files:
        raise ValueError(f"No .pt files found in {data_dir}")

    logging.info(f"Computing mean shape from {len(pt_files)} files in {data_dir}...")

    tps_list = []
    skipped = 0

    for pt_path in pt_files:
        try:
            data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except Exception as e:
            logging.warning(f"Failed to load {pt_path}: {e}")
            skipped += 1
            continue

        tps = data.get("tps")
        if tps is None:
            logging.warning(f"Missing 'tps' key in {pt_path}")
            skipped += 1
            continue

        # Normalize to [0, 1]
        tps_norm = tps.float() / input_size
        tps_list.append(tps_norm)

    if not tps_list:
        raise ValueError("No valid 'tps' tensors found")

    stacked = torch.stack(tps_list, dim=0)
    mean_shape = stacked.mean(dim=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_shape, str(output_path))

    logging.info(
        f"Mean shape saved to {output_path} "
        f"(shape: {mean_shape.shape}, from {len(tps_list)} samples, {skipped} skipped)"
    )
    logging.info(f"Mean shape values:\n{mean_shape}")
    return mean_shape


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-landmark mean shape for Lizard dataset"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing training .pt files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for mean_shape.pt")
    parser.add_argument("--input-size", type=int, default=512,
                        help="Canvas size for normalization (default 512)")
    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"ERROR: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    compute_mean_shape(args.data_dir, args.output, args.input_size)


if __name__ == "__main__":
    main()
