"""Compute per-landmark mean shape from the cephalometric training split.

Reads the "tps" tensor from every ``.pt`` file in the cephalometric
``train/`` directory and computes the element-wise mean across all samples,
producing a (19, 2) float32 tensor saved as a ``.pt`` file.

Unlike the WFLW utility, which reads a split JSON, the cephalometric split is
a directory of preprocessed ``.pt`` files, so this utility globs ``*.pt`` files
from ``train_dir`` directly. No masking is applied; all 19 landmarks are always
present.

Usage:
    python -m landmarking.datasets.cephalometric.compute_mean_shape \\
        --train-dir <data_dir>/train \\
        --output mean_shapes/mean_shape_cephalometric.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def compute_mean_shape(train_dir: str, output_path: str) -> torch.Tensor:
    """Compute mean shape from cephalometric training ``.pt`` files.

    Args:
        train_dir: Path to the cephalometric ``train/`` directory holding
            preprocessed ``.pt`` files, each with a "tps" (19, 2) tensor.
        output_path: Path to save the mean shape ``.pt`` file.

    Returns:
        (19, 2) mean shape tensor (float32).

    Raises:
        ValueError: If no valid "tps" tensors are found in ``train_dir``.
    """
    train_paths = sorted(str(p) for p in Path(train_dir).glob("*.pt"))
    if not train_paths:
        raise ValueError(f"No '.pt' files found in train directory {train_dir}")

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
        raise ValueError(f"No valid 'tps' tensors found in {train_dir}")

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
        description="Compute per-landmark mean shape from cephalometric train dir"
    )
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if not Path(args.train_dir).exists():
        print(f"ERROR: train directory not found: {args.train_dir}", file=sys.stderr)
        sys.exit(1)

    compute_mean_shape(args.train_dir, args.output)


if __name__ == "__main__":
    main()
