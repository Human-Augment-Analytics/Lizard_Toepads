"""
Run HRNet heatmap regression training on WFLW.

Parallel to run_wflw.py (which runs HRNet-GCN).
Both scripts accept the same --split argument so they can run simultaneously
on separate HPC instances using the same split files.

Usage:
    # Instance 1 — GCN
    python run_wflw.py --split ./splits/wflw_0.8_seed42.json

    # Instance 2 — HRNet heatmap (same split)
    python run_wflw_heatmap.py --split ./splits/wflw_0.8_seed42.json

    # Override config:
    python run_wflw_heatmap.py \\
        --split ./splits/wflw_0.25_seed42.json \\
        --config ../../alternative-models/hrnet/configs/wflw-config.json
"""
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SCRIPT_DIR    = Path(__file__).parent.resolve()
ALT_MODELS    = SCRIPT_DIR.parent.parent / "alternative-models"
HRNET_DIR     = ALT_MODELS / "hrnet"
TRAIN_SCRIPT  = "train_heatmap_wflw_ref.py"
DEFAULT_CONFIG = str(HRNET_DIR / "configs" / "wflw-config.json")
MODEL_NAME    = "hrnet_heatmap_wflw"


def main():
    parser = argparse.ArgumentParser(
        description="Run HRNet heatmap regression training on WFLW"
    )
    parser.add_argument(
        "--split", type=str, required=True,
        help="Path to WFLW split JSON file (same file used by run_wflw.py)",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help=f"Path to heatmap config JSON (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    split_file = Path(args.split).resolve()
    if not split_file.exists():
        print(f"ERROR: split file not found: {split_file}", file=sys.stderr)
        sys.exit(1)

    config_file = Path(args.config).resolve()
    if not config_file.exists():
        print(f"ERROR: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    logging.info(f"[{MODEL_NAME}] Starting training...")
    logging.info(f"  split:  {split_file}")
    logging.info(f"  config: {config_file}")

    start = time.time()

    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--config", str(config_file),
        "--split",  str(split_file),
    ]

    result = subprocess.run(cmd, cwd=str(HRNET_DIR), check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        logging.error(
            f"[{MODEL_NAME}] Training failed with exit code {result.returncode}"
        )
    else:
        logging.info(f"[{MODEL_NAME}] Training completed in {elapsed:.1f}s")

    print()
    print("=" * 60)
    print(f"{'Model':<30} {'Exit Code':<12} {'Elapsed (s)':<12}")
    print("-" * 60)
    print(f"{MODEL_NAME:<30} {str(result.returncode):<12} {elapsed:<12.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
