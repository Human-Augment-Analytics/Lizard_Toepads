#!/usr/bin/env python3
"""
Standalone evaluation script for ml-morph PyTorch keypoints.
Compares training and test set performance.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained keypoint model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", help="Path to checkpoint file (optional, auto-finds best if not provided)")
    parser.add_argument("--train-only", action="store_true", help="Only evaluate training set")
    parser.add_argument("--test-only", action="store_true", help="Only evaluate test set")
    return parser.parse_args()


def find_best_checkpoint(config_path: Path) -> Path:
    """Find the best checkpoint from training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_dir = config_path.parent.parent
    trainer_config = config.get('trainer', {})
    checkpoint_dir = Path(base_dir) / trainer_config.get('default_root_dir', 'runs/keypoints/toe')

    # Look for checkpoints in lightning_logs
    pattern = str(checkpoint_dir / "lightning_logs" / "version_*" / "checkpoints" / "*.ckpt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Find checkpoint with lowest validation loss (in filename)
    best_ckpt = min(checkpoints, key=lambda x: float(x.split('val_loss=')[-1].replace('.ckpt', '')))
    return Path(best_ckpt)


def run_predictions(config_path: Path, checkpoint_path: Path, xml_file: str, output_csv: str, base_dir: Path):
    """Run predictions on a dataset."""
    cmd = [
        sys.executable,
        str(base_dir / "pytorch_keypoint" / "predict_keypoints.py"),
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--xml", xml_file,
        "--output", output_csv
    ]
    result = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Prediction failed: {result.stderr}")
        sys.exit(1)


def run_evaluation(csv_file: str, xml_file: str, base_dir: Path):
    """Evaluate predictions against ground truth."""
    cmd = [
        sys.executable,
        str(base_dir / "pytorch_keypoint" / "evaluate_predictions.py"),
        "--csv", csv_file,
        "--xml", xml_file,
        "--root", "."
    ]
    subprocess.run(cmd, cwd=base_dir)


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    train_xml = data_config.get('train_xml', 'train.xml')
    test_xml = data_config.get('val_xml', 'test.xml')

    # Find or use provided checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        print("Finding best checkpoint...")
        checkpoint_path = find_best_checkpoint(config_path)

    print(f"\n{'='*60}")
    print("ML-Morph Evaluation")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Base directory: {base_dir}")

    # Evaluate training set
    if not args.test_only:
        print(f"\n{'='*60}")
        print("Training Set Evaluation")
        print(f"{'='*60}")
        print(f"Generating predictions on {train_xml}...")
        run_predictions(config_path, checkpoint_path, train_xml, "predictions_train.csv", base_dir)

        print("\n📊 Training Set Performance:")
        run_evaluation("predictions_train.csv", train_xml, base_dir)

    # Evaluate test set
    if not args.train_only:
        print(f"\n{'='*60}")
        print("Test Set Evaluation")
        print(f"{'='*60}")
        print(f"Generating predictions on {test_xml}...")
        run_predictions(config_path, checkpoint_path, test_xml, "predictions_test.csv", base_dir)

        print("\n📊 Test Set Performance:")
        run_evaluation("predictions_test.csv", test_xml, base_dir)

    print(f"\n{'='*60}")
    print("✅ Evaluation complete!")
    print(f"{'='*60}")
    print("\nPredictions saved:")
    if not args.test_only:
        print("  - predictions_train.csv")
    if not args.train_only:
        print("  - predictions_test.csv")


if __name__ == "__main__":
    main()
