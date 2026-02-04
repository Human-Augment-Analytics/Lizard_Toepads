#!/usr/bin/env python3
"""
Unified training workflow for ml-morph PyTorch keypoints.
Handles preprocessing (TPS → XML), training, and evaluation in one command.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete training workflow")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip preprocessing if XML files already exist")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation after training")
    return parser.parse_args()


def run_preprocessing(config: dict, base_dir: Path):
    """Run TPS to XML preprocessing (no dlib required)."""
    prep_config = config.get("preprocessing", {})

    tps_file = prep_config.get("tps_file", "consolidated_toe.tps")
    image_dir = prep_config.get("image_dir", "train")
    train_ratio = prep_config.get("train_ratio", 0.8)

    print(f"\n{'='*60}")
    print("STEP 1: Preprocessing (TPS → XML)")
    print(f"{'='*60}")
    print(f"  TPS file: {tps_file}")
    print(f"  Images: {image_dir}")
    print(f"  Split ratio: {train_ratio}")

    # Build preprocessing command (using simple TPS converter)
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "preprocessing" / "tps_to_xml.py"),
        "-t", tps_file,
        "-i", image_dir,
        "--train-ratio", str(train_ratio)
    ]

    # Run preprocessing
    result = subprocess.run(cmd, cwd=base_dir)
    if result.returncode != 0:
        print(f"\n❌ Preprocessing failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n✅ Preprocessing complete!")


def run_training(config_path: Path, base_dir: Path):
    """Run PyTorch Lightning training."""
    print(f"\n{'='*60}")
    print("STEP 2: Training (PyTorch Keypoint Regression)")
    print(f"{'='*60}")

    # Build training command
    cmd = [
        sys.executable,
        str(base_dir / "pytorch_keypoint" / "train_keypoints.py"),
        "--config", str(config_path)
    ]

    # Run training
    result = subprocess.run(cmd, cwd=base_dir)
    if result.returncode != 0:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n✅ Training complete!")

    # Return checkpoint directory for evaluation
    trainer_config = yaml.safe_load(open(config_path))['trainer']
    checkpoint_dir = Path(base_dir) / trainer_config.get('default_root_dir', 'runs/keypoints/toe')
    return checkpoint_dir


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best checkpoint from training."""
    # Look for checkpoints in lightning_logs
    pattern = str(checkpoint_dir / "lightning_logs" / "version_*" / "checkpoints" / "*.ckpt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Find checkpoint with lowest validation loss (in filename)
    best_ckpt = min(checkpoints, key=lambda x: float(x.split('val_loss=')[-1].replace('.ckpt', '')))
    return Path(best_ckpt)


def run_evaluation(config_path: Path, checkpoint_path: Path, base_dir: Path, config: dict):
    """Run evaluation on training and test sets."""
    print(f"\n{'='*60}")
    print("STEP 3: Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path.name}")

    data_config = config.get('data', {})
    train_xml = data_config.get('train_xml', 'train.xml')
    test_xml = data_config.get('val_xml', 'test.xml')

    # Run predictions on training set
    print("\n  Generating predictions on training set...")
    cmd_train = [
        sys.executable,
        str(base_dir / "pytorch_keypoint" / "predict_keypoints.py"),
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--xml", train_xml,
        "--output", "predictions_train.csv"
    ]
    result = subprocess.run(cmd_train, cwd=base_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Training set prediction failed: {result.stderr}")
        return

    # Run predictions on test set
    print("  Generating predictions on test set...")
    cmd_test = [
        sys.executable,
        str(base_dir / "pytorch_keypoint" / "predict_keypoints.py"),
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--xml", test_xml,
        "--output", "predictions_test.csv"
    ]
    result = subprocess.run(cmd_test, cwd=base_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Test set prediction failed: {result.stderr}")
        return

    # Evaluate training set
    print("\n  📊 Training Set Performance:")
    cmd_eval_train = [
        sys.executable,
        str(base_dir / "pytorch_keypoint" / "evaluate_predictions.py"),
        "--csv", "predictions_train.csv",
        "--xml", train_xml,
        "--root", "."
    ]
    subprocess.run(cmd_eval_train, cwd=base_dir)

    # Evaluate test set
    print("\n  📊 Test Set Performance:")
    cmd_eval_test = [
        sys.executable,
        str(base_dir / "pytorch_keypoint" / "evaluate_predictions.py"),
        "--csv", "predictions_test.csv",
        "--xml", test_xml,
        "--root", "."
    ]
    subprocess.run(cmd_eval_test, cwd=base_dir)

    print("\n✅ Evaluation complete!")


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent.parent  # Assumes config is in ml-morph/configs/

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"ML-Morph PyTorch Training Workflow")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Base directory: {base_dir}")

    # Step 1: Preprocessing (unless skipped)
    if not args.skip_preprocessing:
        run_preprocessing(config, base_dir)
    else:
        print("\n⏭️  Skipping preprocessing (using existing XML files)")

    # Step 2: Training
    checkpoint_dir = run_training(config_path, base_dir)

    # Step 3: Evaluation (unless skipped)
    if not args.skip_evaluation:
        try:
            best_checkpoint = find_best_checkpoint(checkpoint_dir)
            run_evaluation(config_path, best_checkpoint, base_dir, config)
        except Exception as e:
            print(f"\n⚠️  Evaluation failed: {e}")
            print("  You can run evaluation manually with pytorch_keypoint/predict_keypoints.py")
    else:
        print("\n⏭️  Skipping evaluation")

    print(f"\n{'='*60}")
    print("✅ Workflow complete!")
    print(f"{'='*60}")
    if not args.skip_evaluation:
        print("\nResults:")
        print("  - Predictions saved: predictions_train.csv, predictions_test.csv")
        print("  - Checkpoint: runs/keypoints/")
    else:
        print("\nNext steps:")
        print("  - Check training results in: runs/keypoints/")
        print("  - Run predictions with: pytorch_keypoint/predict_keypoints.py")


if __name__ == "__main__":
    main()
