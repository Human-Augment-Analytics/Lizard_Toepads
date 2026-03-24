#!/usr/bin/env python3
"""
Classic dlib ml-morph workflow.
Handles preprocessing, detector training, shape predictor training, and inference.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete dlib ml-morph workflow")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip preprocessing if XML files already exist")
    parser.add_argument("--skip-detector", action="store_true",
                       help="Skip detector training if detector already exists")
    parser.add_argument("--skip-shape", action="store_true",
                       help="Skip shape predictor training")
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference step")
    return parser.parse_args()


def run_preprocessing(config: dict, base_dir: Path):
    """Run preprocessing to create train/test split and XML files."""
    prep_config = config.get("preprocessing", {})

    input_dir = prep_config.get("input_dir", "image-examples")
    tps_file = prep_config.get("tps_file")
    train_ratio = prep_config.get("train_ratio", 0.8)

    print(f"\n{'='*60}")
    print("STEP 1: Preprocessing")
    print(f"{'='*60}")
    print(f"  Input directory: {input_dir}")
    print(f"  TPS file: {tps_file}")
    print(f"  Train ratio: {train_ratio}")

    # Build preprocessing command
    cmd = [sys.executable, str(base_dir / "preprocessing.py"), "-i", input_dir]

    if tps_file:
        cmd.extend(["-t", tps_file])

    cmd.extend(["--train-ratio", str(train_ratio)])

    # Run preprocessing
    result = subprocess.run(cmd, cwd=base_dir)
    if result.returncode != 0:
        print(f"\n❌ Preprocessing failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n✅ Preprocessing complete!")


def run_detector_training(config: dict, base_dir: Path):
    """Train object detector using HOG + SVM."""
    detector_config = config.get("detector", {})

    print(f"\n{'='*60}")
    print("STEP 2: Detector Training (HOG + SVM)")
    print(f"{'='*60}")

    # Build detector training command
    cmd = [
        sys.executable,
        str(base_dir / "detector_trainer.py"),
        "-d", detector_config.get("dataset", "train.xml"),
        "-o", detector_config.get("output", "detector.svm"),
        "-n", str(detector_config.get("n_threads", 1))
    ]

    # Add optional test file
    test_file = detector_config.get("test")
    if test_file:
        cmd.extend(["-t", test_file])

    # Add other parameters
    if detector_config.get("symmetrical", False):
        cmd.extend(["-s", "True"])

    cmd.extend(["-e", str(detector_config.get("epsilon", 0.01))])
    cmd.extend(["-c", str(detector_config.get("c_param", 5))])
    cmd.extend(["-u", str(detector_config.get("upsample", 0))])

    window_size = detector_config.get("window_size")
    if window_size:
        cmd.extend(["-w", str(window_size)])

    # Run detector training
    result = subprocess.run(cmd, cwd=base_dir)
    if result.returncode != 0:
        print(f"\n❌ Detector training failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n✅ Detector training complete!")


def run_shape_training(config: dict, base_dir: Path):
    """Train shape predictor using regression trees."""
    shape_config = config.get("shape", {})

    print(f"\n{'='*60}")
    print("STEP 3: Shape Predictor Training")
    print(f"{'='*60}")

    # Build shape training command
    cmd = [
        sys.executable,
        str(base_dir / "shape_trainer.py"),
        "-d", shape_config.get("dataset", "train.xml"),
        "-o", shape_config.get("output", "predictor.dat"),
        "-th", str(shape_config.get("threads", 1))
    ]

    # Add optional test file
    test_file = shape_config.get("test")
    if test_file:
        cmd.extend(["-t", test_file])

    # Add other parameters
    cmd.extend(["-dp", str(shape_config.get("tree_depth", 4))])
    cmd.extend(["-c", str(shape_config.get("cascade_depth", 15))])
    cmd.extend(["-nu", str(shape_config.get("nu", 0.1))])
    cmd.extend(["-os", str(shape_config.get("oversampling", 10))])
    cmd.extend(["-s", str(shape_config.get("test_splits", 20))])
    cmd.extend(["-f", str(shape_config.get("feature_pool_size", 500))])
    cmd.extend(["-n", str(shape_config.get("num_trees", 500))])

    # Run shape training
    result = subprocess.run(cmd, cwd=base_dir)
    if result.returncode != 0:
        print(f"\n❌ Shape predictor training failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n✅ Shape predictor training complete!")


def run_inference(config: dict, base_dir: Path):
    """Run inference on new images."""
    inference_config = config.get("inference", {})

    input_dir = inference_config.get("input_dir", "pred")
    detector = inference_config.get("detector", "detector.svm")
    predictor = inference_config.get("predictor", "predictor.dat")

    print(f"\n{'='*60}")
    print("STEP 4: Inference")
    print(f"{'='*60}")
    print(f"  Input directory: {input_dir}")
    print(f"  Detector: {detector}")
    print(f"  Predictor: {predictor}")

    # Check if input directory exists
    input_path = base_dir / input_dir
    if not input_path.exists():
        print(f"\n⚠️  Input directory '{input_dir}' not found, skipping inference")
        print("  Create the directory and add images to run inference")
        return

    # Build inference command
    cmd = [
        sys.executable,
        str(base_dir / "prediction.py"),
        "-i", input_dir,
        "-d", detector,
        "-p", predictor,
        "-o", inference_config.get("output", "output.xml")
    ]

    # Add optional parameters
    cmd.extend(["-u", str(inference_config.get("upsample_limit", 0))])
    cmd.extend(["-th", str(inference_config.get("threshold", 0))])

    ignore_list = inference_config.get("ignore_list")
    if ignore_list:
        cmd.extend(["-ig", ",".join(map(str, ignore_list))])

    # Run inference
    result = subprocess.run(cmd, cwd=base_dir)
    if result.returncode != 0:
        print(f"\n❌ Inference failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n✅ Inference complete!")


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent.parent  # Assumes config is in ml-morph/configs/

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"ML-Morph Classic (dlib) Workflow")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Base directory: {base_dir}")

    # Step 1: Preprocessing (unless skipped)
    if not args.skip_preprocessing:
        run_preprocessing(config, base_dir)
    else:
        print("\n⏭️  Skipping preprocessing (using existing XML files)")

    # Step 2: Detector training (unless skipped)
    if not args.skip_detector:
        run_detector_training(config, base_dir)
    else:
        print("\n⏭️  Skipping detector training (using existing detector)")

    # Step 3: Shape predictor training (unless skipped)
    if not args.skip_shape:
        run_shape_training(config, base_dir)
    else:
        print("\n⏭️  Skipping shape predictor training (using existing predictor)")

    # Step 4: Inference (unless skipped)
    if not args.skip_inference:
        run_inference(config, base_dir)
    else:
        print("\n⏭️  Skipping inference")

    print(f"\n{'='*60}")
    print("✅ Workflow complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")

    detector_out = config.get("detector", {}).get("output", "detector.svm")
    shape_out = config.get("shape", {}).get("output", "predictor.dat")
    inference_out = config.get("inference", {}).get("output", "output.xml")

    print(f"  - Detector: {detector_out}")
    print(f"  - Shape predictor: {shape_out}")

    inference_dir = base_dir / config.get("inference", {}).get("input_dir", "pred")
    if inference_dir.exists():
        print(f"  - Predictions: {inference_out}")


if __name__ == "__main__":
    main()
