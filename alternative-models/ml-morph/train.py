import sys
import os
import argparse
import subprocess
from pathlib import Path

import dlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ML_MORPH_DIR = Path(__file__).parent.resolve()
SHARED_DATA_DIR = "/storage/ice-shared/cs8903onl/alternative-models/data"


def main():
    ap = argparse.ArgumentParser(description="Train ml-morph shape predictor")
    ap.add_argument("--split", type=str, required=True,
                    help="Path to shared split.json file")
    ap.add_argument("--data-dir", type=str, default=SHARED_DATA_DIR,
                    help="Shared data root directory")
    ap.add_argument("--threads", type=int, default=8,
                    help="number of threads (default: 8)")
    ap.add_argument("--tree-depth", type=int, default=4,
                    help="tree depth (default: 4)")
    ap.add_argument("--cascade-depth", type=int, default=15,
                    help="cascade depth (default: 15)")
    ap.add_argument("--nu", type=float, default=0.1,
                    help="regularization parameter (default: 0.1)")
    ap.add_argument("--oversampling", type=int, default=20,
                    help="oversampling amount (default: 20)")
    ap.add_argument("--test-splits", type=int, default=20,
                    help="number of test splits (default: 20)")
    ap.add_argument("--feature-pool-size", type=int, default=500,
                    help="feature pool size (default: 500)")
    ap.add_argument("--num-trees", type=int, default=500,
                    help="number of trees per cascade level (default: 500)")
    args = ap.parse_args()

    split_path = Path(args.split).resolve()
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        sys.exit(1)

    convert_script = str(ML_MORPH_DIR / "convert_to_xml.py")
    result = subprocess.run(
        [sys.executable, convert_script, "--split", str(split_path), "--data-dir", args.data_dir],
        cwd=str(ML_MORPH_DIR),
    )
    if result.returncode != 0:
        print("ERROR: convert_to_xml.py failed", file=sys.stderr)
        sys.exit(1)

    train_xml = ML_MORPH_DIR / "data" / "train.xml"
    if not train_xml.exists() or train_xml.stat().st_size == 0:
        print("ERROR: train.xml does not exist or is empty after conversion", file=sys.stderr)
        sys.exit(1)

    checkpoints_dir = ML_MORPH_DIR / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dat = str(checkpoints_dir / "ml_morph_best.dat")

    options = dlib.shape_predictor_training_options()
    options.num_trees_per_cascade_level = args.num_trees
    options.nu = args.nu
    options.num_threads = args.threads
    options.tree_depth = args.tree_depth
    options.cascade_depth = args.cascade_depth
    options.feature_pool_size = args.feature_pool_size
    options.num_test_splits = args.test_splits
    options.oversampling_amount = args.oversampling
    options.be_verbose = True

    print(f"Training shape predictor from {train_xml}")
    dlib.train_shape_predictor(str(train_xml), output_dat, options)
    print(f"Training error (mean pixel deviation): {dlib.test_shape_predictor(str(train_xml), output_dat)}")

    val_xml = ML_MORPH_DIR / "data" / "val.xml"
    if val_xml.exists():
        error = dlib.test_shape_predictor(str(val_xml), output_dat)
        print(f"Validation error (mean pixel deviation): {error}")
    else:
        print("WARNING: val.xml not found, skipping validation")

    print(f"Saved predictor to {output_dat}")


if __name__ == "__main__":
    main()
