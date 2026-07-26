"""CLI tool for running preprocessing for a given dataset.

Usage:
    python -m landmarking.scripts.preprocess --config path/to/config.json
    python -m landmarking.scripts.preprocess --dataset lizard --data-root /path/to/data
    python -m landmarking.scripts.preprocess --dataset wflw --data-root /path/to/data
"""

import argparse
import logging
import sys

from ..config.schema import LandmarkingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run preprocessing for a specified dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON configuration file (overrides other arguments).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["lizard", "wflw"],
        default=None,
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for preprocessed files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point for preprocessing CLI."""
    args = parse_args(argv)

    # Build config from file or arguments
    if args.config:
        config = LandmarkingConfig.from_json(args.config)
    elif args.dataset:
        config_data = {"dataset": {"name": args.dataset}}
        if args.data_root:
            config_data["paths"] = {"data_root": args.data_root}
        if args.output_dir:
            config_data.setdefault("paths", {})["output_root"] = args.output_dir
        config = LandmarkingConfig.from_dict(config_data)
    else:
        logger.error("Must provide either --config or --dataset")
        sys.exit(1)

    config.resolve_paths()

    dataset_name = config.dataset.name
    logger.info(f"Preprocessing dataset: {dataset_name}")
    logger.info(f"Data directory: {config.dataset.data_dir}")

    # Dispatch to dataset-specific preprocessing
    if dataset_name == "lizard":
        from ..datasets.lizard.preprocess import run_preprocessing

        run_preprocessing(config.to_dict())
    elif dataset_name == "wflw":
        from ..datasets.wflw.preprocess import preprocess_wflw

        data_dir = config.dataset.data_dir
        ann_dir = data_dir + "/WFLW_annotations/list_98pt_rect_attr_train_test"
        image_root = data_dir + "/WFLW_images"
        output_dir = data_dir + "/pt_crops"

        # Preprocess train split
        logger.info("Preprocessing WFLW train split...")
        preprocess_wflw(
            annotation_file=ann_dir + "/list_98pt_rect_attr_train.txt",
            image_root=image_root,
            output_dir=output_dir + "/train",
        )

        # Preprocess test split
        logger.info("Preprocessing WFLW test split...")
        preprocess_wflw(
            annotation_file=ann_dir + "/list_98pt_rect_attr_test.txt",
            image_root=image_root,
            output_dir=output_dir + "/test",
        )
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        sys.exit(1)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
