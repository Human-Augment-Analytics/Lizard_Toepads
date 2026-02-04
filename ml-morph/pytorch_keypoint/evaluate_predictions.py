import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from .dataset import _parse_xml, Annotation
except ImportError:
    module_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(module_root.parent.parent))
    dataset_module = __import__("ml-morph.pytorch_keypoint.dataset", fromlist=["_parse_xml", "Annotation"])  # type: ignore
    _parse_xml = dataset_module._parse_xml  # type: ignore[attr-defined]
    Annotation = dataset_module.Annotation  # type: ignore[attr-defined]


def load_ground_truth(xml_path: Path, root_dir: Path) -> Dict[str, Annotation]:
    annotations = _parse_xml(xml_path, root_dir)
    mapping: Dict[str, Annotation] = {}
    for ann in annotations:
        key = Path(ann.image_path).name
        mapping[key] = ann
    return mapping


def load_predictions(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(header) < 3 or (len(header) - 1) % 2 != 0:
        raise ValueError("CSV header does not match expected keypoint format.")

    num_keypoints = (len(header) - 1) // 2
    image_paths: List[str] = []
    preds = np.zeros((len(rows), num_keypoints, 2), dtype=np.float32)

    for i, row in enumerate(rows):
        image_paths.append(row[0])
        coords = [float(x) for x in row[1:]]
        preds[i] = np.array(coords, dtype=np.float32).reshape(num_keypoints, 2)

    return image_paths, preds


def compute_errors(
    image_names: List[str],
    predictions: np.ndarray,
    ground_truth: Dict[str, Annotation],
) -> Tuple[np.ndarray, List[str]]:
    num_samples, num_keypoints, _ = predictions.shape
    errors = np.zeros((num_samples, num_keypoints), dtype=np.float32)
    missing: List[str] = []

    for idx, image_path in enumerate(image_names):
        key = Path(image_path).name
        ann = ground_truth.get(key)
        if ann is None:
            missing.append(key)
            continue

        gt = ann.keypoints.astype(np.float32)
        pred = predictions[idx]
        if gt.shape[0] != num_keypoints:
            min_len = min(gt.shape[0], num_keypoints)
            gt = gt[:min_len]
            pred = pred[:min_len]

        distances = np.linalg.norm(pred - gt, axis=1)
        errors[idx, : distances.shape[0]] = distances

    return errors, missing


def main():
    parser = argparse.ArgumentParser(description="Evaluate keypoint prediction CSV against ground truth XML.")
    parser.add_argument("--csv", required=True, help="Path to predictions CSV.")
    parser.add_argument("--xml", required=True, help="Ground truth XML file.")
    parser.add_argument("--root", default="ml-morph", help="Root directory that contains images referenced by XML.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    xml_path = Path(args.xml)
    root_dir = Path(args.root)

    image_paths, predictions = load_predictions(csv_path)
    gt_mapping = load_ground_truth(xml_path, root_dir)

    errors, missing = compute_errors(image_paths, predictions, gt_mapping)

    valid_mask = errors.any(axis=1)
    valid_errors = errors[valid_mask]

    overall_mean = float(valid_errors.mean()) if valid_errors.size > 0 else float("nan")
    per_keypoint_mean = valid_errors.mean(axis=0)

    print(f"Evaluated {valid_errors.shape[0]} samples with {errors.shape[1]} keypoints each.")
    print(f"Overall mean Euclidean error: {overall_mean:.4f} pixels")
    print("Per-keypoint mean error:")
    for idx, value in enumerate(per_keypoint_mean):
        print(f"  keypoint {idx}: {value:.4f}")

    if missing:
        print(f"Warning: {len(missing)} predictions had no matching ground truth entries.")


if __name__ == "__main__":
    main()





