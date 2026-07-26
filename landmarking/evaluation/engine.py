"""Unified evaluation engine for landmark detection.

Dispatches to dataset-specific metric functions based on configuration.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from . import metrics_lizard
from . import metrics_wflw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def evaluate_lizard(predictions: list, ground_truths: list, metadata: list) -> dict:
    """Evaluate Lizard predictions with pixel error and mm conversion.

    Args:
        predictions: List of (N, 2) prediction arrays in pixel space.
        ground_truths: List of (N, 2) GT arrays in pixel space.
        metadata: List of metadata dicts (with ruler_px, etc.).

    Returns:
        Dict with mean_px_error, per_landmark_px, mean_mm (if ruler available).
    """
    errors = []
    errors_mm = []
    per_landmark = [[] for _ in range(9)]

    for pred, gt, meta in zip(predictions, ground_truths, metadata):
        if pred is None:
            continue
        px_err = metrics_lizard.compute_pixel_error(pred, gt)
        errors.append(px_err.mean())
        for lm in range(min(len(px_err), 9)):
            per_landmark[lm].append(px_err[lm])

        ruler_px = meta.get("ruler_px", 0.0)
        if hasattr(ruler_px, "item"):
            ruler_px = ruler_px.item()
        if ruler_px > 0:
            mm_err = metrics_lizard.pixel_to_mm(px_err, ruler_px)
            errors_mm.append(mm_err.mean())

    result = {
        "mean_px_error": float(np.mean(errors)) if errors else None,
        "median_px_error": float(np.median(errors)) if errors else None,
        "per_landmark_px": [
            float(np.mean(e)) if e else None for e in per_landmark
        ],
        "n_evaluated": len(errors),
    }

    if errors_mm:
        result["mean_mm"] = float(np.mean(errors_mm))
        result["median_mm"] = float(np.median(errors_mm))

    return result


def evaluate_wflw(predictions: list, ground_truths: list, metadata: list) -> dict:
    """Evaluate WFLW predictions with NME, FR, AUC.

    Args:
        predictions: List of (98, 2) prediction arrays in pixel space.
        ground_truths: List of (98, 2) GT arrays in pixel space.
        metadata: List of metadata dicts (with attrs).

    Returns:
        Dict with nme, fr, auc for full set and per-attribute subsets.
    """
    attr_names = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
    nme_buckets = {name: [] for name in ["full"] + attr_names}

    for pred, gt, meta in zip(predictions, ground_truths, metadata):
        if pred is None:
            continue
        nme = metrics_wflw.compute_nme(pred, gt)
        if nme is None:
            continue
        nme_buckets["full"].append(nme)

        attrs = meta.get("attrs", None)
        if attrs is not None:
            if hasattr(attrs, "numpy"):
                attrs = attrs.numpy()
            for i, attr_name in enumerate(attr_names):
                if i < len(attrs) and attrs[i] == 1:
                    nme_buckets[attr_name].append(nme)

    subset_keys = ["full"] + attr_names
    result = {
        "nme": {},
        "fr": {},
        "auc": {},
        "counts": {},
    }

    for k in subset_keys:
        nmes = nme_buckets[k]
        result["nme"][k] = float(np.mean(nmes)) if nmes else None
        result["fr"][k] = metrics_wflw.compute_fr(nmes)
        result["auc"][k] = metrics_wflw.compute_auc(nmes)
        result["counts"][k] = len(nmes)

    return result


# Metric dispatch registry
METRIC_REGISTRY: Dict[str, Callable] = {
    "lizard": evaluate_lizard,
    "wflw": evaluate_wflw,
}


class EvaluationEngine:
    """Unified evaluation that dispatches to dataset-specific metrics.

    Args:
        dataset_name: Name of the dataset ("lizard" or "wflw").
    """

    def __init__(self, dataset_name: str):
        if dataset_name not in METRIC_REGISTRY:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(METRIC_REGISTRY.keys())}"
            )
        self.dataset_name = dataset_name
        self.evaluate_fn = METRIC_REGISTRY[dataset_name]

    def evaluate(
        self, predictions: list, ground_truths: list, metadata: list
    ) -> dict:
        """Run evaluation using the dataset-specific metric function.

        Args:
            predictions: List of prediction arrays.
            ground_truths: List of GT arrays.
            metadata: List of metadata dicts.

        Returns:
            Dict of evaluation results.
        """
        return self.evaluate_fn(predictions, ground_truths, metadata)

    def save_results(self, results: dict, output_path: str) -> None:
        """Save evaluation results to JSON.

        Args:
            results: Results dict from evaluate().
            output_path: Path for output JSON file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")
