"""YOLO-OBB card detection behind a small protocol.

``YoloObbDetector`` is the only place that touches Ultralytics. Result parsing
and best-detection selection are pure functions so they are tested without
loading weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image


class WeightsLoadError(RuntimeError):
    """Raised when detector weights are missing or cannot be loaded."""


@dataclass(frozen=True)
class Detection:
    """One card detection: raw-order corners in scan pixels, score, and area."""

    corners: np.ndarray
    confidence: float
    area: float


class Detector(Protocol):
    def detect(self, image: Image.Image) -> list[Detection]: ...


def _numpy(value) -> np.ndarray:
    """Convert numpy arrays or torch-like tensors to a float numpy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def detections_from_result(result) -> list[Detection]:
    """Read corners and confidence from one Ultralytics OBB result."""
    obb = getattr(result, "obb", None)
    if obb is None:
        return []
    corners = _numpy(obb.xyxyxyxy).reshape(-1, 4, 2)
    confidences = _numpy(obb.conf).reshape(-1)
    return [Detection(corners=quad, confidence=float(score),
                      area=abs(float(cv2.contourArea(quad.astype(np.float32)))))
            for quad, score in zip(corners, confidences)]


def select_best(detections: list[Detection], confidence: float) -> Detection | None:
    """Pick the highest-confidence detection at or above the threshold.

    Ties are broken by larger area. Returns None when nothing qualifies.
    """
    kept = [d for d in detections if d.confidence >= confidence]
    return max(kept, key=lambda d: (d.confidence, d.area), default=None)


class YoloObbDetector:
    """Run a trained Ultralytics YOLO-OBB model on CPU."""

    def __init__(self, weights: Path, *, device: str = "cpu", imgsz: int | None = None,
                 confidence: float = 0.05):
        weights = Path(weights)
        if not weights.is_file():
            raise WeightsLoadError(f"Detector weights not found: {weights}")
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(weights))
        except Exception as exc:
            raise WeightsLoadError(f"Could not load detector weights {weights}: {exc}") from exc
        self.device, self.imgsz, self._floor = device, imgsz, confidence

    def detect(self, image: Image.Image) -> list[Detection]:
        """Return every detection above a low floor; callers apply their threshold."""
        options = dict(device=self.device, conf=self._floor, verbose=False)
        if self.imgsz:
            options["imgsz"] = self.imgsz
        # Ultralytics treats numpy arrays as BGR, so convert from RGB.
        frame = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        detections: list[Detection] = []
        for result in self._model.predict(frame, **options):
            detections.extend(detections_from_result(result))
        return detections
