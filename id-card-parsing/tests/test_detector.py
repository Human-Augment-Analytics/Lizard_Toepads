from types import SimpleNamespace

import numpy as np
import pytest

from card_pipeline.detector import (
    Detection,
    WeightsLoadError,
    YoloObbDetector,
    detections_from_result,
    select_best,
)


def _det(conf, w=100, h=60, x=0):
    corners = np.array([[x, 0], [x + w, 0], [x + w, h], [x, h]], float)
    return Detection(corners=corners, confidence=conf, area=float(w * h))


def test_select_best_prefers_highest_confidence():
    best = select_best([_det(0.5), _det(0.9), _det(0.7)], 0.25)
    assert best.confidence == 0.9


def test_select_best_breaks_ties_by_area():
    small, large = _det(0.8, 50, 30), _det(0.8, 200, 120)
    assert select_best([small, large], 0.25) is large


def test_select_best_applies_threshold():
    assert select_best([_det(0.2), _det(0.1)], 0.25) is None
    assert select_best([], 0.25) is None


def test_select_best_includes_detection_at_threshold():
    assert select_best([_det(0.25)], 0.25) is not None


def _result(corners, conf):
    obb = SimpleNamespace(xyxyxyxy=np.asarray(corners, float), conf=np.asarray(conf, float))
    return SimpleNamespace(obb=obb)


def test_detections_from_result_reads_corners_confidence_and_area():
    quad = [[[0, 0], [100, 0], [100, 50], [0, 50]]]
    detections = detections_from_result(_result(quad, [0.9]))
    assert len(detections) == 1
    assert np.allclose(detections[0].corners, quad[0])
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].area == pytest.approx(5000)


def test_detections_from_result_handles_no_boxes_and_missing_obb():
    assert detections_from_result(_result(np.zeros((0, 4, 2)), np.zeros(0))) == []
    assert detections_from_result(SimpleNamespace(obb=None)) == []


def test_detections_from_result_accepts_tensor_like_values():
    class TensorLike:
        def __init__(self, array):
            self._array = np.asarray(array, float)

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    quad = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    obb = SimpleNamespace(xyxyxyxy=TensorLike(quad), conf=TensorLike([0.5]))
    detections = detections_from_result(SimpleNamespace(obb=obb))
    assert detections[0].area == pytest.approx(100)


def test_missing_weights_raise_clear_error(tmp_path):
    with pytest.raises(WeightsLoadError, match="not found"):
        YoloObbDetector(tmp_path / "absent.pt")


def test_unloadable_weights_raise_clear_error(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a model")
    with pytest.raises(WeightsLoadError):
        YoloObbDetector(bad)
