"""Device selection should depend on runtime capability, not platform names."""

import sys
from types import SimpleNamespace

import pytest

from card_ocr.models import resolve_ocr_device
from card_pipeline.detector import resolve_detector_device


@pytest.mark.parametrize(("cuda", "mps", "expected"), [
    (True, True, "cuda:0"),
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_yolo_auto_preference(monkeypatch, cuda, mps, expected):
    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resolve_detector_device() == expected
    assert resolve_detector_device("cpu") == "cpu"
    if not mps:
        with pytest.raises(ValueError, match="not available"):
            resolve_detector_device("mps")


@pytest.mark.parametrize(("compiled", "count", "expected"), [
    (True, 1, "gpu:0"),
    (True, 0, "cpu"),
    (False, 1, "cpu"),
])
def test_paddle_auto_preference(monkeypatch, compiled, count, expected):
    fake = SimpleNamespace(
        is_compiled_with_cuda=lambda: compiled,
        device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: count)),
    )
    monkeypatch.setitem(sys.modules, "paddle", fake)
    assert resolve_ocr_device() == expected
    assert resolve_ocr_device("cpu") == "cpu"
    if expected == "cpu":
        with pytest.raises(ValueError, match="not available"):
            resolve_ocr_device("gpu")
