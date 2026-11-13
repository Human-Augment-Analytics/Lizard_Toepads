from pathlib import Path

import pytest

from card_pipeline.config import PipelineConfig


def _config(**overrides):
    base = dict(weights="w.pt", template_path="t.json", data_dir="data", output_dir="out",
                template_image="template.jpg")
    base.update(overrides)
    return PipelineConfig(**base)


def test_defaults_and_path_coercion():
    config = _config()
    assert config.confidence == 0.25
    assert config.canvas_size is None
    assert config.device == "cpu"
    assert config.replace is False
    assert isinstance(config.weights, Path)
    assert isinstance(config.output_dir, Path)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_confidence_must_be_in_unit_interval(value):
    with pytest.raises(ValueError):
        _config(confidence=value)


def test_device_must_be_cpu():
    with pytest.raises(ValueError):
        _config(device="cuda")


@pytest.mark.parametrize("value", [0, 1, -5])
def test_canvas_override_must_be_at_least_two(value):
    with pytest.raises(ValueError):
        _config(canvas_size=value)


def test_canvas_override_accepted():
    assert _config(canvas_size=1024).canvas_size == 1024


def test_rules_path_coerced_when_given():
    assert _config(rules_path="rules.json").rules_path == Path("rules.json")


def test_detect_and_decode_size_defaults():
    config = _config()
    assert config.detect_size == 1280      # matches the detector's training imgsz
    assert config.decode_size == 4096


def test_alignment_requires_a_template_image():
    with pytest.raises(ValueError):
        _config(template_image=None)
    assert _config(template_image=None, align=False).align is False
    assert isinstance(_config().template_image, Path)


@pytest.mark.parametrize("overrides", [
    dict(detect_size=16), dict(detect_size=9000),
    dict(decode_size=50), dict(decode_size=9000),
    dict(detect_size=2048, decode_size=1024),   # decode must cover the detect size
])
def test_bad_detect_or_decode_sizes_rejected(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)
