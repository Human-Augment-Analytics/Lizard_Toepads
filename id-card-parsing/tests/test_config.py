import pytest

from template_builder.config import TemplateBuilderConfig


def test_defaults_are_valid():
    config = TemplateBuilderConfig()
    assert config.canvas_size == 512
    assert config.pad_mode == "solid"
    assert config.fill == (255, 255, 255)
    assert config.preview_resolution > 0
    # Defaults resolve relative to id-card-parsing/.
    assert config.output_path.name.endswith(".json")


def test_canvas_size_must_be_at_least_two():
    with pytest.raises(ValueError):
        TemplateBuilderConfig(canvas_size=1)


def test_unknown_pad_mode_rejected():
    with pytest.raises(ValueError):
        TemplateBuilderConfig(pad_mode="mirror")


def test_fill_out_of_range_rejected():
    with pytest.raises(ValueError):
        TemplateBuilderConfig(fill=(255, 300, 0))
    with pytest.raises(ValueError):
        TemplateBuilderConfig(fill=(0, 0))  # wrong length


def test_preview_resolution_must_be_positive():
    with pytest.raises(ValueError):
        TemplateBuilderConfig(preview_resolution=0)
