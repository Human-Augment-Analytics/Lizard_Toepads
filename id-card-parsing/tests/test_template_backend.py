import numpy as np
import pytest
from PIL import Image

from annotator.image_loader import ImageLoadError
from annotator.obb import OBB
from template_builder.backend import TemplateBuilderBackend
from template_builder.config import TemplateBuilderConfig
from template_builder.template_record import FieldSpec


def _make_card_jpeg(path, width=1200, height=800, color=(40, 90, 160)):
    Image.new("RGB", (width, height), color).save(path, "JPEG", quality=95)
    return path


def _backend(tmp_path, **overrides):
    config = TemplateBuilderConfig(
        output_path=tmp_path / "field_template.json",
        preview_resolution=512,
        **overrides,
    )
    return TemplateBuilderBackend(config)


def test_load_image_returns_preview_and_dims(tmp_path):
    card = _make_card_jpeg(tmp_path / "card.jpg", 1200, 800)
    backend = _backend(tmp_path)
    preview = backend.load_image(card)
    assert preview.orig_size == (1200, 800)
    assert preview.preview.size == (512, 512)  # letterboxed to preview_resolution


def test_load_image_decode_failure_raises(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a real jpeg")
    backend = _backend(tmp_path)
    with pytest.raises(ImageLoadError):
        backend.load_image(bad)


def test_obb_to_original_maps_back_through_transform(tmp_path):
    card = _make_card_jpeg(tmp_path / "card.jpg", 1200, 800)
    backend = _backend(tmp_path)
    backend.load_image(card)
    # An axis-aligned OBB in preview space; map corners to original pixels.
    obb = OBB(cx=256.0, cy=256.0, w=200.0, h=120.0, angle=0.0)
    corners_original = backend.obb_to_original(obb)
    assert corners_original.shape == (4, 2)
    # Re-apply the forward transform; corners should return to preview space.
    m = backend.preview_transform()
    from annotator.geometry import apply_affine
    back = apply_affine(m, corners_original)
    assert np.allclose(back, obb.corners(), atol=1e-6)
    # Original corners lie within the original image bounds.
    assert corners_original[:, 0].min() >= -1
    assert corners_original[:, 0].max() <= 1201
    assert corners_original[:, 1].max() <= 801


def test_normalize_produces_canvas_sized_image(tmp_path):
    card = _make_card_jpeg(tmp_path / "card.jpg", 1200, 800)
    backend = _backend(tmp_path, canvas_size=256)
    backend.load_image(card)
    obb = OBB(cx=256.0, cy=256.0, w=300.0, h=180.0, angle=0.0)
    corners = backend.obb_to_original(obb)
    result = backend.normalize(corners)
    assert result.image.size == (256, 256)


def test_set_canvas_size_changes_normalization(tmp_path):
    card = _make_card_jpeg(tmp_path / "card.jpg", 1200, 800)
    backend = _backend(tmp_path, canvas_size=256)
    backend.load_image(card)
    obb = OBB(cx=256.0, cy=256.0, w=300.0, h=180.0, angle=0.0)
    corners = backend.obb_to_original(obb)
    backend.set_canvas_size(512)
    result = backend.normalize(corners)
    assert result.image.size == (512, 512)
    assert backend.config.canvas_size == 512


def test_build_save_and_reload_round_trip(tmp_path):
    card = _make_card_jpeg(tmp_path / "card.jpg", 1200, 800)
    backend = _backend(tmp_path)
    backend.load_image(card)
    obb = OBB(cx=256.0, cy=256.0, w=300.0, h=180.0, angle=0.0)
    corners = backend.obb_to_original(obb)
    fields = [
        FieldSpec("day", (0.2, 0.2, 0.4, 0.3), "integer", "0123456789"),
        FieldSpec("perch", (0.5, 0.5, 0.8, 0.6), "text", None),
    ]
    record = backend.build_record(corners, fields)
    assert backend.existing_record() is None  # nothing saved yet
    path = backend.save(record)
    assert path == backend.config.output_path
    reloaded = backend.existing_record()
    assert reloaded == record


def test_build_record_uses_current_config(tmp_path):
    card = _make_card_jpeg(tmp_path / "card.jpg", 1200, 800)
    backend = _backend(tmp_path, canvas_size=256)
    backend.load_image(card)
    corners = backend.obb_to_original(OBB(256.0, 256.0, 300.0, 180.0, 0.0))
    record = backend.build_record(corners, [FieldSpec("day", (0.1, 0.1, 0.2, 0.2), "integer")])
    assert record.canvas_size == 256
    assert record.pad_mode == "solid"
    assert record.fill == (255, 255, 255)
