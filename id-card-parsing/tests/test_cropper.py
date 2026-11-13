import numpy as np
import pytest
from PIL import Image

from card_pipeline.cropper import crop_region, crop_template
from template_builder.template_record import FieldSpec, TemplateRecord


def _card(size=200):
    """Card whose four quadrants have distinct solid colors."""
    pixels = np.zeros((size, size, 3), np.uint8)
    half = size // 2
    pixels[:half, :half] = (255, 0, 0)       # top-left red
    pixels[:half, half:] = (0, 255, 0)       # top-right green
    pixels[half:, :half] = (0, 0, 255)       # bottom-left blue
    pixels[half:, half:] = (255, 255, 0)     # bottom-right yellow
    return Image.fromarray(pixels)


def test_crop_matches_expected_pixels():
    crop = crop_region(_card(200), (0.0, 0.0, 0.5, 0.5))
    assert crop.size == (100, 100)
    assert set(map(tuple, np.asarray(crop).reshape(-1, 3))) == {(255, 0, 0)}


def test_crop_bottom_right_quadrant():
    crop = crop_region(_card(200), (0.5, 0.5, 1.0, 1.0))
    assert crop.size == (100, 100)
    assert set(map(tuple, np.asarray(crop).reshape(-1, 3))) == {(255, 255, 0)}


def test_pixel_box_rounds_outward_like_card_ocr_field():
    # card_ocr.fields.Field.pixel_box floors left/top and ceils right/bottom.
    from card_ocr.fields import Field
    box = (0.123, 0.217, 0.377, 0.481)
    expected = Field("x", box).pixel_box((333, 333))
    crop = crop_region(Image.new("RGB", (333, 333)), box)
    assert crop.size == (expected[2] - expected[0], expected[3] - expected[1])


def test_sliver_box_keeps_at_least_one_pixel():
    crop = crop_region(Image.new("RGB", (10, 10)), (0.50, 0.50, 0.51, 0.51))
    assert crop.size[0] >= 1 and crop.size[1] >= 1


@pytest.mark.parametrize("box", [(0.5, 0.5, 0.5, 0.9), (0.6, 0.1, 0.4, 0.2), (-0.1, 0, 0.5, 0.5),
                                 (0, 0, 1.2, 0.5), (0.1, 0.1, 0.2)])
def test_invalid_boxes_raise(box):
    with pytest.raises(ValueError):
        crop_region(_card(), box)


def _record(*specs):
    return TemplateRecord(canvas_size=512, pad_mode="solid", fill=(255, 255, 255),
                          card_obb_corners=[[0, 0], [1, 0], [1, 1], [0, 1]], fields=list(specs))


def test_crop_template_preserves_order_and_includes_checkboxes():
    record = _record(
        FieldSpec("b_field", (0.5, 0.0, 1.0, 0.5), "text"),
        FieldSpec("a_box", (0.0, 0.5, 0.5, 1.0), crop_type="checkbox"),
        FieldSpec("c_field", (0.0, 0.0, 0.5, 0.5), "text"),
    )
    results = crop_template(_card(), record)
    assert [spec.name for spec, _crop, _err in results] == ["b_field", "a_box", "c_field"]
    assert all(crop is not None and err is None for _s, crop, err in results)


def test_crop_template_isolates_a_bad_region(monkeypatch):
    import card_pipeline.cropper as cropper

    real = cropper.crop_region

    def flaky(card, box):
        if box[0] == 0.25:
            raise ValueError("forced failure")
        return real(card, box)

    monkeypatch.setattr(cropper, "crop_region", flaky)
    record = _record(
        FieldSpec("first", (0.0, 0.0, 0.2, 0.2), "text"),
        FieldSpec("broken", (0.25, 0.0, 0.5, 0.2), "text"),
        FieldSpec("last", (0.6, 0.0, 0.9, 0.2), "text"),
    )
    results = cropper.crop_template(_card(), record)
    assert [(s.name, c is None, e) for s, c, e in results] == [
        ("first", False, None), ("broken", True, "forced failure"), ("last", False, None)]
