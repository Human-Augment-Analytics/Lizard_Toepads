import pytest
from PIL import Image, ImageDraw

from card_ocr.checkboxes import inspect_checkbox


def _box(paper, *, shift=0):
    image = Image.new("RGB", (40, 42), paper)
    draw = ImageDraw.Draw(image)
    for points in ([(13, 8), (9, 8), (9, 34), (13, 34)],
                   [(27, 8), (31, 8), (31, 34), (27, 34)]):
        draw.line([(x + shift, y) for x, y in points], fill=(30, 30, 30), width=2)
    return image


def test_empty_checkbox_with_tinted_paper_and_shifted_print():
    reference = _box((240, 240, 240))
    observed = _box((202, 196, 176), shift=2)
    assert inspect_checkbox(observed, reference)["checked"] is False


@pytest.mark.parametrize("color", [(25, 25, 25), (215, 60, 75), (60, 120, 230),
                                   (70, 180, 100), (250, 225, 50)])
def test_marks_in_black_and_color(color):
    reference = _box((240, 240, 240))
    observed = reference.copy()
    ImageDraw.Draw(observed).line([(11, 20), (18, 29), (34, 10)], fill=color, width=3)
    assert inspect_checkbox(observed, reference)["checked"] is True


def test_small_extra_stroke_is_uncertain():
    reference = _box((240, 240, 240))
    observed = reference.copy()
    ImageDraw.Draw(observed).line([(19, 18), (19, 24)], fill=(30, 30, 30))
    result = inspect_checkbox(observed, reference)
    assert result["state"] == "uncertain"
    assert result["checked"] is None
