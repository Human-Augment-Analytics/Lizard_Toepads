import pytest
from PIL import Image, ImageDraw

from card_ocr.checkboxes import checkbox_pixel_box, inspect_checkbox


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


@pytest.mark.parametrize("size", [(100, 200), (512, 512), (1024, 1024)])
def test_detection_bounds_match_saved_rectangle_at_each_resolution(size):
    from card_ocr.fields import Field

    box = (0.123, 0.217, 0.377, 0.481)
    assert checkbox_pixel_box(box, size) == Field("alive", box).pixel_box(size)


@pytest.mark.parametrize("box,expected", [
    ((0, 0, .2, .2), (0, 0, 20, 40)),
    ((.8, .8, 1, 1), (80, 160, 100, 200)),
])
def test_detection_bounds_stop_at_card_edges(box, expected):
    assert checkbox_pixel_box(box, (100, 200)) == expected


def test_prepare_run_uses_saved_checkbox_rectangle_with_legacy_layout(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from card_ocr.fields import Field
    from card_ocr.prepare import prepare_run

    reference = Image.new("RGB", (100, 100), "white")
    template = tmp_path / "template.png"
    reference.save(template)
    observed = reference.copy()
    ImageDraw.Draw(observed).line([(30, 30), (30, 69)], fill="black")
    data = tmp_path / "data"
    data.mkdir()
    observed.save(data / "0001.png")
    monkeypatch.setattr("card_ocr.prepare._original_corners", lambda *args: (
        np.array([[0, 0], [99, 0], [99, 99], [0, 99]], float), "annotation"))
    monkeypatch.setattr("card_ocr.prepare.FormAligner", lambda _reference: SimpleNamespace(
        register=lambda _card: SimpleNamespace(card_to_template=np.eye(3), diagnostics={})))
    output = tmp_path / "output"
    result = prepare_run(
        data, tmp_path / "annotations", template, [Field("note", (.1, .1, .2, .2))],
        ["0001"], output, card_width=100, decode_size=100, clean_print=False,
        checkboxes=[Field("alive", (.3, .3, .7, .7), margin=(.1, .1))])
    checkbox = result["checkboxes"][0]
    assert checkbox["card_box"] == [30, 30, 70, 70]
    assert checkbox["checked"] is True
    with Image.open(output / checkbox["crop"]) as crop:
        assert crop.size == (40, 40)
