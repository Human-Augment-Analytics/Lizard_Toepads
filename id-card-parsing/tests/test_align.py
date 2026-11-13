import numpy as np
import pytest
from PIL import Image, ImageDraw

from card_pipeline.align import CardAligner, reference_from_template_image

SIZE = 512


def _form(seed=0):
    """A square canvas with a portrait 'printed form' of random strokes and boxes."""
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    left, right = 96, 416                      # portrait card centered on the canvas
    draw.rectangle((left, 0, right, SIZE - 1), outline=(0, 0, 0), width=3)
    for _ in range(140):
        x = int(rng.integers(left + 10, right - 40))
        y = int(rng.integers(70, SIZE - 30))
        w, h = int(rng.integers(8, 40)), int(rng.integers(6, 22))
        if rng.random() < 0.5:
            draw.rectangle((x, y, x + w, y + h), outline=(0, 0, 0), width=2)
        else:
            draw.line((x, y, x + w, y + h), fill=(0, 0, 0), width=2)
    return image


def _difference(a, b):
    """Mean difference after a small blur, so sub-pixel resampling is not penalized."""
    from PIL import ImageFilter
    a, b = (image.convert("L").filter(ImageFilter.GaussianBlur(3)) for image in (a, b))
    return float(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).mean())


def test_upside_down_card_is_turned_upright():
    reference = _form()
    flipped = reference.rotate(180)
    result = CardAligner(reference).align(flipped)
    assert result.rotation_degrees == pytest.approx(180, abs=5) or \
        result.rotation_degrees == pytest.approx(-180, abs=5)
    assert _difference(result.image, reference) < _difference(flipped, reference) / 3


def test_upright_card_is_left_essentially_unchanged():
    reference = _form()
    result = CardAligner(reference).align(reference.copy())
    assert abs(result.rotation_degrees) < 2
    assert _difference(result.image, reference) < 3


def test_small_shift_is_corrected():
    reference = _form()
    shifted = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    shifted.paste(reference, (6, -5))
    result = CardAligner(reference).align(shifted)
    assert _difference(result.image, reference) < _difference(shifted, reference)


def test_output_size_and_diagnostics():
    reference = _form()
    result = CardAligner(reference).align(reference.rotate(180))
    assert result.image.size == (SIZE, SIZE)
    assert result.diagnostics["inliers"] >= 20


def test_featureless_card_raises_value_error():
    reference = _form()
    blank = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    with pytest.raises(ValueError):
        CardAligner(reference).align(blank)


def test_size_mismatch_raises():
    with pytest.raises(ValueError):
        CardAligner(_form()).align(Image.new("RGB", (256, 256)))


def test_reference_from_template_image_is_square_and_padded(tmp_path):
    path = tmp_path / "template.jpg"
    Image.new("RGB", (400, 640), (90, 90, 90)).save(path, "JPEG")   # portrait card crop
    reference = reference_from_template_image(path, 256, fill=(255, 255, 255))
    assert reference.size == (256, 256)
    pixels = np.asarray(reference)
    assert tuple(pixels[128, 2]) == (255, 255, 255)        # side padding
    assert pixels[128, 128].mean() < 120                   # card content
