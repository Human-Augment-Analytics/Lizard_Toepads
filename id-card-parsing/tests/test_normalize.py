import numpy as np
import pytest
from PIL import Image

from template_builder.normalize import NormalizeResult, rectify_to_square


def _solid(width, height, color):
    return Image.new("RGB", (width, height), color)


def test_axis_aligned_square_pads_to_zero_offset():
    # A square card occupying a square region -> no padding on either axis.
    image = _solid(400, 400, (0, 128, 255))
    corners = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], dtype=float)
    result = rectify_to_square(image, corners, 512)
    assert isinstance(result, NormalizeResult)
    assert result.image.size == (512, 512)
    assert result.pad_offset == (0, 0)


def test_output_is_always_canvas_sized():
    image = _solid(800, 400, (10, 20, 30))  # 2:1 wide card
    corners = np.array([[0, 0], [799, 0], [799, 399], [0, 399]], dtype=float)
    for size in (128, 256, 512):
        result = rectify_to_square(image, corners, size)
        assert result.image.size == (size, size)


def test_wide_card_pads_on_vertical_axis_only():
    # 2:1 card fit into a square: full width, vertical letterbox.
    image = _solid(800, 400, (200, 200, 200))
    corners = np.array([[0, 0], [799, 0], [799, 399], [0, 399]], dtype=float)
    result = rectify_to_square(image, corners, 512)
    ox, oy = result.pad_offset
    assert ox == 0
    assert oy > 0
    nw, nh = result.rect_size_on_canvas
    # Padding accounts for the full canvas on each axis (within rounding).
    assert abs(nw + 2 * ox - 512) <= 1
    assert abs(nh + 2 * oy - 512) <= 1


def test_tall_card_pads_on_horizontal_axis_only():
    image = _solid(400, 800, (200, 200, 200))  # 1:2 tall card
    corners = np.array([[0, 0], [399, 0], [399, 799], [0, 799]], dtype=float)
    result = rectify_to_square(image, corners, 512)
    ox, oy = result.pad_offset
    assert oy == 0
    assert ox > 0


def test_fill_color_used_for_padding():
    image = _solid(800, 400, (0, 0, 0))  # black card
    corners = np.array([[0, 0], [799, 0], [799, 399], [0, 399]], dtype=float)
    result = rectify_to_square(image, corners, 512, fill=(255, 0, 0))
    pixels = np.asarray(result.image)
    # Top-left pixel is in the vertical pad band -> fill color (red).
    assert tuple(pixels[0, 0]) == (255, 0, 0)


def test_rotated_region_lands_upright():
    # Paint a red marker into the TL corner region of a tilted *card*, then
    # confirm rectification brings it to the TL corner of the upright card.
    # The card is a 400x400 square centered at (300, 300) rotated ~20 degrees.
    angle = np.deg2rad(20)
    cx, cy, half = 300.0, 300.0, 200.0
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = local @ rot.T + np.array([cx, cy])

    # Fill a quarter-size patch anchored at the card's TL corner, in card space,
    # mapped into image pixels so it tracks the rotation.
    quarter = np.array([[-half, -half], [0.0, -half], [0.0, 0.0], [-half, 0.0]])
    patch = quarter @ rot.T + np.array([cx, cy])
    image = Image.new("RGB", (600, 600), (255, 255, 255))
    from PIL import ImageDraw
    ImageDraw.Draw(image).polygon([tuple(p) for p in patch], fill=(255, 0, 0))

    result = rectify_to_square(image, corners, 512)
    out = np.asarray(result.image)
    # The TL quadrant of the upright card should be predominantly red; the
    # opposite (BR) quadrant should have essentially none.
    tl = out[:256, :256].reshape(-1, 3)
    br = out[256:, 256:].reshape(-1, 3)
    tl_red = np.mean((tl[:, 0] > 150) & (tl[:, 1] < 100))
    br_red = np.mean((br[:, 0] > 150) & (br[:, 1] < 100))
    assert tl_red > 0.5
    assert br_red < 0.05


def test_degenerate_corners_raise():
    image = _solid(400, 400, (0, 0, 0))
    collapsed = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=float)
    with pytest.raises(ValueError):
        rectify_to_square(image, collapsed, 512)


def test_wrong_corner_count_raises():
    image = _solid(400, 400, (0, 0, 0))
    with pytest.raises(ValueError):
        rectify_to_square(image, np.array([[0, 0], [1, 1], [2, 2]], dtype=float), 512)


def test_canvas_size_must_be_valid():
    image = _solid(400, 400, (0, 0, 0))
    corners = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], dtype=float)
    with pytest.raises(ValueError):
        rectify_to_square(image, corners, 1)
