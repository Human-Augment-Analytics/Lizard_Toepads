import itertools

import cv2
import numpy as np
import pytest

from card_pipeline.corners import canonicalize_corners


def _rect(cx, cy, w, h, degrees):
    """Corners TL,TR,BR,BL of a w x h rectangle rotated clockwise on screen."""
    angle = np.deg2rad(degrees)
    local = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return local @ rot.T + np.array([cx, cy])


def _area(points):
    return cv2.contourArea(np.asarray(points, np.float32), oriented=True)


def test_already_canonical_is_unchanged():
    quad = _rect(100, 80, 60, 40, 0)
    assert np.allclose(canonicalize_corners(quad), quad)


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_any_permutation_maps_to_same_result(order):
    quad = _rect(100, 80, 60, 40, 0)
    result = canonicalize_corners(quad[list(order)])
    assert np.allclose(result, quad)


@pytest.mark.parametrize("degrees", [-30, -10, 5, 15, 30, 40])
def test_mildly_rotated_card_keeps_physical_top_edge(degrees):
    quad = _rect(200, 150, 120, 80, degrees)
    shuffled = quad[[2, 0, 3, 1]]
    result = canonicalize_corners(shuffled)
    assert np.allclose(result, quad)
    assert _area(result) > 0


def test_positive_oriented_area_for_reflected_winding():
    quad = _rect(100, 80, 60, 40, 10)[::-1]  # opposite winding
    result = canonicalize_corners(quad)
    assert _area(result) > 0


def test_top_edge_is_longer_horizontal_side_for_landscape_card():
    quad = _rect(100, 80, 90, 50, 70)  # landscape card turned steeply
    result = canonicalize_corners(quad[[1, 3, 0, 2]])
    top = np.linalg.norm(result[1] - result[0])
    side = np.linalg.norm(result[3] - result[0])
    assert top > side


def test_output_accepted_by_rectify():
    from PIL import Image
    from template_builder.normalize import rectify_to_square
    quad = _rect(300, 300, 200, 120, 20)[[3, 1, 0, 2]]
    image = Image.new("RGB", (600, 600), (255, 255, 255))
    rectify_to_square(image, canonicalize_corners(quad), 128)  # must not raise


@pytest.mark.parametrize("bad", [
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [1, 1], [2, 2], [3, 3]],
    [[0, 0], [1, 0], [np.nan, 1], [0, 1]],
])
def test_degenerate_quads_raise(bad):
    with pytest.raises(ValueError):
        canonicalize_corners(np.array(bad, dtype=float))


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        canonicalize_corners(np.zeros((3, 2)))


@pytest.mark.parametrize("degrees", [-30, -5, 0, 10, 35])
def test_portrait_aspect_puts_short_side_on_top(degrees):
    quad = _rect(300, 300, 60, 100, degrees)          # portrait card
    result = canonicalize_corners(quad[[3, 1, 0, 2]], aspect=0.6)
    assert np.allclose(result, quad)


def test_portrait_card_lying_sideways_is_stood_up():
    # A portrait card scanned on its side appears landscape on screen.
    quad = _rect(300, 300, 60, 100, 88)
    result = canonicalize_corners(quad, aspect=0.6)
    top = np.linalg.norm(result[1] - result[0])
    side = np.linalg.norm(result[3] - result[0])
    assert top < side and _area(result) > 0


def test_landscape_aspect_matches_default():
    quad = _rect(100, 80, 90, 50, 20)[[2, 3, 0, 1]]
    assert np.allclose(canonicalize_corners(quad, aspect=1.7), canonicalize_corners(quad))


def test_non_convex_quad_raises():
    with pytest.raises(ValueError):
        canonicalize_corners(np.array([[0, 0], [10, 0], [2, 2], [0, 10]], float))
