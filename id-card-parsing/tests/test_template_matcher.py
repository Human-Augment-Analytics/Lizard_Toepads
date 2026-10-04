import cv2
import numpy as np

from annotator.obb import OBB
from annotator.template_matcher import _clip_to_scan, _rectangle_from_corners, _refine_card_edges


def test_refinement_corrects_each_side_independently():
    scene = np.full((400, 300), 90, dtype=np.uint8)
    scene[50:351, 60:241] = 235
    initial = OBB(153, 198, 192, 312, 0)
    refined = _refine_card_edges(scene, initial)
    np.testing.assert_allclose(
        refined.corners(), [[59.5, 49.5], [240.5, 49.5],
                            [240.5, 350.5], [59.5, 350.5]], atol=1)


def test_refinement_ignores_weak_edges_and_printed_text():
    scene = np.full((400, 300), 230, dtype=np.uint8)
    cv2.putText(scene, 'ID', (63, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, 10, 2)
    initial = OBB(150, 200, 180, 300, 0)
    assert _refine_card_edges(scene, initial) == initial


def test_refinement_preserves_off_image_edge():
    scene = np.full((400, 300), 90, dtype=np.uint8)
    scene[50:351, :181] = 235
    initial = OBB(85, 200, 190, 300, 0)
    refined = _refine_card_edges(scene, initial)
    assert abs(refined.corners()[0, 0] + 10) < 1e-6
    assert abs(refined.corners()[1, 0] - 180.5) <= 1


def test_rectangle_fit_preserves_rotated_rectangle():
    original = OBB(150, 200, 180, 300, 0.25)
    fitted = _rectangle_from_corners(original.corners())
    np.testing.assert_allclose(fitted.corners(), original.corners(), atol=1e-6)


def test_bottom_ignores_nearby_black_letterbox_border():
    scene = np.full((400, 300), 210, dtype=np.uint8)
    scene[50:351, 60:241] = 235
    scene[358:] = 0
    initial = OBB(150, 200, 180, 300, 0)
    refined = _refine_card_edges(scene, initial)
    assert abs(refined.corners()[2, 1] - 350.5) <= 1


def test_bottom_prefers_card_edge_over_stronger_nearby_object():
    scene = np.full((400, 300), 210, dtype=np.uint8)
    scene[50:351, 60:241] = 235
    scene[358:380, 50:251] = 10
    initial = OBB(150, 200, 180, 300, 0)
    refined = _refine_card_edges(scene, initial)
    assert abs(refined.corners()[2, 1] - 350.5) <= 1


def test_tilted_card_refines_partially_visible_left_edge():
    scene = np.full((400, 300), 90, dtype=np.uint8)
    card = OBB(94, 200, 180, 300, -0.06)
    cv2.fillConvexPoly(scene, np.round(card.corners()).astype(np.int32), 235)
    u = np.array([np.cos(card.angle), np.sin(card.angle)])
    initial = OBB(card.cx-3.5*u[0], card.cy-3.5*u[1], 187, 300, card.angle)
    refined = _refine_card_edges(scene, initial)
    np.testing.assert_allclose(refined.corners()[[0, 3]],
                               card.corners()[[0, 3]], atol=1.5)


def test_left_edge_moves_out_of_black_strip_to_card_boundary():
    scene = np.full((400, 300), 90, dtype=np.uint8)
    scene[:, :10] = 0
    scene[50:351, 10:241] = 235
    initial = OBB(123, 200, 244, 300, 0)
    refined = _refine_card_edges(scene, initial)
    np.testing.assert_allclose(refined.corners()[[0, 3], 0], 9.5, atol=1)


def test_clipped_card_box_stays_inside_scan_and_preserves_angle():
    scene = np.full((400, 300), 120, dtype=np.uint8)
    scene[:20] = 0
    scene[380:] = 0
    original = OBB(85, 200, 190, 300, -0.025)
    clipped = _clip_to_scan(scene, original)
    assert clipped is not None and clipped.angle == original.angle
    assert clipped.corners()[:, 0].min() >= -1e-6
    np.testing.assert_allclose(clipped.corners()[[1, 2]],
                               original.corners()[[1, 2]], atol=1e-6)


def test_clipping_does_not_change_fully_visible_box():
    scene = np.full((400, 300), 120, dtype=np.uint8)
    original = OBB(150, 200, 180, 300, .05)
    assert _clip_to_scan(scene, original) == original
