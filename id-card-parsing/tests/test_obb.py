import numpy as np

from annotator.obb import OBB


def test_from_corners_round_trip():
    box = OBB(cx=512.0, cy=400.0, w=300.0, h=120.0, angle=0.4)
    rebuilt = OBB.from_corners(box.corners())
    assert np.isclose(rebuilt.cx, box.cx, atol=1e-6)
    assert np.isclose(rebuilt.cy, box.cy, atol=1e-6)
    assert np.isclose(rebuilt.w, box.w, atol=1e-6)
    assert np.isclose(rebuilt.h, box.h, atol=1e-6)
    assert np.isclose(rebuilt.angle, box.angle, atol=1e-6)


def test_axis_aligned_corners():
    box = OBB(cx=100.0, cy=100.0, w=40.0, h=20.0, angle=0.0)
    expected = np.array(
        [[80.0, 90.0], [120.0, 90.0], [120.0, 110.0], [80.0, 110.0]]
    )
    assert np.allclose(box.corners(), expected)


def test_corner_order_stable_under_rotation():
    # 90-degree CCW rotation: the TL corner should move predictably and the
    # ordering (TL,TR,BR,BL) must be preserved.
    box = OBB(cx=0.0, cy=0.0, w=2.0, h=1.0, angle=np.pi / 2)
    c = box.corners()
    # first vertex is the rotated top-left of the local frame
    assert np.allclose(c[0], [0.5, -1.0], atol=1e-6)
    # consecutive corners stay a constant distance apart (closed quad)
    sides = np.linalg.norm(np.diff(np.vstack([c, c[0]]), axis=0), axis=1)
    assert np.allclose(sides, [2.0, 1.0, 2.0, 1.0], atol=1e-6)
