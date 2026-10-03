import numpy as np

from annotator.geometry import apply_affine, build_transform, invert_affine


def test_forward_inverse_round_trip():
    m = build_transform(0.05, (12.0, 7.5))
    m_inv = invert_affine(m)
    pts = np.array([[0.0, 0.0], [1000.0, 2000.0], [19999.0, 10599.0]])
    back = apply_affine(m_inv, apply_affine(m, pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_known_point_mapping():
    # scale 0.5, offset (10, 20): (100, 200) -> (60, 120)
    m = build_transform(0.5, (10.0, 20.0))
    out = apply_affine(m, np.array([[100.0, 200.0]]))
    assert np.allclose(out, [[60.0, 120.0]])


def test_non_square_letterbox_offsets():
    # Wide image fit into a square: no x pad, positive y pad.
    orig_w, orig_h, target = 2000, 1000, 1024
    scale = target / orig_w  # width is the limiting dimension
    thumb_h = orig_h * scale
    oy = (target - thumb_h) / 2
    m = build_transform(scale, (0.0, oy))
    # top-left original maps to the top of the letterbox band
    tl = apply_affine(m, np.array([[0.0, 0.0]]))[0]
    assert np.isclose(tl[0], 0.0)
    assert np.isclose(tl[1], oy)
    # bottom-right original maps to within the canvas
    br = apply_affine(m, np.array([[orig_w, orig_h]]))[0]
    assert np.isclose(br[0], target)
    assert np.isclose(br[1], target - oy)
