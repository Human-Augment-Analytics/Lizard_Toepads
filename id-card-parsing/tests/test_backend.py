import numpy as np
from PIL import Image

from annotator.backend import AnnotatorBackend
from annotator.config import AnnotatorConfig
from annotator.geometry import apply_affine
from annotator.obb import OBB


def _make_jpeg(path, w, h):
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    Image.fromarray(np.stack([arr, arr, arr], -1), "RGB").save(path, "JPEG", quality=70)


def _backend(tmp_path, target=256):
    data = tmp_path / "data"
    data.mkdir()
    _make_jpeg(data / "0002.jpg", 4000, 2000)
    cfg = AnnotatorConfig(
        data_dir=data, out_dir=tmp_path / "annotations", target_resolution=target
    )
    return AnnotatorBackend(cfg)


def test_image_ids(tmp_path):
    be = _backend(tmp_path)
    assert be.image_ids() == ["0002"]


def test_open_returns_preview_and_transform(tmp_path):
    be = _backend(tmp_path, target=256)
    preview, m = be.open("0002")
    assert preview.preview.size == (256, 256)
    assert m.shape == (2, 3)


def test_export_reconstructs_original_space(tmp_path):
    be = _backend(tmp_path, target=256)
    preview, _ = be.open("0002")
    box = OBB(cx=128.0, cy=128.0, w=80.0, h=40.0, angle=0.2)
    path = be.export("0002", preview, box)
    assert path.exists()

    rec = be.existing_record("0002")
    assert rec is not None
    m_inv = np.array(rec.transform_inverse)
    orig = apply_affine(m_inv, np.array(rec.obb_corners))
    # reconstructed corners fall within the original 4000x2000 image
    assert np.all(orig[:, 0] >= 0) and np.all(orig[:, 0] <= 4000)
    assert np.all(orig[:, 1] >= 0) and np.all(orig[:, 1] <= 2000)


def test_export_clamps_negative_and_overflow(tmp_path):
    be = _backend(tmp_path, target=256)
    preview, _ = be.open("0002")
    # Axis-aligned box straddling the top-left corner: spans x [-50, 50],
    # y [-30, 70]; after clamping it must sit in [0, 50] x [0, 70].
    box = OBB(cx=0.0, cy=20.0, w=100.0, h=100.0, angle=0.0)
    be.export("0002", preview, box)
    rec = be.existing_record("0002")
    corners = np.array(rec.obb_corners)
    assert corners.min() >= 0.0
    assert corners.max() <= 256.0
    # exact clamp for an axis-aligned box
    assert np.isclose(corners[:, 0].min(), 0.0)
    assert np.isclose(corners[:, 0].max(), 50.0)
    assert np.isclose(corners[:, 1].min(), 0.0)
    assert np.isclose(corners[:, 1].max(), 70.0)
    # xywhr derived from the clamped corners stays consistent
    cx, cy, w, h, _ = rec.obb_xywhr
    assert np.isclose(cx, 25.0) and np.isclose(cy, 35.0)
    assert np.isclose(w, 50.0) and np.isclose(h, 70.0)


def test_export_in_bounds_box_unchanged(tmp_path):
    be = _backend(tmp_path, target=256)
    preview, _ = be.open("0002")
    box = OBB(cx=128.0, cy=128.0, w=80.0, h=40.0, angle=0.0)
    be.export("0002", preview, box)
    rec = be.existing_record("0002")
    # a fully in-bounds box is untouched by the clamp
    assert np.allclose(rec.obb_corners, box.corners())


def test_export_overwrites_existing(tmp_path):
    be = _backend(tmp_path, target=256)
    preview, _ = be.open("0002")
    be.export("0002", preview, OBB(100.0, 100.0, 20.0, 20.0, 0.0))
    be.export("0002", preview, OBB(150.0, 150.0, 40.0, 30.0, 0.1))
    rec = be.existing_record("0002")
    assert rec.obb_xywhr[0] == 150.0
