import numpy as np
import pytest
from PIL import Image

from annotator.image_loader import ImageLoadError, ImageLoader


def _make_jpeg(path, w, h):
    # A cheap gradient so JPEG compresses well and writes fast.
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    rgb = np.stack([arr, arr, arr], axis=-1)
    Image.fromarray(rgb, "RGB").save(path, "JPEG", quality=70)


def test_list_ids_sorted_and_case_insensitive(tmp_path):
    _make_jpeg(tmp_path / "0003.jpg", 32, 16)
    _make_jpeg(tmp_path / "0002.JPG", 32, 16)
    (tmp_path / "notes.txt").write_text("ignore me")
    loader = ImageLoader(target_resolution=64)
    assert loader.list_ids(tmp_path) == ["0002", "0003"]


def test_list_ids_missing_dir_returns_empty(tmp_path):
    loader = ImageLoader(target_resolution=64)
    assert loader.list_ids(tmp_path / "nope") == []


def test_load_fits_target_and_preserves_aspect(tmp_path):
    # Wide 4000x1000 image into a 256 target.
    src = tmp_path / "0002.jpg"
    _make_jpeg(src, 4000, 1000)
    loader = ImageLoader(target_resolution=256)
    pv = loader.load(src)

    assert pv.preview.size == (256, 256)          # square letterbox canvas
    assert pv.orig_size == (4000, 1000)           # from header
    assert pv.scale == pytest.approx(256 / 4000, rel=1e-3)
    # wide image: no horizontal pad, vertical pad present
    ox, oy = pv.offset
    assert ox == 0
    assert oy > 0
    # letterbox is symmetric
    assert oy == pytest.approx((256 - 1000 * pv.scale) / 2, abs=1.0)


def test_load_applies_draft_downscale(tmp_path):
    # The decoded preview must be far smaller than the source, proving we did
    # not materialize the full-resolution buffer.
    src = tmp_path / "0002.jpg"
    _make_jpeg(src, 4096, 2048)
    loader = ImageLoader(target_resolution=128)
    pv = loader.load(src)
    assert max(pv.preview.size) == 128
    assert pv.scale < 0.05


def test_load_bad_file_raises(tmp_path):
    bad = tmp_path / "0002.jpg"
    bad.write_bytes(b"not a real jpeg")
    loader = ImageLoader(target_resolution=64)
    with pytest.raises(ImageLoadError):
        loader.load(bad)
