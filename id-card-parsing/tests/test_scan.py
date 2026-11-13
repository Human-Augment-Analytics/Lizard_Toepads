import numpy as np
import pytest
from PIL import Image

from annotator.image_loader import ImageLoadError
from card_pipeline.scan import load_scan


def _write_scan(path, width=2000, height=1000):
    Image.new("RGB", (width, height), (200, 200, 200)).save(path, "JPEG", quality=95)
    return path


def test_preview_is_square_letterboxed_at_detect_size(tmp_path):
    scan = load_scan(_write_scan(tmp_path / "a.jpg"), detect_size=512, decode_size=1024)
    assert scan.preview.size == (512, 512)
    assert scan.original_size == (2000, 1000)


def test_decoded_is_bounded_by_decode_size(tmp_path):
    scan = load_scan(_write_scan(tmp_path / "a.jpg"), detect_size=512, decode_size=1024)
    assert max(scan.decoded.size) <= 1024
    assert scan.decoded.size[0] / scan.decoded.size[1] == pytest.approx(2.0, rel=0.02)


def test_preview_points_map_to_decoded_pixels(tmp_path):
    scan = load_scan(_write_scan(tmp_path / "a.jpg"), detect_size=512, decode_size=1024)
    # 2:1 scan letterboxed into 512: content spans x 0..512, y 128..384.
    top_left = scan.preview_to_decoded([[0, 128]])[0]
    bottom_right = scan.preview_to_decoded([[512, 384]])[0]
    width, height = scan.decoded.size
    assert np.allclose(top_left, [0, 0], atol=2)
    assert np.allclose(bottom_right, [width, height], atol=2)


def test_png_scans_load(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (300, 200), "white").save(path)
    scan = load_scan(path, detect_size=128, decode_size=256)
    assert scan.original_size == (300, 200)


def test_unreadable_scan_raises_load_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(ImageLoadError):
        load_scan(bad, detect_size=128, decode_size=256)
