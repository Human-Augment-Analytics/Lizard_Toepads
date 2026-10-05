import numpy as np
import pytest
import threading

from annotator.geometry import apply_affine, build_transform, invert_affine
from annotator.obb import OBB
from annotator.record import AnnotationRecord, load_record, save_record


def _sample_record():
    scale, offset = 0.05, (0.0, 245.0)
    m = build_transform(scale, offset)
    m_inv = invert_affine(m)
    box = OBB(cx=512.0, cy=400.0, w=300.0, h=120.0, angle=0.3)
    return AnnotationRecord(
        image_id="0002",
        target_resolution=1024,
        transform=m.tolist(),
        transform_inverse=m_inv.tolist(),
        obb_xywhr=[box.cx, box.cy, box.w, box.h, box.angle],
        obb_corners=box.corners().tolist(),
    )


def test_json_round_trip():
    rec = _sample_record()
    back = AnnotationRecord.from_json(rec.to_json())
    assert back == rec


def test_save_creates_dir_and_file(tmp_path):
    rec = _sample_record()
    out = tmp_path / "annotations"
    path = save_record(rec, out)
    assert path.exists()
    assert path.name == "0002.json"
    assert load_record("0002", out) == rec


def test_load_missing_returns_none(tmp_path):
    assert load_record("9999", tmp_path) is None


def test_inverse_corners_land_in_original_bounds():
    rec = _sample_record()
    orig_w, orig_h = 20400, 10644
    m_inv = np.array(rec.transform_inverse)
    orig_corners = apply_affine(m_inv, np.array(rec.obb_corners))
    assert np.all(orig_corners[:, 0] >= 0) and np.all(orig_corners[:, 0] <= orig_w)
    assert np.all(orig_corners[:, 1] >= 0) and np.all(orig_corners[:, 1] <= orig_h)


def test_failed_atomic_save_preserves_existing_record(tmp_path, monkeypatch):
    rec = _sample_record()
    path = save_record(rec, tmp_path)
    before = path.read_bytes()
    rec.obb_xywhr[0] += 10
    def fail(*args):
        raise OSError("Disk write failed")
    monkeypatch.setattr('annotator.record.os.replace', fail)
    with pytest.raises(OSError):
        save_record(rec, tmp_path)
    assert path.read_bytes() == before
    assert list(tmp_path.glob('*.tmp')) == []


def test_exclusive_save_never_overwrites_existing_file(tmp_path):
    rec = _sample_record()
    path = save_record(rec, tmp_path)
    before = path.read_bytes()
    rec.obb_xywhr[0] += 10
    with pytest.raises(FileExistsError):
        save_record(rec, tmp_path, overwrite=False)
    assert path.read_bytes() == before
    assert list(tmp_path.glob('*.tmp')) == []


def test_cancellation_before_commit_leaves_no_partial_file(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        save_record(_sample_record(), tmp_path, cancel=cancel)
    assert list(tmp_path.iterdir()) == []


def test_cancellation_during_write_preserves_previous_record(tmp_path, monkeypatch):
    rec = _sample_record()
    path = save_record(rec, tmp_path)
    before = path.read_bytes()
    cancel = threading.Event()
    monkeypatch.setattr('annotator.record.os.fsync', lambda fd: cancel.set())
    rec.obb_xywhr[0] += 10
    with pytest.raises(InterruptedError):
        save_record(rec, tmp_path, cancel=cancel)
    assert path.read_bytes() == before
    assert list(tmp_path.glob('*.tmp')) == []
