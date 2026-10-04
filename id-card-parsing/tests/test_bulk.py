import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from annotator.backend import AnnotatorBackend
from annotator.bulk import ids_in_range, save_template_range
from annotator.config import AnnotatorConfig
from annotator.obb import OBB


def test_range_is_inclusive_numeric_and_reports_gaps():
    selected, missing = ids_in_range(['0004', '0001', '0003', 'bask_0048', '0038 (2)'], '1', '4')
    assert selected == ['0001', '0003', '0004']
    assert missing == 1


@pytest.mark.parametrize('start,end', [('', '4'), ('4', '1'), ('a', '4')])
def test_invalid_range(start, end):
    with pytest.raises(ValueError):
        ids_in_range(['0001'], start, end)


def make_backend(tmp_path, colors):
    data = tmp_path / 'data'
    data.mkdir()
    for index, color in enumerate(colors, 1):
        Image.new('RGB', (100, 160), (color,) * 3).save(data / f'{index:04}.jpg')
    return AnnotatorBackend(AnnotatorConfig(data_dir=data, out_dir=tmp_path/'annotations', target_resolution=256))


def test_bulk_detects_each_image_and_preserves_existing_files(tmp_path):
    backend = make_backend(tmp_path, [100, 120, 140, 160, 180])
    backend.config.out_dir.mkdir()
    existing = backend.config.out_dir / '0001.json'
    existing.write_text('existing annotation untouched')
    calls = []

    class Matcher:
        def match(self, preview):
            color = preview.getpixel((128, 128))[0]
            calls.append(color)
            if color == 140:
                return None
            if color == 160:
                raise ValueError('bad image')
            return SimpleNamespace(obb=OBB(color, 120, 30, 80, .1))

    updates = []
    summary = save_template_range(backend, Matcher(), backend.image_ids(), threading.Event(), updates.append)
    assert (summary.saved, summary.existing, summary.unmatched, summary.failed) == (2, 1, 1, 1)
    assert summary.completed == 5 and len(updates) == 5
    assert updates[0].completed == 1  # Progress snapshots are independent.
    assert len(updates[0].outcomes) == 1
    assert summary.outcomes[0] == ('0001', 'Skipped', 'Annotation already exists')
    assert summary.outcomes[2] == ('0003', 'Skipped', 'No reliable detection')
    assert summary.outcomes[3] == ('0004', 'Failed', 'bad image')
    assert existing.read_text() == 'existing annotation untouched'
    assert backend.existing_record('0002').obb_xywhr[0] == 120
    assert backend.existing_record('0005').obb_xywhr[0] == 180
    assert not (backend.config.out_dir/'0003.json').exists()
    assert not (backend.config.out_dir/'0004.json').exists()


def test_cancel_stops_before_next_image_and_retains_saved_files(tmp_path):
    backend = make_backend(tmp_path, [100, 120])
    cancel = threading.Event()
    matcher = SimpleNamespace(match=lambda image: SimpleNamespace(obb=OBB(100, 120, 30, 80, 0)))
    summary = save_template_range(backend, matcher, backend.image_ids(), cancel, lambda progress: cancel.set())
    assert summary.cancelled and summary.saved == 1 and summary.completed == 1
    assert backend.existing_record('0001') is not None
    assert not (backend.config.out_dir/'0002.json').exists()


def test_cancel_during_detection_does_not_write_current_image(tmp_path):
    backend = make_backend(tmp_path, [100])
    cancel = threading.Event()

    def match(image):
        cancel.set()
        return SimpleNamespace(obb=OBB(100, 120, 30, 80, 0))

    summary = save_template_range(backend, SimpleNamespace(match=match), backend.image_ids(), cancel, lambda progress: None)
    assert summary.cancelled and summary.saved == 0
    assert not backend.config.out_dir.exists()
