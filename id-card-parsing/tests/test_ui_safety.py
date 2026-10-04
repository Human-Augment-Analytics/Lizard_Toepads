"""Tk regression checks; skip on hosts without a GUI display."""
import threading
import time
import tkinter as tk
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from annotator.backend import AnnotatorBackend
from annotator.config import AnnotatorConfig
from annotator.obb import OBB
from annotator.ui.app import App


def wait(app, condition):
    deadline = time.monotonic() + 4
    while not condition() and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    assert condition()


@pytest.fixture
def app(tmp_path, monkeypatch):
    data = tmp_path/'data'
    data.mkdir()
    backend = AnnotatorBackend(AnnotatorConfig(data_dir=data, out_dir=tmp_path/'annotations',
                                             template_path=tmp_path/'missing.jpg', target_resolution=256))
    for number in range(1, 5):
        image_id = f'{number:04}'
        Image.new('RGB', (100, 160), 'white').save(data/f'{image_id}.jpg')
        preview, _ = backend.open(image_id)
        backend.export(image_id, preview, OBB(100, 120, 60, 100, .1))
    monkeypatch.setattr('annotator.ui.app.messagebox.askokcancel', lambda *a, **k: True)
    try:
        window = App(backend)
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    window.withdraw()
    wait(window, lambda: window._current_id == '0001')
    yield window
    window._close()


def test_edits_survive_navigation_and_failed_save(app, monkeypatch):
    edited = OBB(120, 140, 50, 90, .3)
    app._canvas.set_obb(edited)
    app._select_index(1)
    wait(app, lambda: app._current_id == '0002')
    app._select_index(0)
    wait(app, lambda: app._current_id == '0001')
    assert app._canvas.get_obb() == edited
    before = (app.backend.config.out_dir/'0001.json').read_bytes()
    def fail(*args, **kwargs):
        raise OSError('Disk full')
    monkeypatch.setattr(app.backend, 'export', fail)
    app._save()
    assert 'Save failed' in app._status.get()
    assert app._canvas.get_obb() == edited
    assert (app.backend.config.out_dir/'0001.json').read_bytes() == before


def test_rapid_navigation_cancels_queued_images(app, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = app.backend.open
    calls = []
    def slow(image_id):
        calls.append(image_id)
        if image_id == '0002':
            started.set()
            assert release.wait(4)
        return original(image_id)
    monkeypatch.setattr(app.backend, 'open', slow)
    try:
        app._select_index(1)
        wait(app, started.is_set)
        before = app._canvas.get_obb().corners().copy()
        app._canvas._on_press(SimpleNamespace(x=10, y=10))
        assert (app._canvas.get_obb().corners() == before).all()
        app._select_index(2)
        app._select_index(3)
        release.set()
        wait(app, lambda: app._current_id == '0004')
        assert '0003' not in calls
        assert not app._loading_overlay.winfo_manager()
        assert app._canvas._editing_enabled
    finally:
        release.set()


def test_failed_load_restores_selection_and_keeps_edits(app, monkeypatch):
    edited = OBB(120, 140, 50, 90, .3)
    app._canvas.set_obb(edited)
    def fail(image_id):
        raise OSError('Unreadable image')
    monkeypatch.setattr(app.backend, 'open', fail)
    app._select_index(1)
    wait(app, lambda: app._loading_id is None)
    assert app._current_id == '0001'
    assert app._listbox.curselection() == (0,)
    assert app._canvas.get_obb() == edited
    assert 'Load failed' in app._status.get()


def test_unreadable_annotation_still_allows_manual_save(app):
    path = app.backend.config.out_dir/'0002.json'
    path.write_text('broken JSON')
    app._select_index(1)
    wait(app, lambda: app._current_id == '0002')
    assert 'saved annotation unreadable' in app._status.get()
    assert path.read_text() == 'broken JSON'
    app._canvas.set_obb(OBB(100, 120, 60, 100, 0))
    app._save()
    assert app.backend.existing_record('0002') is not None


def test_close_can_be_cancelled_when_edits_are_unsaved(app, monkeypatch):
    app._canvas.set_obb(OBB(120, 140, 50, 90, .3))
    monkeypatch.setattr('annotator.ui.app.messagebox.askokcancel', lambda *a, **k: False)
    app._close()
    assert not app._closed.is_set()
    assert app.winfo_exists()
    monkeypatch.setattr('annotator.ui.app.messagebox.askokcancel', lambda *a, **k: True)


def test_close_during_load_discards_late_result(app, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = app.backend.open
    def slow(image_id):
        started.set()
        assert release.wait(4)
        return original(image_id)
    monkeypatch.setattr(app.backend, 'open', slow)
    try:
        app._select_index(1)
        wait(app, started.is_set)
        future = app._load_future
        app._close()
        release.set()
        future.result(timeout=3)
        assert app._closed.is_set()
        assert app._load_results.empty()
    finally:
        release.set()


def test_close_during_bulk_does_not_write_current_detection(app):
    path = app.backend.config.out_dir/'0002.json'
    path.unlink()
    started, release = threading.Event(), threading.Event()
    def match(preview):
        started.set()
        assert release.wait(4)
        return SimpleNamespace(obb=OBB(100, 120, 60, 100, 0))
    app._template_matcher = SimpleNamespace(match=match)
    app._template_initialized = True
    try:
        app._open_bulk_dialog()
        app._range_start.set('2')
        app._range_end.set('2')
        app._start_bulk()
        wait(app, started.is_set)
        future = app._bulk_future
        app._close()
        release.set()
        future.result(timeout=3)
        assert not path.exists()
    finally:
        release.set()


def test_small_window_can_scroll_to_sidebar_controls(app):
    app.geometry('800x400')
    app.deiconify()
    app.update()
    assert app._sidebar.bbox('all')[3] > app._sidebar.winfo_height()
    app._sidebar.yview_moveto(1)
    assert app._sidebar.yview()[0] > 0
    assert app._save_btn.winfo_exists() and app._reset_btn.winfo_exists()


def test_close_during_startup_has_no_late_widget_updates(tmp_path, monkeypatch):
    config = AnnotatorConfig(data_dir=tmp_path, out_dir=tmp_path, target_resolution=256)
    backend = AnnotatorBackend(config)
    started, release = threading.Event(), threading.Event()
    def slow(config):
        started.set()
        assert release.wait(4)
        return backend
    monkeypatch.setattr('annotator.backend.AnnotatorBackend', slow)
    try:
        window = App()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    window.withdraw()
    try:
        wait(window, started.is_set)
        window._close()
        release.set()
        window._startup_future.result(timeout=3)
        assert window._closed.is_set()
    finally:
        release.set()
        window._close()


def _isolated_tk(test):
    # Aqua Tk can hang when many root interpreters are created/destroyed in one
    # process. Run each case like a real launch, with a single root interpreter.
    if os.environ.get('ANNOTATOR_GUI_TEST_CHILD') == '1':
        return test
    def isolated():
        environment = dict(os.environ, ANNOTATOR_GUI_TEST_CHILD='1')
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', f'{Path(__file__).resolve()}::{test.__name__}',
             '-q', '--tb=short'],
            env=environment, capture_output=True, text=True, timeout=30,
        )
        if '1 skipped' in result.stdout:
            pytest.skip('Tk display unavailable')
        assert result.returncode == 0, result.stdout+result.stderr
    isolated.__name__ = test.__name__
    return isolated


for _name, _test in list(globals().items()):
    if _name.startswith('test_') and callable(_test):
        globals()[_name] = _isolated_tk(_test)
