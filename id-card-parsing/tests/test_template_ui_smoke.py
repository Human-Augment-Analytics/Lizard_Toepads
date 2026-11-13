"""Headless smoke test: drive the app's backend-facing actions without a human.

Tkinter widgets are constructed (requires a display/Tk build) but no event loop
runs; we call the same methods the event handlers call. Skips cleanly if Tk is
unavailable in the environment.
"""

from __future__ import annotations

import tkinter as tk

import numpy as np
import pytest
from PIL import Image

from annotator.obb import OBB
from template_builder.backend import TemplateBuilderBackend
from template_builder.config import TemplateBuilderConfig
from template_builder.template_record import load_record
from template_builder.ui.field_view import FieldBox


@pytest.fixture
def app(tmp_path):
    import time

    from template_builder.ui.app import App
    config = TemplateBuilderConfig(output_path=tmp_path / "field_template.json",
                                   preview_resolution=512)
    instance = None
    # Back-to-back Tk root creation/teardown can transiently fail; retry before
    # deciding Tk is genuinely unavailable.
    for attempt in range(3):
        try:
            instance = App(TemplateBuilderBackend(config))
            break
        except tk.TclError:
            time.sleep(0.2)
    if instance is None:
        pytest.skip("Tk not available in this environment")
    yield instance
    instance.destroy()


def _card(path):
    Image.new("RGB", (1200, 800), (60, 110, 180)).save(path, "JPEG", quality=95)
    return path


def test_full_flow_crop_normalize_field_save(app, tmp_path):
    card = _card(tmp_path / "card.jpg")
    preview = app.backend.load_image(card)
    app.crop_canvas.show_preview(preview.preview)
    # Draw a card OBB programmatically and normalize via the same path the button uses.
    app.crop_canvas.set_obb(OBB(cx=256.0, cy=256.0, w=360.0, h=240.0, angle=0.0))
    app._normalize()
    assert app.field_frame.winfo_ismapped() or app._corners_original is not None

    # Add a named, typed field box and save.
    app.field_canvas.set_boxes([
        FieldBox(0.2, 0.2, 0.5, 0.35, name="day", datatype="integer", dictionary="0123456789"),
    ])
    specs, error = app._collect_fields()
    assert error is None
    assert specs[0].name == "day"
    app._save()

    record = load_record(config_path := app.backend.config.output_path)
    assert record is not None
    assert [f.name for f in record.fields] == ["day"]
    assert record.canvas_size == 512
    assert record.fields[0].datatype == "integer"
    assert record.fields[0].dictionary == "0123456789"


def test_save_blocked_without_valid_names(app, tmp_path):
    card = _card(tmp_path / "card.jpg")
    app.backend.load_image(card)
    app._corners_original = app.backend.obb_to_original(OBB(256.0, 256.0, 360.0, 240.0, 0.0))
    # A box with no name must block saving.
    app.field_canvas.set_boxes([FieldBox(0.2, 0.2, 0.5, 0.35, name="")])
    _specs, error = app._collect_fields()
    assert error is not None
    assert app.backend.existing_record() is None


def test_checkbox_crop_saves_and_routes(app, tmp_path):
    card = _card(tmp_path / "card.jpg")
    app.backend.load_image(card)
    app.crop_canvas.show_preview(app.backend._preview.preview)
    app.crop_canvas.set_obb(OBB(256.0, 256.0, 360.0, 240.0, 0.0))
    app._normalize()
    # One OCR field and one checkbox crop.
    app.field_canvas.set_boxes([
        FieldBox(0.2, 0.2, 0.5, 0.35, name="day", datatype="integer"),
        FieldBox(0.7, 0.1, 0.78, 0.16, name="alive", crop_type="checkbox"),
    ])
    specs, error = app._collect_fields()
    assert error is None
    app._save()

    record = load_record(app.backend.config.output_path)
    assert {f.name: f.crop_type for f in record.fields} == {"day": "field", "alive": "checkbox"}

    # Export routing keeps checkboxes out of the OCR fields layout.
    from template_builder.export_layout import to_checkboxes_layout, to_fields_layout
    assert [f["name"] for f in to_fields_layout(record)["fields"]] == ["day"]
    assert [c["name"] for c in to_checkboxes_layout(record)["checkboxes"]] == ["alive"]


def test_auto_restore_field_boxes_on_open(app, tmp_path):
    # Save a template, then re-open the same card and confirm the fields come
    # back for editing after normalize (not just the card OBB).
    card = _card(tmp_path / "card.jpg")
    app.backend.load_image(card)
    app._corners_original = app.backend.obb_to_original(OBB(256.0, 256.0, 360.0, 240.0, 0.0))
    app.field_canvas.set_boxes([
        FieldBox(0.2, 0.2, 0.5, 0.35, name="day", datatype="integer", dictionary="0123456789"),
        FieldBox(0.7, 0.1, 0.78, 0.16, name="alive", crop_type="checkbox"),
    ])
    specs, _ = app._collect_fields()
    app._save()

    # Re-open the card: existing_record is found, stashed, and applied on normalize.
    app.field_canvas.set_boxes([])  # clear the editor
    app._open_image_from_path(card)
    app.crop_canvas.set_obb(OBB(256.0, 256.0, 360.0, 240.0, 0.0))
    app._normalize()

    restored = {b.name: b.crop_type for b in app.field_canvas.boxes()}
    assert restored == {"day": "field", "alive": "checkbox"}
    day = next(b for b in app.field_canvas.boxes() if b.name == "day")
    assert day.dictionary == "0123456789"
