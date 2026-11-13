"""Headless interaction tests for FieldCanvas tool modes.

Constructs a real Tk canvas (skips if Tk unavailable) and drives the press/
drag/release handlers directly, the same calls Tk makes on mouse events.
"""

from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace

import pytest
from PIL import Image

from template_builder.ui.field_view import FieldBox, FieldCanvas

_SIZE = 400


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk not available")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def canvas(tk_root):
    cv = FieldCanvas(tk_root, _SIZE, on_change=lambda: None)
    # A square card fills the whole canvas so canvas px == fraction * size.
    cv.show_card(Image.new("RGB", (_SIZE, _SIZE), (0, 0, 0)))
    yield cv
    cv.destroy()


def _drag(cv, x0, y0, x1, y1):
    cv._on_press(SimpleNamespace(x=x0, y=y0))
    cv._on_drag(SimpleNamespace(x=x1, y=y1))
    cv._on_release(SimpleNamespace(x=x1, y=y1))


def test_default_tool_is_draw(canvas):
    assert canvas.tool() == "draw"


def test_draw_creates_a_box(canvas):
    _drag(canvas, 40, 40, 140, 100)
    assert len(canvas.boxes()) == 1


def test_second_draw_keeps_the_first_box(canvas):
    # The core bug report: a second drag must NOT remove the first box.
    canvas.set_tool("draw")
    _drag(canvas, 40, 40, 140, 100)
    canvas.set_tool("draw")
    _drag(canvas, 200, 200, 300, 280)
    assert len(canvas.boxes()) == 2


def test_overlapping_second_draw_still_creates_new_box(canvas):
    # Even if the second drag starts inside the first box, draw mode spawns a new
    # box rather than moving/resizing the existing one.
    canvas.set_tool("draw")
    _drag(canvas, 40, 40, 200, 200)
    canvas.set_tool("draw")
    _drag(canvas, 60, 60, 260, 260)  # starts inside box 1
    assert len(canvas.boxes()) == 2


def test_select_tool_moves_not_creates(canvas):
    canvas.set_tool("draw")
    _drag(canvas, 40, 40, 200, 160)
    assert len(canvas.boxes()) == 1
    box = canvas.boxes()[0]
    left_before = box.left
    canvas.set_tool("select")
    # Drag from inside the box; should move it, not create a second box.
    _drag(canvas, 120, 100, 160, 100)
    assert len(canvas.boxes()) == 1
    assert canvas.boxes()[0].left != left_before


def test_select_tool_click_selects_box(canvas):
    canvas.set_tool("draw")
    _drag(canvas, 40, 40, 160, 120)
    canvas.set_tool("draw")
    _drag(canvas, 220, 220, 320, 300)
    canvas.set_tool("select")
    # Click inside the first box selects it.
    canvas._on_press(SimpleNamespace(x=100, y=80))
    canvas._on_release(SimpleNamespace(x=100, y=80))
    assert canvas.selected_index() == 0


def test_delete_selected_removes_only_that_box(canvas):
    canvas.set_tool("draw")
    _drag(canvas, 40, 40, 160, 120)
    canvas.set_tool("draw")
    _drag(canvas, 220, 220, 320, 300)
    canvas.set_tool("select")
    canvas._on_press(SimpleNamespace(x=100, y=80))
    canvas._on_release(SimpleNamespace(x=100, y=80))
    assert canvas.selected_index() == 0
    canvas.delete_selected()
    assert len(canvas.boxes()) == 1


def test_draw_auto_selects_new_box_for_editing(canvas):
    canvas.set_tool("draw")
    _drag(canvas, 40, 40, 160, 120)
    # The freshly drawn box is selected so its metadata can be edited immediately.
    assert canvas.selected_index() == 0
