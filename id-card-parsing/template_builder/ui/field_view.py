"""Tkinter canvas for drawing axis-aligned field boxes on the Normalized_Card.

Boxes are stored in *fraction* space (0..1 of the canvas side) so they stay
valid when the Normalized_Card is regenerated at a different canvas size. The
view fits the square Normalized_Card to the widget; boxes are drawn, selected,
moved, resized (via corner handles), and deleted.

Each box carries editable metadata (name, datatype, optional dictionary) owned
by the containing app; this canvas only manages geometry and selection and
calls back when the selection or geometry changes.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image, ImageTk

_HANDLE_PX = 5
_HIT_PX = 10
_MIN_FRAC = 0.01     # minimum box side as a fraction of the canvas


@dataclass
class FieldBox:
    """One crop region in fraction space plus its downstream metadata.

    ``crop_type`` is ``"field"`` (OCR, uses datatype/dictionary) or ``"checkbox"``
    (non-OCR: checkbox / pixel intensity).
    """

    left: float
    top: float
    right: float
    bottom: float
    name: str = ""
    datatype: str = "text"
    dictionary: str | None = None
    crop_type: str = "field"

    def as_box(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)


class FieldCanvas(tk.Canvas):
    """Display the Normalized_Card and edit axis-aligned field boxes."""

    def __init__(self, master, size: int, on_change: Callable[[], None]):
        super().__init__(master, width=size, height=size, bg="#202020",
                         highlightthickness=0)
        self._size = size
        self._on_change = on_change
        self._pil: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._image_item: int | None = None
        self._display = (size, size)     # fitted raster size in canvas px
        self._origin = (0, 0)            # top-left of the fitted raster
        self._boxes: list[FieldBox] = []
        self._selected: int | None = None
        self._tool = "draw"              # "draw" | "select"
        self._mode: str | None = None    # active drag: "create" | "move" | "resize"
        self._resize_corner: int | None = None
        self._drag_anchor = (0.0, 0.0)   # fraction-space anchor for move/create
        self._bind_events()

    # --- public API --------------------------------------------------------

    def show_card(self, pil_image: Image.Image) -> None:
        """Display a new Normalized_Card, keeping existing boxes (fraction space)."""
        self._pil = pil_image
        self._fit()
        self._redraw()

    def set_boxes(self, boxes: list[FieldBox]) -> None:
        self._boxes = list(boxes)
        self._selected = None
        self._redraw()
        self._on_change()

    def tool(self) -> str:
        """Return the active tool: ``"draw"`` or ``"select"``."""
        return self._tool

    def set_tool(self, tool: str) -> None:
        """Switch between drawing new boxes and selecting/editing existing ones."""
        if tool not in ("draw", "select"):
            raise ValueError("tool must be 'draw' or 'select'")
        self._tool = tool
        self._mode = None
        self._resize_corner = None

    def boxes(self) -> list[FieldBox]:
        return self._boxes

    def selected_index(self) -> int | None:
        return self._selected

    def selected_box(self) -> FieldBox | None:
        return self._boxes[self._selected] if self._selected is not None else None

    def delete_selected(self) -> None:
        if self._selected is not None:
            self._boxes.pop(self._selected)
            self._selected = None
            self._redraw()
            self._on_change()

    # --- coordinate mapping (fraction <-> canvas px) ----------------------

    def _fit(self) -> None:
        if self._pil is None:
            return
        width, height = self._pil.size
        scale = min(self._size / width, self._size / height)
        dw, dh = max(1, round(width * scale)), max(1, round(height * scale))
        self._display = (dw, dh)
        self._origin = ((self._size - dw) // 2, (self._size - dh) // 2)
        resized = self._pil.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        if self._image_item is None:
            self._image_item = self.create_image(*self._origin, anchor="nw",
                                                  image=self._photo, tags=("card",))
        else:
            self.itemconfigure(self._image_item, image=self._photo)
            self.coords(self._image_item, *self._origin)

    def _to_canvas(self, fx: float, fy: float) -> tuple[float, float]:
        ox, oy = self._origin
        dw, dh = self._display
        return ox + fx * dw, oy + fy * dh

    def _to_fraction(self, cx: float, cy: float) -> tuple[float, float]:
        ox, oy = self._origin
        dw, dh = self._display
        fx = (cx - ox) / dw if dw else 0.0
        fy = (cy - oy) / dh if dh else 0.0
        return min(1.0, max(0.0, fx)), min(1.0, max(0.0, fy))

    # --- events ------------------------------------------------------------

    def _bind_events(self) -> None:
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Key-Delete>", lambda e: self.delete_selected())
        self.bind("<Enter>", lambda e: self.focus_set())

    def _on_press(self, event) -> None:
        self.focus_set()
        if self._tool == "draw":
            # Draw mode always spawns a fresh box; existing boxes are untouched.
            fx, fy = self._to_fraction(event.x, event.y)
            self._boxes.append(FieldBox(fx, fy, fx + _MIN_FRAC, fy + _MIN_FRAC))
            self._selected = len(self._boxes) - 1
            self._mode = "create"
            self._drag_anchor = (fx, fy)
            self._redraw()
            self._on_change()
            return

        # Select mode: resize the selected box via a handle, else select/move.
        if self._selected is not None:
            corner = self._hit_handle(self._selected, event.x, event.y)
            if corner is not None:
                self._mode = "resize"
                self._resize_corner = corner
                return
        hit = self._hit_box(event.x, event.y)
        if hit is not None:
            self._selected = hit
            self._mode = "move"
            fx, fy = self._to_fraction(event.x, event.y)
            box = self._boxes[hit]
            self._drag_anchor = (fx - box.left, fy - box.top)
            self._redraw()
            self._on_change()
        else:
            # Click on empty space clears the selection without creating a box.
            self._selected = None
            self._mode = None
            self._redraw()
            self._on_change()

    def _on_drag(self, event) -> None:
        if self._mode is None or self._selected is None:
            return
        fx, fy = self._to_fraction(event.x, event.y)
        box = self._boxes[self._selected]
        if self._mode == "create":
            ax, ay = self._drag_anchor
            box.left, box.right = min(ax, fx), max(ax, fx)
            box.top, box.bottom = min(ay, fy), max(ay, fy)
            self._enforce_min(box)
        elif self._mode == "move":
            width = box.right - box.left
            height = box.bottom - box.top
            ax, ay = self._drag_anchor
            new_left = min(1 - width, max(0.0, fx - ax))
            new_top = min(1 - height, max(0.0, fy - ay))
            box.left, box.right = new_left, new_left + width
            box.top, box.bottom = new_top, new_top + height
        elif self._mode == "resize":
            self._resize(box, self._resize_corner, fx, fy)
        self._redraw()
        self._on_change()

    def _on_release(self, event) -> None:
        was_creating = self._mode == "create"
        self._mode = None
        self._resize_corner = None
        if was_creating:
            # After drawing, drop into select mode so the new box can be labeled
            # and the next click edits rather than spawning another box.
            self._tool = "select"
            self._on_change()

    # --- geometry helpers --------------------------------------------------

    def _enforce_min(self, box: FieldBox) -> None:
        if box.right - box.left < _MIN_FRAC:
            box.right = min(1.0, box.left + _MIN_FRAC)
        if box.bottom - box.top < _MIN_FRAC:
            box.bottom = min(1.0, box.top + _MIN_FRAC)

    def _resize(self, box: FieldBox, corner: int, fx: float, fy: float) -> None:
        # corners order: 0 TL, 1 TR, 2 BR, 3 BL
        if corner in (0, 3):
            box.left = min(fx, box.right - _MIN_FRAC)
        else:
            box.right = max(fx, box.left + _MIN_FRAC)
        if corner in (0, 1):
            box.top = min(fy, box.bottom - _MIN_FRAC)
        else:
            box.bottom = max(fy, box.top + _MIN_FRAC)

    def _corners_canvas(self, box: FieldBox):
        return [
            self._to_canvas(box.left, box.top),
            self._to_canvas(box.right, box.top),
            self._to_canvas(box.right, box.bottom),
            self._to_canvas(box.left, box.bottom),
        ]

    def _hit_handle(self, index: int, cx: float, cy: float) -> int | None:
        for i, (hx, hy) in enumerate(self._corners_canvas(self._boxes[index])):
            if abs(cx - hx) <= _HIT_PX and abs(cy - hy) <= _HIT_PX:
                return i
        return None

    def _hit_box(self, cx: float, cy: float) -> int | None:
        # Topmost (last drawn) box wins.
        for index in reversed(range(len(self._boxes))):
            box = self._boxes[index]
            x0, y0 = self._to_canvas(box.left, box.top)
            x1, y1 = self._to_canvas(box.right, box.bottom)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return index
        return None

    # --- rendering ---------------------------------------------------------

    def _redraw(self) -> None:
        self.delete("overlay")
        for index, box in enumerate(self._boxes):
            corners = self._corners_canvas(box)
            coords = [c for point in corners for c in point]
            selected = index == self._selected
            # Checkbox crops render green; OCR fields render blue; selection yellow.
            base = "#30d060" if box.crop_type == "checkbox" else "#00b0ff"
            color = "#ffcc00" if selected else base
            self.create_polygon(coords, outline=color, fill="",
                                 width=2 if selected else 1, tags="overlay")
            prefix = "\u2611 " if box.crop_type == "checkbox" else ""
            label = prefix + (box.name or f"field{index + 1}")
            lx, ly = corners[0]
            self.create_text(lx + 2, ly - 8, anchor="w", text=label,
                             fill=color, tags="overlay")
            if selected:
                for hx, hy in corners:
                    self.create_rectangle(hx - _HANDLE_PX, hy - _HANDLE_PX,
                                          hx + _HANDLE_PX, hy + _HANDLE_PX,
                                          outline="#ffcc00", fill="#332a00", tags="overlay")
