"""Tkinter canvas: one OBB with draggable corner handles, plus scroll-to-zoom.

Coordinate spaces
-----------------
- *image space*: the preview/target-resolution pixel coordinates the OBB is
  stored in (0..target). This is what the backend exports.
- *canvas space*: on-screen pixels, related to image space by a uniform zoom
  and a pan offset: ``canvas = zoom * image + pan``.

The OBB is always kept in image space; the view transform only affects
rendering and hit-testing, so zoom/pan never alter the exported geometry.

Interaction
-----------
- Click-drag on empty space: create a box from the press point (one corner) to
  the cursor (the opposite corner). The box grows from where you clicked.
- Drag a corner handle: move that corner, keeping the opposite corner fixed.
- Mouse wheel: zoom in/out centered on the cursor.
- The angle slider rotates the box about its center (``set_angle``).

While a box is rotated, corner editing operates in the box's own axis-aligned
frame so handles stay intuitive.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Callable

import numpy as np
from PIL import Image, ImageTk

from ..obb import OBB

_HANDLE_PX = 5          # handle half-size in canvas pixels
_HIT_PX = 10            # click tolerance for grabbing a handle
_MIN_SIZE = 2.0         # minimum box side in image space
_ZOOM_STEP = 1.15
_ZOOM_MIN = 0.25
_ZOOM_MAX = 16.0
_PAN_STEP = 40          # arrow-key pan step in canvas pixels


class ImageCanvas(tk.Canvas):
    def __init__(self, master, size: int, on_box_change: Callable[[], None]):
        super().__init__(master, width=size, height=size, bg="black",
                         highlightthickness=0)
        self._size = size
        self._on_box_change = on_box_change

        self._pil: Image.Image | None = None     # preview in image space
        self._photo: ImageTk.PhotoImage | None = None
        self._obb: OBB | None = None

        # view transform: canvas = zoom * image + pan
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0])

        # drag state
        self._mode: str | None = None            # "create" | "corner"
        self._fixed_img: np.ndarray | None = None  # opposite corner (create)
        self._drag_corner: int | None = None     # index into local corner order

        self._pan_start: np.ndarray | None = None  # middle-drag anchor (canvas px)

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        # Windows / macOS wheel
        self.bind("<MouseWheel>", self._on_wheel)
        # Linux wheel
        self.bind("<Button-4>", lambda e: self._on_wheel(e, 120))
        self.bind("<Button-5>", lambda e: self._on_wheel(e, -120))
        # middle-click drag to pan
        self.bind("<ButtonPress-2>", self._on_pan_press)
        self.bind("<B2-Motion>", self._on_pan_drag)
        self.bind("<ButtonRelease-2>", self._on_pan_release)
        # arrow keys to pan (canvas must hold focus)
        self.bind("<Left>", lambda e: self._pan_by(_PAN_STEP, 0))
        self.bind("<Right>", lambda e: self._pan_by(-_PAN_STEP, 0))
        self.bind("<Up>", lambda e: self._pan_by(0, _PAN_STEP))
        self.bind("<Down>", lambda e: self._pan_by(0, -_PAN_STEP))
        # take keyboard focus when the pointer enters or the canvas is clicked
        self.bind("<Enter>", lambda e: self.focus_set())
        self.bind("<Button-1>", lambda e: self.focus_set(), add="+")

    # --- public API -----------------------------------------------------
    def show_preview(self, pil_image: Image.Image) -> None:
        self._pil = pil_image
        self._obb = None
        self._reset_view()
        self._redraw()
        self._on_box_change()

    def set_obb(self, obb: OBB | None) -> None:
        self._obb = obb
        self._redraw()
        self._on_box_change()

    def get_obb(self) -> OBB | None:
        return self._obb

    def set_angle(self, angle: float) -> None:
        if self._obb is not None:
            self._obb.angle = angle
            self._redraw()
            self._on_box_change()

    # --- view transform -------------------------------------------------
    def _reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0])

    def _to_canvas(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=float).reshape(-1, 2) * self._zoom + self._pan

    def _to_image(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=float).reshape(-1, 2) - self._pan) / self._zoom

    # --- events: creation / corner editing ------------------------------
    def _on_press(self, event) -> None:
        p_img = self._to_image([[event.x, event.y]])[0]
        corner = self._hit_corner(event.x, event.y)
        if corner is not None:
            self._mode = "corner"
            self._drag_corner = corner
        else:
            # start a new box: pressed point is one corner, opposite is fixed
            self._mode = "create"
            self._fixed_img = p_img
            self._obb = OBB(cx=float(p_img[0]), cy=float(p_img[1]),
                            w=_MIN_SIZE, h=_MIN_SIZE, angle=0.0)
            self._redraw()
            self._on_box_change()

    def _on_drag(self, event) -> None:
        if self._mode is None:
            return
        p_img = self._to_image([[event.x, event.y]])[0]
        if self._mode == "create":
            self._update_from_opposite(self._fixed_img, p_img)
        elif self._mode == "corner":
            self._move_corner(self._drag_corner, p_img)
        self._redraw()
        self._on_box_change()

    def _on_release(self, event) -> None:
        self._mode = None
        self._fixed_img = None
        self._drag_corner = None

    # --- geometry helpers (axis-aligned in the box's own frame) ---------
    def _local_axes(self) -> tuple[np.ndarray, np.ndarray]:
        a = self._obb.angle
        ux = np.array([math.cos(a), math.sin(a)])
        uy = np.array([-math.sin(a), math.cos(a)])
        return ux, uy

    def _update_from_opposite(self, fixed: np.ndarray, moving: np.ndarray) -> None:
        """Rebuild the box so ``fixed`` and ``moving`` are opposite corners,
        measured in the box's rotated frame (angle preserved)."""
        ux, uy = self._local_axes()
        d = moving - fixed
        w = abs(float(d @ ux))
        h = abs(float(d @ uy))
        center = (fixed + moving) / 2.0
        self._obb.cx, self._obb.cy = float(center[0]), float(center[1])
        self._obb.w = max(w, _MIN_SIZE)
        self._obb.h = max(h, _MIN_SIZE)

    def _move_corner(self, idx: int, moving: np.ndarray) -> None:
        """Move corner ``idx`` to ``moving`` keeping the opposite corner fixed."""
        corners = self._obb.corners()
        fixed = corners[(idx + 2) % 4]
        self._update_from_opposite(fixed, moving)

    def _hit_corner(self, cx: float, cy: float) -> int | None:
        if self._obb is None:
            return None
        corners_canvas = self._to_canvas(self._obb.corners())
        for i, (hx, hy) in enumerate(corners_canvas):
            if abs(cx - hx) <= _HIT_PX and abs(cy - hy) <= _HIT_PX:
                return i
        return None

    # --- events: zoom ---------------------------------------------------
    def _on_wheel(self, event, delta: int | None = None) -> None:
        d = event.delta if delta is None else delta
        if d == 0:
            return
        factor = _ZOOM_STEP if d > 0 else 1.0 / _ZOOM_STEP
        new_zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        # keep the image point under the cursor fixed on screen
        cursor = np.array([event.x, event.y], dtype=float)
        img_pt = (cursor - self._pan) / self._zoom
        self._zoom = new_zoom
        self._pan = cursor - img_pt * self._zoom
        self._redraw()

    # --- events: pan ----------------------------------------------------
    def _on_pan_press(self, event) -> None:
        self._pan_start = np.array([event.x, event.y], dtype=float)

    def _on_pan_drag(self, event) -> None:
        if self._pan_start is None:
            return
        cur = np.array([event.x, event.y], dtype=float)
        self._pan = self._pan + (cur - self._pan_start)
        self._pan_start = cur
        self._redraw()

    def _on_pan_release(self, event) -> None:
        self._pan_start = None

    def _pan_by(self, dx: float, dy: float) -> None:
        self._pan = self._pan + np.array([dx, dy], dtype=float)
        self._redraw()

    # --- rendering ------------------------------------------------------
    def _redraw(self) -> None:
        self.delete("all")
        if self._pil is not None:
            disp_w = max(1, int(round(self._pil.width * self._zoom)))
            disp_h = max(1, int(round(self._pil.height * self._zoom)))
            resample = Image.NEAREST if self._zoom >= 1.0 else Image.BILINEAR
            img = self._pil.resize((disp_w, disp_h), resample)
            self._photo = ImageTk.PhotoImage(img)
            self.create_image(self._pan[0], self._pan[1], anchor="nw",
                              image=self._photo)
        if self._obb is not None:
            pts = self._to_canvas(self._obb.corners())
            flat = [c for xy in pts for c in xy]
            self.create_polygon(flat, outline="#00ff6a", fill="", width=1)
            for hx, hy in pts:
                self.create_rectangle(
                    hx - _HANDLE_PX, hy - _HANDLE_PX,
                    hx + _HANDLE_PX, hy + _HANDLE_PX,
                    outline="#00ff6a", fill="#003318", width=1,
                )
