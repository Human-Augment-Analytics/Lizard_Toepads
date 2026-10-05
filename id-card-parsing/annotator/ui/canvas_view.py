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
- Middle-drag, arrow keys, and scrollbars pan the preview. Fit recenters it.
- The angle slider rotates the box about its center (``set_angle`` takes radians).

Disabling editing blocks mouse drawing while leaving view navigation available.

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
    """Display a preview and edit one box independently of zoom and pan."""

    def __init__(self, master, size: int, on_box_change: Callable[[], None]):
        super().__init__(master, width=size, height=size, bg="black",
                         highlightthickness=0)
        self._size = size
        self._on_box_change = on_box_change
        self._editing_enabled = True

        self._pil: Image.Image | None = None     # preview in image space
        self._photo: ImageTk.PhotoImage | None = None
        self._image_item: int | None = None
        self._obb: OBB | None = None
        self._overlay_items: list[int] = []

        # view transform: canvas = zoom * image + pan
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0])
        self._scrollbars = (None, None)
        self.bind("<Configure>", lambda e: self._update_scrollbars())

        # drag state
        self._mode: str | None = None            # "create" | "corner"
        self._fixed_img: np.ndarray | None = None  # opposite corner (create)
        self._drag_corner: int | None = None     # index into local corner order

        self._pan_start: np.ndarray | None = None  # middle-drag anchor (canvas px)

        self._bind_events()

    def _bind_events(self) -> None:
        """Connect drawing and view navigation for supported mouse platforms."""
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
        """Fit a new preview and clear the previous box."""
        self._pil = pil_image
        self._obb = None
        self.fit_image()
        self._on_box_change()

    def attach_scrollbars(self, horizontal, vertical) -> None:
        """Connect scrollbar indicators to the custom zoom and pan transform."""
        self._scrollbars = (horizontal.set, vertical.set)
        self._update_scrollbars()

    def _viewport(self) -> tuple[int, int]:
        """Return canvas dimensions, using requested dimensions before layout."""
        return (self.winfo_width() if self.winfo_width() > 1 else self.winfo_reqwidth(),
                self.winfo_height() if self.winfo_height() > 1 else self.winfo_reqheight())

    def fit_image(self) -> None:
        """Center the complete preview without changing annotation geometry."""
        if self._pil is None:
            return
        width, height = self._viewport()
        self._zoom = min(width / self._pil.width, height / self._pil.height, 1.0)
        self._pan = np.array([(width-self._pil.width*self._zoom)/2,
                              (height-self._pil.height*self._zoom)/2])
        self._render_image()
        self._redraw_overlay()

    def _view_fraction(self, axis: int) -> tuple[float, float]:
        """Return visible fractions for the horizontal (0) or vertical (1) axis."""
        if self._pil is None:
            return (0.0, 1.0)
        extent = (self._pil.width, self._pil.height)[axis] * self._zoom
        viewport = self._viewport()[axis]
        if extent <= viewport:
            return (0.0, 1.0)
        first = float(np.clip(-self._pan[axis]/extent, 0, 1-viewport/extent))
        return (first, min(1.0, first+viewport/extent))

    def _update_scrollbars(self) -> None:
        """Synchronize scrollbar thumbs with the current view."""
        for axis, callback in enumerate(self._scrollbars):
            if callback is not None:
                callback(*self._view_fraction(axis))

    def _scroll_view(self, axis: int, *args):
        """Handle Tk scrollbar commands by panning, leaving box coordinates intact."""
        if not args:
            return self._view_fraction(axis)
        if self._pil is None:
            return
        extent = (self._pil.width, self._pil.height)[axis] * self._zoom
        viewport = self._viewport()[axis]
        if extent <= viewport:
            return
        if args[0] == "moveto":
            position = float(args[1]) * extent
        elif args[0] == "scroll":
            step = viewport*0.9 if args[2] == "pages" else _PAN_STEP
            position = -self._pan[axis]+int(args[1])*step
        else:
            return
        self._pan[axis] = -float(np.clip(position, 0, extent-viewport))
        self._move_items()

    def xview(self, *args):
        """Query or scroll horizontally through the custom view transform."""
        return self._scroll_view(0, *args)

    def yview(self, *args):
        """Query or scroll vertically through the custom view transform."""
        return self._scroll_view(1, *args)

    def set_obb(self, obb: OBB | None) -> None:
        """Display the supplied box by reference and notify the change callback."""
        self._obb = obb
        self._redraw_overlay()
        self._on_box_change()

    def get_obb(self) -> OBB | None:
        """Return the live box in preview-image coordinates, rather than a copy."""
        return self._obb

    def set_editing_enabled(self, enabled: bool) -> None:
        """Enable mouse editing or cancel its active drag; navigation stays enabled."""
        self._editing_enabled = enabled
        if not enabled:
            self._mode = None
            self._fixed_img = None
            self._drag_corner = None

    def set_angle(self, angle: float) -> None:
        """Rotate the current box to an angle in radians and notify listeners."""
        if self._obb is not None:
            self._obb.angle = angle
            self._redraw_overlay()
            self._on_box_change()

    # --- view transform -------------------------------------------------
    def _reset_view(self) -> None:
        """Reset view state to unit scale and zero pan without rendering."""
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0])

    def _to_canvas(self, pts: np.ndarray) -> np.ndarray:
        """Map preview-image points to canvas pixels for rendering and hit testing."""
        return np.asarray(pts, dtype=float).reshape(-1, 2) * self._zoom + self._pan

    def _to_image(self, pts: np.ndarray) -> np.ndarray:
        """Map canvas pixels back to preview-image coordinates for editing."""
        return (np.asarray(pts, dtype=float).reshape(-1, 2) - self._pan) / self._zoom

    # --- events: creation / corner editing ------------------------------
    def _on_press(self, event) -> None:
        if not self._editing_enabled:
            return
        image_point = self._to_image([[event.x, event.y]])[0]
        corner = self._hit_corner(event.x, event.y)
        if corner is not None:
            self._mode = "corner"
            self._drag_corner = corner
        else:
            # start a new box: pressed point is one corner, opposite is fixed
            self._mode = "create"
            self._fixed_img = image_point
            self._obb = OBB(cx=float(image_point[0]), cy=float(image_point[1]),
                            w=_MIN_SIZE, h=_MIN_SIZE, angle=0.0)
            self._redraw_overlay()
            self._on_box_change()

    def _on_drag(self, event) -> None:
        if not self._editing_enabled or self._mode is None:
            return
        image_point = self._to_image([[event.x, event.y]])[0]
        if self._mode == "create":
            self._update_from_opposite(self._fixed_img, image_point)
        elif self._mode == "corner":
            self._move_corner(self._drag_corner, image_point)
        self._redraw_overlay()
        self._on_box_change()

    def _on_release(self, event) -> None:
        self._mode = None
        self._fixed_img = None
        self._drag_corner = None

    # --- geometry helpers (axis-aligned in the box's own frame) ---------
    def _local_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return width and height unit vectors for the current box orientation."""
        angle = self._obb.angle
        width_axis = np.array([math.cos(angle), math.sin(angle)])
        height_axis = np.array([-math.sin(angle), math.cos(angle)])
        return width_axis, height_axis

    def _update_from_opposite(self, fixed: np.ndarray, moving: np.ndarray) -> None:
        """Rebuild the box so ``fixed`` and ``moving`` are opposite corners,
        measured in the box's rotated frame (angle preserved)."""
        width_axis, height_axis = self._local_axes()
        displacement = moving - fixed
        width = abs(float(displacement @ width_axis))
        height = abs(float(displacement @ height_axis))
        center = (fixed + moving) / 2.0
        self._obb.cx, self._obb.cy = float(center[0]), float(center[1])
        self._obb.w = max(width, _MIN_SIZE)
        self._obb.h = max(height, _MIN_SIZE)

    def _move_corner(self, idx: int, moving: np.ndarray) -> None:
        """Move corner ``idx`` to ``moving`` keeping the opposite corner fixed."""
        corners = self._obb.corners()
        fixed = corners[(idx + 2) % 4]
        self._update_from_opposite(fixed, moving)

    def _hit_corner(self, cx: float, cy: float) -> int | None:
        """Find a corner within the fixed canvas-pixel hit tolerance."""
        if self._obb is None:
            return None
        corners_canvas = self._to_canvas(self._obb.corners())
        for i, (hx, hy) in enumerate(corners_canvas):
            if abs(cx - hx) <= _HIT_PX and abs(cy - hy) <= _HIT_PX:
                return i
        return None

    # --- events: zoom ---------------------------------------------------
    def _on_wheel(self, event, delta: int | None = None) -> None:
        wheel_delta = event.delta if delta is None else delta
        if wheel_delta == 0:
            return
        factor = _ZOOM_STEP if wheel_delta > 0 else 1.0 / _ZOOM_STEP
        new_zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        # keep the image point under the cursor fixed on screen
        cursor = np.array([event.x, event.y], dtype=float)
        image_point = (cursor - self._pan) / self._zoom
        self._zoom = new_zoom
        self._pan = cursor - image_point * self._zoom
        self._render_image()
        self._redraw_overlay()

    # --- events: pan ----------------------------------------------------
    def _on_pan_press(self, event) -> None:
        self._pan_start = np.array([event.x, event.y], dtype=float)

    def _on_pan_drag(self, event) -> None:
        if self._pan_start is None:
            return
        cursor = np.array([event.x, event.y], dtype=float)
        self._pan = self._pan + (cursor - self._pan_start)
        self._pan_start = cursor
        self._move_items()

    def _on_pan_release(self, event) -> None:
        self._pan_start = None

    def _pan_by(self, dx: float, dy: float) -> None:
        self._pan = self._pan + np.array([dx, dy], dtype=float)
        self._move_items()

    # --- rendering ------------------------------------------------------
    def _render_image(self) -> None:
        """Rebuild the displayed image raster when the preview or zoom changes."""
        if self._pil is None:
            return
        display_width = max(1, int(round(self._pil.width * self._zoom)))
        display_height = max(1, int(round(self._pil.height * self._zoom)))
        resample = Image.NEAREST if self._zoom >= 1.0 else Image.BILINEAR
        resized_image = self._pil.resize((display_width, display_height), resample)
        # Keep the Tk image alive while the canvas references it.
        self._photo = ImageTk.PhotoImage(resized_image)
        if self._image_item is None:
            self._image_item = self.create_image(
                self._pan[0], self._pan[1], anchor="nw", image=self._photo,
                tags=("preview",),
            )
        else:
            self.itemconfigure(self._image_item, image=self._photo)
            self.coords(self._image_item, self._pan[0], self._pan[1])
        self._update_scrollbars()

    def _move_items(self) -> None:
        """Pan the retained raster and redraw handles without resizing the image."""
        if self._image_item is not None:
            self.coords(self._image_item, self._pan[0], self._pan[1])
        self._redraw_overlay()
        self._update_scrollbars()

    def _redraw_overlay(self) -> None:
        """Replace the box outline and handles using the current view transform."""
        for item in self._overlay_items:
            self.delete(item)
        self._overlay_items.clear()
        if self._obb is not None:
            canvas_corners = self._to_canvas(self._obb.corners())
            coordinates = [coordinate for point in canvas_corners for coordinate in point]
            self._overlay_items.append(self.create_polygon(
                coordinates, outline="#00ff6a", fill="", width=1,
            ))
            for hx, hy in canvas_corners:
                self._overlay_items.append(self.create_rectangle(
                    hx - _HANDLE_PX, hy - _HANDLE_PX,
                    hx + _HANDLE_PX, hy + _HANDLE_PX,
                    outline="#00ff6a", fill="#003318", width=1,
                ))
