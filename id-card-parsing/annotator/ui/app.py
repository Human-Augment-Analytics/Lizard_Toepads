"""Tkinter application window wiring the backend to the canvas."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

from ..backend import AnnotatorBackend
from ..image_loader import ImageLoadError
from ..obb import OBB
from .canvas_view import ImageCanvas


class App(tk.Tk):
    def __init__(self, backend: AnnotatorBackend):
        super().__init__()
        self.backend = backend
        self.title("ID Card OBB Annotator")
        self._ids = backend.image_ids()
        self._preview = None          # current PreviewImage
        self._current_id: str | None = None

        self._build_widgets()
        self._populate_list()
        if self._ids:
            self._listbox.selection_set(0)
            self._load_selected()
        else:
            self._status.set("No images found in data directory.")

    # --- layout ---------------------------------------------------------
    def _build_widgets(self) -> None:
        target = self.backend.config.target_resolution

        left = ttk.Frame(self, padding=6)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="Images").pack(anchor="w")
        self._listbox = tk.Listbox(left, width=14, height=24, exportselection=False)
        self._listbox.pack(fill="y", expand=True)
        self._listbox.bind("<<ListboxSelect>>", lambda e: self._load_selected())

        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=4)
        ttk.Button(nav, text="Prev", command=lambda: self._step(-1)).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="Next", command=lambda: self._step(1)).pack(side="left", expand=True, fill="x")

        ttk.Label(left, text="Angle (deg)").pack(anchor="w", pady=(8, 0))
        self._angle = tk.DoubleVar(value=0.0)
        self._angle_scale = ttk.Scale(
            left, from_=-180, to=180, variable=self._angle,
            command=lambda v: self._on_angle(),
        )
        self._angle_scale.pack(fill="x")

        self._save_btn = ttk.Button(left, text="Save", command=self._save, state="disabled")
        self._save_btn.pack(fill="x", pady=(8, 0))

        self._status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._status, wraplength=140, foreground="#555").pack(
            anchor="w", pady=(8, 0)
        )

        self._canvas = ImageCanvas(self, size=target, on_box_change=self._on_box_change)
        self._canvas.grid(row=0, column=1, padx=6, pady=6)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    def _populate_list(self) -> None:
        self._listbox.delete(0, tk.END)
        for image_id in self._ids:
            self._listbox.insert(tk.END, image_id)

    # --- navigation / loading ------------------------------------------
    def _step(self, delta: int) -> None:
        if not self._ids:
            return
        sel = self._listbox.curselection()
        idx = (sel[0] if sel else 0) + delta
        idx = max(0, min(len(self._ids) - 1, idx))
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(idx)
        self._listbox.see(idx)
        self._load_selected()

    def _load_selected(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        image_id = self._ids[sel[0]]
        try:
            preview, _ = self.backend.open(image_id)
        except ImageLoadError as exc:
            self._status.set(f"Load failed: {exc}")
            return

        self._preview = preview
        self._current_id = image_id
        self._canvas.show_preview(preview.preview)

        rec = self.backend.existing_record(image_id)
        if rec is not None:
            cx, cy, w, h, angle = rec.obb_xywhr
            self._canvas.set_obb(OBB(cx, cy, w, h, angle))
            self._angle.set(math.degrees(angle))
            self._status.set(f"{image_id}: loaded existing annotation.")
        else:
            self._angle.set(0.0)
            self._status.set(f"{image_id}: draw a box.")

    # --- box / angle / save --------------------------------------------
    def _on_angle(self) -> None:
        self._canvas.set_angle(math.radians(self._angle.get()))

    def _on_box_change(self) -> None:
        has_box = self._canvas.get_obb() is not None
        self._save_btn.config(state="normal" if has_box else "disabled")

    def _save(self) -> None:
        obb = self._canvas.get_obb()
        if obb is None or self._preview is None or self._current_id is None:
            return
        path = self.backend.export(self._current_id, self._preview, obb)
        self._status.set(f"Saved {path.name}")
