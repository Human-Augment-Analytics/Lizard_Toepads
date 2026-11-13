"""Template builder app: two-stage flow over the GUI-free backend.

Stage 1 (crop): load a card image, draw a Card_OBB on the preview (reusing the
annotator's ``ImageCanvas``), then normalize.
Stage 2 (fields): draw/select/move/resize/name/type field boxes on the
Normalized_Card and save a Template_Record JSON.

All geometry and persistence go through ``TemplateBuilderBackend``; this module
only wires widgets and user actions to it.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from annotator.image_loader import ImageLoadError
from annotator.obb import OBB
from annotator.ui.canvas_view import ImageCanvas

from ..backend import TemplateBuilderBackend
from ..config import TemplateBuilderConfig
from ..template_record import VALID_DATATYPES, FieldSpec, TemplateRecord
from .field_view import FieldBox, FieldCanvas

_CANVAS_PX = 640      # on-screen canvas size for both stages


class App(tk.Tk):
    def __init__(self, backend: TemplateBuilderBackend | None = None):
        super().__init__()
        self.title("ID Card Template Builder")
        self.backend = backend or TemplateBuilderBackend(TemplateBuilderConfig())
        self._corners_original: np.ndarray | None = None
        # A saved template found on image open, restored into the fields stage
        # after the card is normalized.
        self._pending_record: TemplateRecord | None = None

        self._container = ttk.Frame(self, padding=8)
        self._container.pack(fill="both", expand=True)
        self._status = tk.StringVar(value="Load a card image to begin.")
        ttk.Label(self, textvariable=self._status, anchor="w",
                  relief="sunken").pack(fill="x", side="bottom")

        self._build_crop_stage()
        self._build_field_stage()
        self._show_stage("crop")

    # --- stage 1: crop -----------------------------------------------------

    def _build_crop_stage(self) -> None:
        self.crop_frame = ttk.Frame(self._container)
        controls = ttk.Frame(self.crop_frame)
        controls.pack(side="left", fill="y", padx=(0, 8))
        ttk.Button(controls, text="Open image…", command=self._open_image).pack(fill="x")
        ttk.Label(controls, text="Angle").pack(anchor="w", pady=(12, 0))
        self._angle = tk.DoubleVar(value=0.0)
        ttk.Scale(controls, from_=-180, to=180, variable=self._angle,
                  command=self._on_angle).pack(fill="x")
        self._normalize_btn = ttk.Button(controls, text="Normalize \u2192",
                                          command=self._normalize, state="disabled")
        self._normalize_btn.pack(fill="x", pady=(12, 0))
        self.crop_canvas = ImageCanvas(self.crop_frame, _CANVAS_PX, self._on_crop_change)
        self.crop_canvas.pack(side="right")

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open card image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")])
        if path:
            self._open_image_from_path(path)

    def _open_image_from_path(self, path) -> None:
        """Load a card and stage any existing template for restore (testable seam)."""
        try:
            preview = self.backend.load_image(path)
        except ImageLoadError as exc:
            self._status.set(f"Could not open image: {exc}")
            return
        self.crop_canvas.show_preview(preview.preview)
        self._status.set(f"Loaded {path} ({preview.orig_size[0]}x{preview.orig_size[1]}). "
                         "Drag to draw the card box, then Normalize.")
        # Restore a prior template for review/edit, if present: the Card_OBB now,
        # and its field boxes after the card is normalized.
        record = self.backend.existing_record()
        self._pending_record = record
        if record is not None:
            self._restore_card_obb(record)
            self._status.set(f"Loaded {path}. Found an existing template "
                             f"({len(record.fields)} boxes); Normalize to edit it.")

    def _on_angle(self, _value) -> None:
        self.crop_canvas.set_angle(math.radians(self._angle.get()))

    def _on_crop_change(self) -> None:
        has_box = self.crop_canvas.get_obb() is not None
        self._normalize_btn.configure(state="normal" if has_box else "disabled")

    def _normalize(self) -> None:
        obb = self.crop_canvas.get_obb()
        if obb is None:
            return
        try:
            corners = self.backend.obb_to_original(obb)
            result = self.backend.normalize(corners)
        except (ValueError, RuntimeError) as exc:
            self._status.set(f"Could not normalize: {exc}")
            return
        self._corners_original = corners
        self.field_canvas.show_card(result.image)
        # Auto-restore field boxes from a loaded template (once), so an existing
        # template can be edited rather than redrawn.
        if self._pending_record is not None:
            self._restore_field_boxes(self._pending_record)
            self._pending_record = None
            self._status.set(f"Normalized to {self.backend.config.canvas_size}px. "
                             "Restored existing field boxes for editing.")
        else:
            self._status.set(f"Normalized to {self.backend.config.canvas_size}px. "
                             "Draw field boxes and name each one.")
        self._refresh_save_state()
        self._show_stage("fields")

    def _restore_card_obb(self, record: TemplateRecord) -> None:
        """Map a stored original-pixel Card_OBB back into preview space to edit."""
        from annotator.geometry import apply_affine
        m = self.backend.preview_transform()
        corners_preview = apply_affine(m, np.asarray(record.card_obb_corners, float))
        self.crop_canvas.set_obb(OBB.from_corners(corners_preview))

    def _restore_field_boxes(self, record: TemplateRecord) -> None:
        """Rebuild editable FieldBoxes from a loaded template's FieldSpecs."""
        boxes = [
            FieldBox(*spec.box, name=spec.name, datatype=spec.datatype,
                     dictionary=spec.dictionary, crop_type=spec.crop_type)
            for spec in record.fields
        ]
        self.field_canvas.set_boxes(boxes)

    # --- stage 2: fields ---------------------------------------------------

    def _build_field_stage(self) -> None:
        self.field_frame = ttk.Frame(self._container)
        side = ttk.Frame(self.field_frame)
        side.pack(side="left", fill="y", padx=(0, 8))

        ttk.Button(side, text="\u2190 Back to crop",
                   command=lambda: self._show_stage("crop")).pack(fill="x")

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Tool").pack(anchor="w")
        self._tool = tk.StringVar(value="draw")
        tool_row = ttk.Frame(side)
        tool_row.pack(fill="x")
        ttk.Radiobutton(tool_row, text="Draw", value="draw", variable=self._tool,
                        command=self._apply_tool).pack(side="left")
        ttk.Radiobutton(tool_row, text="Select", value="select", variable=self._tool,
                        command=self._apply_tool).pack(side="left")
        ttk.Label(side, text="Draw: drag to add a box. Select: click to pick, "
                             "drag to move, handles to resize.",
                  wraplength=150, foreground="#555").pack(anchor="w")

        ttk.Label(side, text="Canvas size").pack(anchor="w", pady=(12, 0))
        self._canvas_size = tk.IntVar(value=self.backend.config.canvas_size)
        size_row = ttk.Frame(side)
        size_row.pack(fill="x")
        ttk.Entry(size_row, textvariable=self._canvas_size, width=6).pack(side="left")
        ttk.Button(size_row, text="Apply", command=self._apply_canvas_size).pack(side="left")

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Selected box").pack(anchor="w")
        ttk.Label(side, text="Name").pack(anchor="w")
        self._field_name = tk.StringVar()
        name_entry = ttk.Entry(side, textvariable=self._field_name)
        name_entry.pack(fill="x")
        self._field_name.trace_add("write", lambda *_: self._apply_metadata())

        ttk.Label(side, text="Crop type").pack(anchor="w", pady=(8, 0))
        self._field_crop_type = tk.StringVar(value="field")
        crop_row = ttk.Frame(side)
        crop_row.pack(fill="x")
        ttk.Radiobutton(crop_row, text="Field (OCR)", value="field",
                        variable=self._field_crop_type,
                        command=self._apply_metadata).pack(side="left")
        ttk.Radiobutton(crop_row, text="Checkbox", value="checkbox",
                        variable=self._field_crop_type,
                        command=self._apply_metadata).pack(side="left")

        self._datatype_label = ttk.Label(side, text="Datatype")
        self._datatype_label.pack(anchor="w", pady=(8, 0))
        self._field_type = tk.StringVar(value=VALID_DATATYPES[0])
        self._datatype_combo = ttk.Combobox(side, textvariable=self._field_type,
                                            values=list(VALID_DATATYPES), state="readonly")
        self._datatype_combo.pack(fill="x")
        self._field_type.trace_add("write", lambda *_: self._apply_metadata())

        self._dict_label = ttk.Label(side, text="Dictionary (optional)")
        self._dict_label.pack(anchor="w", pady=(8, 0))
        self._field_dict = tk.StringVar()
        self._dict_entry = ttk.Entry(side, textvariable=self._field_dict)
        self._dict_entry.pack(fill="x")
        self._field_dict.trace_add("write", lambda *_: self._apply_metadata())

        ttk.Button(side, text="Delete field",
                   command=self._delete_field).pack(fill="x", pady=(8, 0))

        ttk.Separator(side).pack(fill="x", pady=8)
        self._save_btn = ttk.Button(side, text="Save template", command=self._save,
                                    state="disabled")
        self._save_btn.pack(fill="x")

        self.field_canvas = FieldCanvas(self.field_frame, _CANVAS_PX, self._on_field_change)
        self.field_canvas.pack(side="right")
        self._syncing = False

    def _apply_tool(self) -> None:
        self.field_canvas.set_tool(self._tool.get())

    def _apply_canvas_size(self) -> None:
        if self._corners_original is None:
            return
        try:
            self.backend.set_canvas_size(int(self._canvas_size.get()))
            result = self.backend.normalize(self._corners_original)
        except (ValueError, RuntimeError, tk.TclError) as exc:
            self._status.set(f"Invalid canvas size: {exc}")
            return
        self.field_canvas.show_card(result.image)
        self._status.set(f"Canvas size set to {self.backend.config.canvas_size}px.")

    def _on_field_change(self) -> None:
        # Keep the tool radio in sync (the canvas auto-switches to Select after
        # a box is drawn so it can be labeled immediately).
        if hasattr(self, "_tool") and self._tool.get() != self.field_canvas.tool():
            self._tool.set(self.field_canvas.tool())
        box = self.field_canvas.selected_box()
        if box is not None and not self._syncing:
            self._syncing = True
            self._field_name.set(box.name)
            self._field_crop_type.set(box.crop_type)
            self._field_type.set(box.datatype if box.datatype in VALID_DATATYPES else VALID_DATATYPES[0])
            self._field_dict.set(box.dictionary or "")
            self._syncing = False
        self._update_ocr_controls()
        self._refresh_save_state()

    def _apply_metadata(self) -> None:
        if self._syncing:
            return
        box = self.field_canvas.selected_box()
        if box is None:
            return
        box.name = self._field_name.get().strip()
        box.crop_type = self._field_crop_type.get()
        box.datatype = self._field_type.get()
        box.dictionary = self._field_dict.get().strip() or None
        self._update_ocr_controls()
        self.field_canvas._redraw()
        self._refresh_save_state()

    def _update_ocr_controls(self) -> None:
        """Datatype/dictionary are only meaningful for OCR (field) crops."""
        is_field = self._field_crop_type.get() == "field"
        self._datatype_combo.configure(state="readonly" if is_field else "disabled")
        self._dict_entry.configure(state="normal" if is_field else "disabled")

    def _delete_field(self) -> None:
        self.field_canvas.delete_selected()
        self._refresh_save_state()

    def _collect_fields(self) -> tuple[list[FieldSpec], str | None]:
        """Build validated FieldSpecs; return (specs, error_message)."""
        boxes = self.field_canvas.boxes()
        if not boxes:
            return [], "Add at least one field box."
        specs, names = [], set()
        for index, box in enumerate(boxes, 1):
            if not box.name:
                return [], f"Field {index} needs a name."
            if box.name in names:
                return [], f"Duplicate field name: {box.name}"
            names.add(box.name)
            try:
                specs.append(FieldSpec(box.name, box.as_box(), box.datatype,
                                       box.dictionary, crop_type=box.crop_type))
            except ValueError as exc:
                return [], f"Field '{box.name}': {exc}"
        return specs, None

    def _refresh_save_state(self) -> None:
        _specs, error = self._collect_fields()
        if error is None:
            self._save_btn.configure(state="normal")
        else:
            self._save_btn.configure(state="disabled")
            if self.field_frame.winfo_ismapped():
                self._status.set(error)

    def _save(self) -> None:
        specs, error = self._collect_fields()
        if error is not None or self._corners_original is None:
            self._status.set(error or "Nothing to save; crop a card first.")
            return
        try:
            record = self.backend.build_record(self._corners_original, specs)
            path = self.backend.save(record)
        except (ValueError, OSError) as exc:
            self._status.set(f"Save failed: {exc}")
            return
        self._status.set(f"Saved template: {path}")
        messagebox.showinfo("Saved", f"Template written to:\n{path}")

    # --- stage switching ---------------------------------------------------

    def _show_stage(self, stage: str) -> None:
        self.crop_frame.pack_forget()
        self.field_frame.pack_forget()
        (self.crop_frame if stage == "crop" else self.field_frame).pack(fill="both", expand=True)
