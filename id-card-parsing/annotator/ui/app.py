"""Coordinate the annotator's Tkinter UI and background processing.

Manage navigation, detection, session edits, and single-image or bulk saves.
Workers report through queues; Tk widgets are updated only on the main thread.
Imaging libraries are imported lazily so the loading window can appear first.
"""

from __future__ import annotations

import math
import queue
import tkinter as tk
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from ..bulk import BulkSummary, ids_in_range, save_template_range

if TYPE_CHECKING:
    from ..backend import AnnotatorBackend
    from ..image_loader import PreviewImage
    from ..record import AnnotationRecord
    from ..obb import OBB
    from .canvas_view import ImageCanvas
    from ..template_matcher import CardTemplateMatcher, TemplateMatch


_POLL_INTERVAL_MS = 40
_LOADING_ANIMATION_INTERVAL_MS = 12


@dataclass(frozen=True)
class _StartupResult:
    """Carry lazily loaded dependencies or a startup error to the Tk thread."""

    backend: AnnotatorBackend | None = None
    image_ids: list[str] = field(default_factory=list)
    box_class: type[OBB] | None = None
    canvas_class: type[ImageCanvas] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ImageLoadResult:
    """Identify an image request and its preview, annotation, and errors."""

    request: int
    image_id: str
    preview: PreviewImage | None = None
    record: AnnotationRecord | None = None
    match: TemplateMatch | None = None
    match_error: str | None = None
    record_error: str | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class _BulkResult:
    """Report batch progress or completion without touching Tk widgets."""

    done: bool
    summary: BulkSummary | None = None
    error: str | None = None


class App(tk.Tk):
    """Manage the annotation window, session edits, and background jobs.

    A single worker serializes loading and matching. Request IDs prevent stale
    results from replacing newer selections; drafts survive navigation until
    explicitly saved or discarded on close.
    """

    def __init__(self, backend: AnnotatorBackend | None = None):
        """Start with an injected backend or load default dependencies lazily."""
        super().__init__()
        self.backend = backend
        self.title("ID Card OBB Annotator")
        self._ids: list[str] = []
        self._preview: PreviewImage | None = None
        self._current_id: str | None = None
        self._closed = threading.Event()
        self._load_future: Future[None] | None = None
        self._startup_future: Future[None] | None = None
        self._drafts: dict[str, OBB | None] = {}
        self._committed_obb: OBB | None = None
        self._initial_obb: OBB | None = None
        self._initial_box_source = "empty box"
        self._load_executor = ThreadPoolExecutor(max_workers=1)
        self._load_results: queue.Queue[_ImageLoadResult] = queue.Queue()
        self._load_updates: queue.Queue[tuple[int, str]] = queue.Queue()
        self._load_request = 0
        self._loading_id: str | None = None
        self._template_matcher: CardTemplateMatcher | None = None
        self._template_error: str | None = None
        self._template_initialized = False
        self._bulk_results: queue.Queue[_BulkResult] = queue.Queue()
        self._bulk_cancel = threading.Event()
        self._bulk_running = False
        self._bulk_future: Future[None] | None = None
        self._bulk_dialog: tk.Toplevel | None = None
        self.protocol("WM_DELETE_WINDOW", self._close)
        if backend is None:
            self._start_window()
        else:
            # Explicit backends remain supported by tests and embedded callers.
            from ..obb import OBB as box_class
            from .canvas_view import ImageCanvas as canvas_class
            self._finish_startup(backend, backend.image_ids(), box_class, canvas_class)

    def _finish_startup(
        self, backend: AnnotatorBackend, image_ids: list[str],
        box_class: type[OBB], canvas_class: type[ImageCanvas],
    ) -> None:
        """Install loaded dependencies and build the editor on the Tk thread."""
        self._box_class = box_class
        self._canvas_class = canvas_class
        self.backend = backend
        self._ids = image_ids
        self._build_widgets()
        self._populate_list()
        if self._ids:
            self._listbox.selection_set(0)
            self._load_selected()
        else:
            self._status.set("No images found in data directory.")
        self._poll_after_id = self.after(_POLL_INTERVAL_MS, self._poll_load_results)

    def _start_window(self) -> None:
        """Show the main window before importing imaging/numerical libraries."""
        self.geometry(
            f"{max(400, min(1324, self.winfo_screenwidth()-80))}x"
            f"{max(400, min(900, self.winfo_screenheight()-100))}"
        )
        self._startup_panel = ttk.Frame(self, padding=24)
        self._startup_panel.pack(fill="both", expand=True)
        self._startup_card = ttk.Frame(self._startup_panel, padding=24)
        self._startup_card.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(self._startup_card, text="ID Card OBB Annotator",
                  font=("TkDefaultFont", 14, "bold")).pack()
        ttk.Label(self._startup_card, text="Loading…").pack(pady=(12, 8))
        self._progress = ttk.Progressbar(self._startup_card, mode="indeterminate", length=240)
        self._progress.pack(fill="x")
        self._progress.start(_LOADING_ANIMATION_INTERVAL_MS)
        self._startup_results: queue.Queue[_StartupResult] = queue.Queue()

        def prepare() -> None:
            try:
                from ..backend import AnnotatorBackend
                from ..config import AnnotatorConfig
                from ..obb import OBB as box_class
                from .canvas_view import ImageCanvas as canvas_class
                backend = AnnotatorBackend(AnnotatorConfig())
                self._startup_results.put(_StartupResult(
                    backend=backend, image_ids=backend.image_ids(),
                    box_class=box_class, canvas_class=canvas_class,
                ))
            except Exception as exc:
                self._startup_results.put(_StartupResult(error=str(exc)))
        self._startup_future = self._load_executor.submit(prepare)
        self._poll_after_id = self.after(_POLL_INTERVAL_MS, self._poll_startup)

    def _poll_startup(self) -> None:
        """Transition from the loading screen or display a startup error."""
        try:
            result = self._startup_results.get_nowait()
        except queue.Empty:
            self._poll_after_id = self.after(_POLL_INTERVAL_MS, self._poll_startup)
            return
        self._progress.stop()
        if result.error is not None:
            ttk.Label(self._startup_card, text=f"Could not start: {result.error}",
                      foreground="#b42318", wraplength=500).pack(pady=12)
            return
        self._startup_panel.destroy()
        assert result.backend is not None and result.box_class is not None and result.canvas_class is not None
        self._finish_startup(result.backend, result.image_ids, result.box_class, result.canvas_class)

    # --- layout ---------------------------------------------------------
    def _build_widgets(self) -> None:
        """Compose the sidebar and viewer, then size the window to the display."""
        self._build_sidebar()
        self._build_viewer()
        target = self.backend.config.target_resolution

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.geometry(
            f"{max(400, min(target+300, self.winfo_screenwidth()-80))}x"
            f"{max(400, min(max(700, target+120), self.winfo_screenheight()-100))}"
        )

    def _build_sidebar(self) -> None:
        """Build scrollable navigation, annotation actions, and status controls."""
        sidebar_frame = ttk.Frame(self)
        sidebar_frame.grid(row=0, column=0, sticky="ns")
        sidebar = tk.Canvas(sidebar_frame, width=240, highlightthickness=0)
        self._sidebar = sidebar
        sidebar.pack(side="left", fill="both", expand=True)
        side_scroll = ttk.Scrollbar(sidebar_frame, orient="vertical", command=sidebar.yview)
        side_scroll.pack(side="right", fill="y")
        sidebar.config(yscrollcommand=side_scroll.set)
        left = ttk.Frame(sidebar, padding=12)
        side_window = sidebar.create_window(0, 0, anchor="nw", window=left, width=240)
        left.bind("<Configure>", lambda e: sidebar.config(scrollregion=sidebar.bbox("all")))
        sidebar.bind("<Configure>", lambda e: sidebar.itemconfigure(side_window, width=e.width))

        ttk.Label(left, text="Images", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(left, text=f"{len(self._ids):,} image files").pack(anchor="w", pady=(2, 8))
        self._search_id = tk.StringVar()
        ttk.Label(left, text="Image number (e.g. 68)").pack(anchor="w", pady=(4, 0))
        search = ttk.Frame(left)
        search.pack(fill="x", pady=(2, 6))
        self._search_entry = ttk.Entry(search, textvariable=self._search_id, width=9)
        self._search_entry.pack(side="left", fill="x", expand=True)
        self._search_entry.bind("<Return>", lambda e: self._search())
        self._search_entry.bind("<KP_Enter>", lambda e: self._search())
        ttk.Button(search, text="Search", command=self._search).pack(side="left", padx=(4, 0))
        self._search_feedback = tk.StringVar(value="Use the number or exact image ID.")
        self._search_feedback_label = ttk.Label(
            left, textvariable=self._search_feedback, wraplength=210,
            foreground="#555",
        )
        self._search_feedback_label.pack(anchor="w", fill="x", pady=(0, 8))

        image_list = ttk.Frame(left)
        image_list.pack(fill="both", expand=True)
        self._listbox = tk.Listbox(image_list, width=14, height=12, exportselection=False)
        self._listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(image_list, orient="vertical", command=self._listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self._listbox.config(yscrollcommand=scrollbar.set)
        self._listbox.bind("<<ListboxSelect>>", lambda e: self._load_selected())

        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=4)
        ttk.Button(nav, text="Prev", command=lambda: self._step(-1)).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="Next", command=lambda: self._step(1)).pack(side="left", expand=True, fill="x")

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="Annotation", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self._angle = tk.DoubleVar(value=0.0)
        self._angle_text = tk.StringVar(value="Angle: 0.0°")
        self._angle.trace_add("write", lambda *_: self._angle_text.set(
            f"Angle: {self._angle.get():.1f}°"
        ))
        ttk.Label(left, textvariable=self._angle_text).pack(anchor="w", pady=(8, 0))
        self._angle_scale = ttk.Scale(
            left, from_=-180, to=180, variable=self._angle,
            command=lambda v: self._on_angle(),
        )
        self._angle_scale.pack(fill="x")

        self._save_btn = ttk.Button(left, text="Save annotation", command=self._save, state="disabled")
        self._save_btn.pack(fill="x", pady=(8, 0))
        self._reset_btn = ttk.Button(left, text="Reset box", command=self._reset_box, state="disabled")
        self._reset_btn.pack(fill="x", pady=(4, 0))
        ttk.Button(left, text="Save range…", command=self._open_bulk_dialog).pack(fill="x", pady=(8, 0))

        self._status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._status, wraplength=210, foreground="#555").pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="Draw: left-click and drag\nAdjust: drag a corner\nPan: middle-click and drag\nZoom: mouse wheel",
                  wraplength=210, foreground="#555").pack(anchor="w")

    def _build_viewer(self) -> None:
        """Build the image header, scrollable canvas, and centered loading panel."""
        target = self.backend.config.target_resolution

        viewer = ttk.Frame(self, padding=(6, 10, 12, 10))
        viewer.grid(row=0, column=1, sticky="nsew")
        self._image_title = tk.StringVar(value="Select an image" if self._ids else "No images available")
        ttk.Label(viewer, textvariable=self._image_title,
                  font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w")
        self._load_text = tk.StringVar(value="")
        ttk.Label(viewer, textvariable=self._load_text).grid(row=1, column=0, sticky="w", pady=(3, 6))
        ttk.Button(viewer, text="Fit image", command=lambda: self._canvas.fit_image()).grid(
            row=0, column=1, sticky="e", padx=(8, 0)
        )
        canvas_frame = ttk.Frame(viewer)
        canvas_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self._canvas = self._canvas_class(canvas_frame, size=target, on_box_change=self._on_box_change)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._canvas.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._canvas.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self._canvas.attach_scrollbars(horizontal, vertical)
        self._loading_overlay = ttk.Frame(canvas_frame, padding=24, relief="solid", borderwidth=1)
        self._loading_message = tk.StringVar(value="Loading…")
        ttk.Label(self._loading_overlay, textvariable=self._loading_message,
                  font=("TkDefaultFont", 13, "bold")).pack(pady=(0, 12))
        self._progress = ttk.Progressbar(self._loading_overlay, mode="indeterminate", length=220)
        self._progress.pack(fill="x")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(3, weight=1)

    def _populate_list(self) -> None:
        """Populate the image list from the available IDs."""
        self._listbox.delete(0, tk.END)
        for image_id in self._ids:
            self._listbox.insert(tk.END, image_id)

    # --- navigation / loading ------------------------------------------
    def _search_message(self, message: str, error: bool = False) -> None:
        """Show search feedback next to the input, highlighting invalid requests."""
        self._search_feedback.set(message)
        self._search_feedback_label.config(foreground="#b42318" if error else "#555")

    def _search(self) -> None:
        """Resolve an exact ID, JPEG filename, or unpadded numeric image ID."""
        query = self._search_id.get().strip()
        if query.lower().endswith(".jpg"):
            query = query[:-4]
        if not query:
            self._search_message("Enter an image number to search.", error=True)
            self._search_entry.focus_set()
            return
        if query in self._ids:
            self._select_index(self._ids.index(query))
            return
        # Accept 68 for 0068 without assuming that every filename has four digits.
        if query.isascii() and query.isdecimal():
            matches = [index for index, image_id in enumerate(self._ids)
                       if image_id.isascii() and image_id.isdecimal()
                       and int(image_id) == int(query)]
            if len(matches) == 1:
                self._select_index(matches[0])
                return
            if len(matches) > 1:
                self._search_message("Multiple matches; enter the exact image ID.", error=True)
                return
        self._search_message(f"Image {query} not found in data.", error=True)

    def _select_index(self, idx: int) -> None:
        """Select and reveal a list item, then request its image."""
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(idx)
        self._listbox.activate(idx)
        self._listbox.see(idx)
        self._search_message(f"Selected image {self._ids[idx]}.")
        self._load_selected()

    def _step(self, delta: int) -> None:
        """Move to the adjacent image without stepping beyond the available list."""
        if not self._ids:
            return
        sel = self._listbox.curselection()
        idx = (sel[0] if sel else 0) + delta
        idx = max(0, min(len(self._ids) - 1, idx))
        self._select_index(idx)

    def _load_selected(self) -> None:
        """Preserve edits and queue the latest selection, cancelling obsolete loads."""
        if self._closed.is_set():
            return
        sel = self._listbox.curselection()
        if not sel:
            return
        image_id = self._ids[sel[0]]
        if image_id == self._loading_id or (
            image_id == self._current_id and self._loading_id is None
        ):
            return

        self._stash_edits()
        self._load_request += 1
        if self._load_future is not None:
            self._load_future.cancel()
        request = self._load_request
        self._loading_id = image_id
        self._status.set("Loading…")
        self._load_text.set("")
        self._loading_message.set("Loading…")
        self._loading_overlay.place(relx=0.5, rely=0.5, anchor="center")
        self._loading_overlay.lift()
        self._progress.start(_LOADING_ANIMATION_INTERVAL_MS)
        self._save_btn.config(state="disabled")
        self._reset_btn.config(state="disabled")
        self._angle_scale.config(state="disabled")
        self._canvas.set_editing_enabled(False)

        def load() -> None:
            try:
                if self._closed.is_set() or request != self._load_request:
                    return
                preview, _ = self.backend.open(image_id)
                if self._closed.is_set() or request != self._load_request:
                    return
                record_error = None
                try:
                    rec = self.backend.existing_record(image_id)
                    if rec is not None and (
                        len(rec.obb_xywhr) != 5 or not all(math.isfinite(value) for value in rec.obb_xywhr)
                        or rec.obb_xywhr[2] <= 0 or rec.obb_xywhr[3] <= 0
                    ):
                        raise ValueError("Invalid saved box geometry")
                except Exception as exc:
                    rec = None
                    record_error = str(exc)
                match = None
                match_error = None
                if rec is None and not self._template_initialized:
                    self._load_updates.put((request, "Loading…"))
                    self._prepare_template()
                    self._load_updates.put((request, "Loading…"))
                if rec is None and self._template_matcher is not None:
                    if self._closed.is_set() or request != self._load_request:
                        return
                    try:
                        match = self._template_matcher.match(preview.preview)
                    except Exception as exc:
                        match_error = str(exc)
                result = _ImageLoadResult(
                    request=request, image_id=image_id, preview=preview, record=rec,
                    match=match, match_error=match_error, record_error=record_error,
                )
            except Exception as exc:  # deliver worker errors on the Tk thread
                result = _ImageLoadResult(request=request, image_id=image_id, error=exc)
            self._load_results.put(result)

        self._load_future = self._load_executor.submit(load)

    def _poll_load_results(self) -> None:
        """Apply current worker results on the Tk thread and ignore stale requests."""
        if self._closed.is_set():
            return
        self._poll_bulk_results()
        try:
            while True:
                request, message = self._load_updates.get_nowait()
                if request == self._load_request and self._loading_id is not None:
                    self._loading_message.set(message)
        except queue.Empty:
            pass
        try:
            while True:
                result = self._load_results.get_nowait()
                image_id = result.image_id
                if result.request != self._load_request:
                    continue  # a newer selection was made while this scan loaded
                self._loading_id = None
                self._progress.stop()
                self._loading_overlay.place_forget()
                self._angle_scale.config(state="normal")
                self._canvas.set_editing_enabled(True)
                if result.error is not None:
                    self._status.set(f"Load failed: {result.error}")
                    self._load_text.set(f"Could not load image {image_id}.")
                    if self._current_id is not None:
                        self._listbox.selection_clear(0, tk.END)
                        self._listbox.selection_set(self._ids.index(self._current_id))
                    self._on_box_change()
                    continue

                assert result.preview is not None
                self._preview = result.preview
                self._current_id = image_id
                self._image_title.set(
                    f"Image {image_id}  ·  {self._ids.index(image_id)+1:,} of {len(self._ids):,}"
                )
                self._load_text.set("Ready — review the box before saving.")
                self._canvas.show_preview(result.preview.preview)
                self._apply_record(image_id, result.record, result.match, result.match_error)
                if result.record_error is not None:
                    self._status.set(f"{image_id}: saved annotation unreadable ({result.record_error}). Review and Save to replace it.")
        except queue.Empty:
            pass
        if self.winfo_exists():
            self._poll_after_id = self.after(_POLL_INTERVAL_MS, self._poll_load_results)

    def _apply_record(
        self,
        image_id: str,
        rec: AnnotationRecord | None,
        match: TemplateMatch | None,
        match_error: str | None,
    ) -> None:
        """Set the saved or detected baseline, then restore any unsaved session edits."""
        self._initial_obb = None
        self._initial_box_source = "empty box"
        if rec is not None:
            cx, cy, w, h, angle = rec.obb_xywhr
            self._canvas.set_obb(self._box_class(cx, cy, w, h, angle))
            self._angle.set(math.degrees(angle))
            self._status.set(f"{image_id}: loaded existing annotation.")
            self._initial_box_source = "saved box"
        elif match is not None:
            self._canvas.set_obb(match.obb)
            self._angle.set(math.degrees(match.obb.angle))
            self._status.set(
                f"{image_id}: template found card "
                f"({match.inliers}/{match.good_matches} matches)."
            )
            self._initial_box_source = "template box"
        elif self._template_error is not None:
            self._angle.set(0.0)
            self._status.set(f"Template unavailable: {self._template_error}")
        elif match_error is not None:
            self._angle.set(0.0)
            self._status.set(f"Template match failed: {match_error}")
        else:
            self._angle.set(0.0)
            self._status.set(f"{image_id}: no template match; draw a box.")
        box = self._canvas.get_obb()
        self._initial_obb = replace(box) if box is not None else None
        self._committed_obb = replace(box) if box is not None else None
        if image_id in self._drafts:
            draft = self._drafts[image_id]
            self._canvas.set_obb(replace(draft) if draft is not None else None)
            self._angle.set(math.degrees(draft.angle) if draft is not None else 0.0)
            self._status.set(f"{image_id}: restored unsaved edits.")
        self._on_box_change()

    def _stash_edits(self) -> None:
        """Cache changed geometry for the current image without writing to disk."""
        if self._current_id is None:
            return
        box = self._canvas.get_obb()
        if box != self._committed_obb:
            self._drafts[self._current_id] = replace(box) if box is not None else None
        else:
            self._drafts.pop(self._current_id, None)

    def _close(self) -> None:
        """Warn before discarding drafts, cancel pending work, and destroy the window."""
        if self._closed.is_set():
            return
        if hasattr(self, "_canvas"):
            self._stash_edits()
        if self._drafts and not messagebox.askokcancel(
            "Unsaved changes", "There are unsaved edits. Close and discard them?", parent=self
        ):
            return
        self._closed.set()
        self._bulk_cancel.set()
        self.after_cancel(self._poll_after_id)
        self._progress.stop()
        self._load_executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    # --- bulk template save --------------------------------------------
    def _prepare_template(self) -> None:
        """Initialize the cached matcher once on the worker and retain any error."""
        if not self._template_initialized:
            try:
                from ..template_matcher import CardTemplateMatcher
                self._template_matcher = CardTemplateMatcher(self.backend.config.template_path)
            except Exception as exc:
                self._template_error = str(exc)
            finally:
                self._template_initialized = True

    def _open_bulk_dialog(self) -> None:
        """Open or raise the modal bulk-save dialog without starting a batch."""
        if self._bulk_dialog is not None and self._bulk_dialog.winfo_exists():
            self._bulk_dialog.lift()
            return
        dialog = self._bulk_dialog = tk.Toplevel(self)
        dialog.title("Save range")
        dialog.transient(self)
        dialog.grab_set()
        self._build_bulk_widgets(dialog)

    def _build_bulk_widgets(self, dialog: tk.Toplevel) -> None:
        """Build the batch range form, progress controls, and result details."""
        panel = ttk.Frame(dialog, padding=16)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Save range", font=("TkDefaultFont", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(panel, text="Each image uses its own template detection.\nExisting files and failed detections are skipped.\nCurrent manual edits are not used; save those first.",
                  wraplength=380).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 12))
        self._range_start = tk.StringVar(value=self._current_id if self._current_id and self._current_id.isdecimal() else "")
        self._range_end = tk.StringVar(value=self._range_start.get())
        self._range_entries = []
        for row, label, variable in ((2, "Start ID", self._range_start), (3, "End ID (inclusive)", self._range_end)):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(panel, textvariable=variable, width=16)
            entry.grid(row=row, column=1, sticky="ew", padx=(12, 0))
            self._range_entries.append(entry)
        self._bulk_message = tk.StringVar(value="Enter an inclusive numeric ID range, then click Save range.")
        ttk.Label(panel, textvariable=self._bulk_message, wraplength=380).grid(row=4, column=0, columnspan=2, sticky="w", pady=12)
        self._bulk_progress = ttk.Progressbar(panel, mode="determinate")
        self._bulk_progress.grid(row=5, column=0, columnspan=2, sticky="ew")
        self._bulk_start_btn = ttk.Button(panel, text="Save range", command=self._start_bulk)
        self._bulk_start_btn.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        self._bulk_cancel_btn = ttk.Button(panel, text="Close", command=self._cancel_or_close_bulk)
        self._bulk_cancel_btn.grid(row=6, column=1, sticky="ew", padx=(8, 0), pady=(12, 0))
        self._bulk_details_frame = ttk.Frame(panel)
        self._bulk_details_frame.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        ttk.Label(self._bulk_details_frame, text="Results by image ID").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self._bulk_details = ttk.Treeview(
            self._bulk_details_frame, columns=("id", "result", "reason"),
            show="headings", height=7,
        )
        for column, title, width in (("id", "Image ID", 85), ("result", "Result", 95), ("reason", "Reason", 260)):
            self._bulk_details.heading(column, text=title)
            self._bulk_details.column(column, width=width, minwidth=60)
        self._bulk_details.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self._bulk_details_frame, orient="vertical", command=self._bulk_details.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self._bulk_details.config(yscrollcommand=scrollbar.set)
        self._bulk_details_frame.columnconfigure(0, weight=1)
        self._bulk_details_frame.rowconfigure(1, weight=1)
        self._bulk_details_frame.grid_remove()
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(7, weight=1)
        dialog.protocol("WM_DELETE_WINDOW", self._cancel_or_close_bulk)

    def _start_bulk(self) -> None:
        """Validate an inclusive numeric range and queue sequential template saves."""
        if self._bulk_running:
            return
        try:
            selected, self._bulk_missing = ids_in_range(
                self._ids, self._range_start.get(), self._range_end.get()
            )
            if not selected:
                raise ValueError("No image files found in this range.")
        except ValueError as exc:
            self._bulk_message.set(str(exc))
            return
        self._bulk_cancel.clear()
        self._bulk_selected = selected
        self._bulk_bounds = (int(self._range_start.get()), int(self._range_end.get()))
        self._bulk_id_width = max(4, len(self._range_start.get().strip()), len(self._range_end.get().strip()))
        self._bulk_details.delete(*self._bulk_details.get_children())
        self._bulk_details_frame.grid_remove()
        self._bulk_running = True
        self._bulk_start_btn.config(state="disabled")
        for entry in self._range_entries:
            entry.config(state="disabled")
        self._bulk_cancel_btn.config(text="Cancel")
        self._bulk_progress.config(maximum=len(selected), value=0)
        self._bulk_message.set(f"Preparing {len(selected)} images; {self._bulk_missing} missing IDs skipped…")

        def run() -> None:
            try:
                if not self._bulk_cancel.is_set() and any(
                    not (self.backend.config.out_dir / f"{image_id}.json").exists()
                    for image_id in selected
                ):
                    self._prepare_template()
                    if self._template_matcher is None:
                        raise ValueError(self._template_error or "Template unavailable")
                summary = save_template_range(
                    self.backend, self._template_matcher, selected, self._bulk_cancel,
                    lambda progress: self._bulk_results.put(_BulkResult(done=False, summary=progress)),
                )
                self._bulk_results.put(_BulkResult(done=True, summary=summary))
            except Exception as exc:
                self._bulk_results.put(_BulkResult(done=True, error=str(exc)))
        # Use the image worker so the cached OpenCV matcher is never used concurrently.
        self._bulk_future = self._load_executor.submit(run)

    def _poll_bulk_results(self) -> None:
        """Update batch progress and completion details on the Tk thread."""
        try:
            while True:
                result = self._bulk_results.get_nowait()
                done, summary, error = result.done, result.summary, result.error
                if summary is not None:
                    self._bulk_progress.config(value=summary.completed)
                    state = ("Cancelled" if summary.cancelled else "Finished") if done else f"Image {summary.current_id}"
                    message = (f"{state}: {summary.completed}/{summary.total} processed.\n"
                               f"Saved {summary.saved}; existing {summary.existing}; "
                               f"no detection {summary.unmatched}; failed {summary.failed}.\n"
                               f"Missing IDs skipped: {self._bulk_missing}.")
                    if done and summary.saved:
                        message += "\nReview the generated boxes before using them."
                    elif done and not summary.cancelled:
                        message += "\nNo new annotation files were saved."
                    if summary.last_error:
                        message += f"\nLast error: {summary.last_error}"
                    self._bulk_message.set(message)
                if done:
                    self._bulk_progress.config(value=0)
                    self._show_bulk_details(summary)
                    self._bulk_running = False
                    self._bulk_start_btn.config(state="normal")
                    for entry in self._range_entries:
                        entry.config(state="normal")
                    self._bulk_cancel_btn.config(text="Close", state="normal")
                    if error is not None:
                        self._bulk_message.set(f"Bulk save failed: {error}")
        except queue.Empty:
            pass

    def _show_bulk_details(self, summary: BulkSummary | None) -> None:
        """List per-image outcomes, unprocessed IDs, and compact missing-ID ranges."""
        outcomes = summary.outcomes if summary is not None else []
        for values in outcomes:
            self._bulk_details.insert("", "end", values=values)
        processed = {values[0] for values in outcomes}
        for image_id in self._bulk_selected:
            if image_id not in processed:
                self._bulk_details.insert("", "end", values=(image_id, "Not processed", "Cancelled" if summary and summary.cancelled else "Batch could not finish"))
        # Represent long missing gaps as exact inclusive ranges, without iterating
        # every integer in a potentially very large requested range.
        low, high = self._bulk_bounds
        cursor = low
        present = sorted({int(image_id) for image_id in self._bulk_selected})
        for value in present + [high+1]:
            if cursor < value:
                first = f"{cursor:0{self._bulk_id_width}}"
                last = f"{value-1:0{self._bulk_id_width}}"
                label = first if cursor == value-1 else f"{first}–{last}"
                self._bulk_details.insert("", "end", values=(label, "Skipped", "Image file missing"))
            cursor = value+1
        self._bulk_details_frame.grid()

    def _cancel_or_close_bulk(self) -> None:
        """Request cancellation during a batch, or close an idle batch dialog."""
        if self._bulk_running:
            self._bulk_cancel.set()
            self._bulk_cancel_btn.config(state="disabled")
            self._bulk_message.set("Cancelling after the current operation; saved files will remain.")
        elif self._bulk_dialog is not None:
            self._bulk_dialog.destroy()
            self._bulk_dialog = None

    # --- box / angle / save --------------------------------------------
    def _on_angle(self) -> None:
        """Apply the slider angle in radians to the current editor box."""
        self._canvas.set_angle(math.radians(self._angle.get()))

    def _on_box_change(self) -> None:
        """Enable Save and Reset according to the current box and loading state."""
        box = self._canvas.get_obb()
        has_box = box is not None and self._loading_id is None
        self._save_btn.config(state="normal" if has_box else "disabled")
        changed = box != self._initial_obb and self._loading_id is None
        self._reset_btn.config(state="normal" if changed else "disabled")

    def _reset_box(self) -> None:
        """Restore the initial box and angle without changing its annotation file."""
        if self._loading_id is not None or self._current_id is None:
            return
        box = replace(self._initial_obb) if self._initial_obb is not None else None
        self._angle.set(math.degrees(box.angle) if box is not None else 0.0)
        self._canvas.set_obb(box)
        self._status.set(
            f"{self._current_id}: restored {self._initial_box_source}. Save to keep this box."
        )

    def _save(self) -> None:
        """Save only the displayed box, retaining edits if publication fails."""
        obb = self._canvas.get_obb()
        if (obb is None or self._preview is None or self._current_id is None
                or self._loading_id is not None):
            return
        try:
            path = self.backend.export(self._current_id, self._preview, obb)
        except Exception as exc:
            self._status.set(f"Save failed: {exc}. Your edits are still available.")
            return
        self._committed_obb = replace(obb)
        self._drafts.pop(self._current_id, None)
        self._status.set(f"Saved {path.name}")
