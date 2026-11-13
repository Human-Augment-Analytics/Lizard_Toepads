"""Run scans through detect -> normalize -> template crop -> recognize.

Failures are isolated per scan and per field and recorded as statuses; only
setup problems (template, rules, weights, overwrite, unknown ids) abort the
run, and they do so before any scan is processed.
"""

from __future__ import annotations

from pathlib import Path

from annotator.image_loader import ImageLoadError
from card_ocr.extraction import extract_records, load_rules
from card_ocr.fields import Field
from card_ocr.images import select_images
from card_ocr.prepare import draw_fields
from template_builder.normalize import rectify_to_square
from template_builder.template_record import TemplateRecord, load_record

import numpy as np

from .align import CardAligner
from .config import PipelineConfig
from .corners import canonicalize_corners
from .cropper import crop_template
from .detector import Detector, YoloObbDetector, select_best
from .dictionary import apply_dictionary
from .output import ensure_writable
from .recognizer import FieldRecognizer, PaddleFieldRecognizer
from .scan import load_scan

_SCAN_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PAYLOAD_VERSION = 1


def _load_template(path: Path) -> TemplateRecord:
    """Load and validate the template, failing before any scan is touched."""
    try:
        record = load_record(path)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid template {path}: missing or malformed {exc}") from exc
    if record is None:
        raise FileNotFoundError(f"Template not found: {path}")
    record.validate()
    return record


def _card_aspect(record: TemplateRecord) -> float:
    """Width / height of the template's card, from its saved crop corners."""
    tl, tr, br, bl = np.asarray(record.card_obb_corners, float)
    width = (np.hypot(*(tr - tl)) + np.hypot(*(br - bl))) / 2
    height = (np.hypot(*(bl - tl)) + np.hypot(*(br - tr))) / 2
    return float(width / height)


def _scan_paths(data_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(Path(data_dir).iterdir())
            if p.is_file() and p.suffix.lower() in _SCAN_SUFFIXES}


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _process_fields(scan_id, normalized, record, recognizer, scan_dir, output_dir):
    """Crop every template region, then recognize field crops or defer checkboxes."""
    crops_dir = scan_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    fields = []
    for spec, crop, crop_error in crop_template(normalized, record):
        entry = {"name": spec.name, "crop_type": spec.crop_type, "status": "ok", "crop": None}
        if crop is not None:
            crop_path = crops_dir / f"{spec.name}.png"
            crop.save(crop_path)
            entry["crop"] = crop_path.relative_to(output_dir).as_posix()
        if crop_error is not None:
            entry.update(status="error", error=f"crop: {crop_error}")
        elif spec.is_checkbox:
            entry["status"] = "deferred"       # v1: cropped and saved, not inspected
        else:
            entry.update(datatype=spec.datatype, dictionary=spec.dictionary)
            try:
                raw = recognizer.recognize(crop)
            except Exception as exc:           # one field must not sink the scan
                entry.update(status="error", error=_error_text(exc))
            else:
                entry.update(raw_text=raw, text=apply_dictionary(raw, spec.dictionary))
        fields.append(entry)
    return fields


def _typed_values(scan_id, fields, rules):
    """Apply card_ocr extraction rules to recognized text, if rules were given."""
    rows = [dict(image_id=scan_id, field=f["name"], text=f.get("text"),
                 status="ok" if f["status"] == "ok" else "error")
            for f in fields if f["status"] != "deferred"]
    if not rows:
        return None
    record = extract_records(rows, rules)[0]
    return {key: value for key, value in record.items() if key != "image_id"}


def _process_scan(scan_id, path, config, record, size, detector, recognizer, aligner,
                  rules, output_dir):
    scan = {"scan_id": scan_id, "status": "ok", "error": None, "detection": None,
            "alignment": None, "fields": [], "values": None}
    try:
        loaded = load_scan(path, detect_size=config.detect_size, decode_size=config.decode_size)
    except ImageLoadError as exc:
        return {**scan, "status": "unreadable_scan", "error": str(exc)}
    try:
        detections = detector.detect(loaded.preview)
    except Exception as exc:
        return {**scan, "status": "detector_error", "error": _error_text(exc)}
    best = select_best(detections, config.confidence)
    if best is None:
        return {**scan, "status": "no_card",
                "error": f"no detection at or above confidence {config.confidence}"}
    try:
        corners = canonicalize_corners(best.corners, aspect=_card_aspect(record))
    except ValueError as exc:
        return {**scan, "status": "degenerate_obb", "error": str(exc)}
    scan["detection"] = {"confidence": best.confidence,
                         "corners": loaded.preview_to_original(corners).tolist()}
    try:
        normalized = rectify_to_square(loaded.decoded, loaded.preview_to_decoded(corners), size,
                                       pad_mode=record.pad_mode, fill=record.fill)
    except Exception as exc:
        return {**scan, "status": "rectify_failed", "error": _error_text(exc)}

    scan_dir = output_dir / "scans" / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)
    card = normalized.image
    if aligner is not None:
        card.save(scan_dir / "card_unaligned.png")
        try:
            aligned = aligner.align(card)
        except ValueError as exc:
            return {**scan, "status": "alignment_failed", "error": str(exc)}
        card = aligned.image
        scan["alignment"] = {
            "rotation_degrees": round(aligned.rotation_degrees, 2),
            "inliers": aligned.diagnostics.get("inliers"),
            "median_error_px": aligned.diagnostics.get("median_error_px"),
        }
    card.save(scan_dir / "card.png")
    draw_fields(card, [Field(spec.name, spec.box) for spec in record.fields]
                ).save(scan_dir / "overlay.png")
    scan["fields"] = _process_fields(scan_id, card, record, recognizer, scan_dir, output_dir)
    if rules is not None:
        scan["values"] = _typed_values(scan_id, scan["fields"], rules)
    return scan


def _summary(scans: list[dict]) -> dict:
    statuses = [s["status"] for s in scans]
    field_statuses = [f["status"] for s in scans for f in s["fields"]]
    counted = {"ok", "no_card", "degenerate_obb", "unreadable_scan", "alignment_failed"}
    return {
        "scans": len(scans),
        "normalized": statuses.count("ok"),
        "no_card": statuses.count("no_card"),
        "degenerate_obb": statuses.count("degenerate_obb"),
        "unreadable_scan": statuses.count("unreadable_scan"),
        "alignment_failed": statuses.count("alignment_failed"),
        # detector_error, rectify_failed, and anything else not counted above.
        "other_scan_failures": sum(status not in counted for status in statuses),
        "fields_recognized": sum(s == "ok" for s in field_statuses),
        "fields_deferred": field_statuses.count("deferred"),
        "field_failures": field_statuses.count("error"),
    }


def run_pipeline(config: PipelineConfig, *, detector: Detector | None = None,
                 recognizer: FieldRecognizer | None = None, aligner=None,
                 scan_ids: list[str] | None = None,
                 log=lambda message: print(message, flush=True)) -> dict:
    """Process scans and return the predictions payload (see ``output.save_predictions``).

    ``detector``, ``recognizer``, and ``aligner`` default to the real YOLO-OBB,
    PaddleOCR, and form-registration components; tests inject fakes. The
    aligner is used only when ``config.align`` is set. Crops and overlays are
    written under ``config.output_dir/scans/<scan_id>/``.
    """
    record = _load_template(config.template_path)
    rules = load_rules(config.rules_path) if config.rules_path else None
    output_dir = config.output_dir
    ensure_writable(output_dir, replace=config.replace)
    ids = select_images(config.data_dir, ids=scan_ids, count=10**9, require_count=False)
    paths = _scan_paths(config.data_dir)
    size = config.canvas_size or record.canvas_size
    if not config.align:
        aligner = None
    elif aligner is None:
        if not config.template_image.is_file():
            raise FileNotFoundError(f"Template image not found: {config.template_image}")
        aligner = CardAligner.from_template_image(config.template_image, size, fill=record.fill)
    if detector is None:
        detector = YoloObbDetector(config.weights, device=config.device)
    if recognizer is None:
        recognizer = PaddleFieldRecognizer()

    scans = []
    for index, scan_id in enumerate(ids, 1):
        scan = _process_scan(scan_id, paths[scan_id], config, record, size,
                             detector, recognizer, aligner, rules, output_dir)
        scans.append(scan)
        log(f"[{index}/{len(ids)}] {scan_id}: {scan['status']}"
            + (f" ({scan['error']})" if scan["error"] else ""))

    extra = ["date"] if rules is not None and "date" in rules else []
    return {
        "version": PAYLOAD_VERSION,
        "weights": str(config.weights),
        "template": str(config.template_path),
        "canvas_size": size,
        "aligned": aligner is not None,
        "model_id": getattr(recognizer, "model_id", None),
        "confidence": config.confidence,
        "fields": [spec.name for spec in record.fields],
        "extra_columns": extra,
        "scans": scans,
        "summary": _summary(scans),
    }
