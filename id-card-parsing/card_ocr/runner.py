"""Run PaddleOCR on prepared crops and save text and recognition errors."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from PIL import Image

from .models import create_model
from .extraction import DEFAULT_RULES, apply_rules, extract_retry_value, extract_value, load_rules


def _recognize_crop(adapter, path: Path, *, binary=False) -> str:
    """Recognize a crop, optionally masking light paper texture for a numeric retry."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if binary:
            image = image.convert("L").point(lambda pixel: 255 if pixel > 140 else 0).convert("RGB")
        results = adapter.recognize_batch([image])
    if len(results) != 1:
        raise ValueError("Adapter must return one prediction per field")
    return results[0].text


def run_model(manifest_path: Path, model_name: str = "paddle", *, adapter=None, fields=None,
              rules_path: Path = DEFAULT_RULES, **options) -> dict:
    """Recognize each selected field, retaining individual failures.

    Setup errors become error rows for the selected fields.
    """
    if model_name != "paddle":
        raise ValueError("Only PaddleOCR is supported; choose paddle")
    rules = load_rules(rules_path)
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if fields:
        missing = set(fields) - {sample["field"] for sample in samples}
        if missing:
            raise ValueError(f"Fields absent from manifest: {sorted(missing)}")
        samples = [sample for sample in samples if sample["field"] in fields]
    seen = set()
    for sample in samples:
        key = (sample["image_id"], sample["field"])
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
        for key in ("crop", "raw_crop"):
            if key in sample and not (root / sample[key]).resolve().is_relative_to(root):
                raise ValueError(f"Crop path is outside the run directory: {sample[key]}")
    if not samples:
        raise ValueError("No prepared samples to recognize")
    load_error = None
    try:
        if adapter is None:
            adapter = create_model(model_name, **options)
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    predictions = []
    for index, sample in enumerate(samples, 1):
        print(f"{model_name}: {sample['image_id']}/{sample['field']} ({index}/{len(samples)})", flush=True)
        row = dict(image_id=sample["image_id"], field=sample["field"],
                   text=None, status="error", error=None)
        try:
            if load_error:
                raise RuntimeError(load_error)
            row.update(text=_recognize_crop(adapter, root / sample["crop"]), status="ok")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rule = rules["fields"].get(sample["field"])
        if (row["status"] == "ok" and row["text"].strip() and rule
                and rule["type"] in {"integer", "number", "time"}
                and extract_value(row["text"], rule) is None and sample.get("raw_crop")):
            try:
                retry = _recognize_crop(adapter, root / sample["raw_crop"])
                row.update(retry_text=retry, retry_crop=sample["raw_crop"])
                if extract_retry_value(row["text"], retry, rule) is None:
                    binary_retry = _recognize_crop(adapter, root / sample["raw_crop"], binary=True)
                    if extract_retry_value(row["text"], binary_retry, rule) is not None:
                        row.update(retry_text=binary_retry, retry_processing="binary_140")
            except Exception as exc:
                print(f"Raw crop retry failed for {sample['image_id']}/{sample['field']}: {exc}", flush=True)
        predictions.append(row)
    payload = dict(model=model_name, model_id=getattr(adapter, "model_id", None),
                   recipe=manifest.get("recipe"), manifest=str(manifest_path),
                   predictions=predictions)
    return apply_rules(payload, rules)


def save_predictions(payload: dict, path: Path, *, replace: bool = False) -> None:
    """Save raw OCR and structured JSON plus a CSV of extracted card records."""
    path = Path(path)
    csv_path = path.with_suffix(".csv")
    if not replace and (path.exists() or csv_path.exists()):
        raise FileExistsError(f"Predictions already exist: {path}; use --replace to rerun")
    rows = payload["data"]
    if not rows:
        raise ValueError("No extracted records to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".predictions-", dir=path.parent) as temporary:
        staging = Path(temporary)
        (staging / "new.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with (staging / "new.csv").open("w", encoding="utf-8", newline="") as handle:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        backups, published = [], []
        try:
            for target, source in [(path, staging / "new.json"), (csv_path, staging / "new.csv")]:
                backup = staging / ("old" + target.suffix)
                if target.exists():
                    target.rename(backup)
                    backups.append((target, backup))
                source.rename(target)
                published.append(target)
        except OSError:
            for target in published:
                target.unlink()
            for target, backup in backups:
                backup.rename(target)
            raise
