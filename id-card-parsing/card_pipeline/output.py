"""Atomic JSON and flat CSV output for a pipeline run."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

PREDICTIONS_JSON = "predictions.json"
PREDICTIONS_CSV = "predictions.csv"


def ensure_writable(output_dir: Path, *, replace: bool) -> None:
    """Fail before any work if predictions exist and replacement was not requested."""
    output_dir = Path(output_dir)
    existing = [output_dir / name for name in (PREDICTIONS_JSON, PREDICTIONS_CSV)
                if (output_dir / name).exists()]
    if existing and not replace:
        raise FileExistsError(f"Predictions already exist: {existing[0]}; use --replace to rerun")


def _cell(scan: dict, name: str):
    """Typed value when extraction ran, else the filtered text; blank on failure."""
    values = scan.get("values") or {}
    if name in values:
        return values[name]
    for field in scan.get("fields", []):
        if field["name"] == name and field.get("status") == "ok":
            return field.get("text")
    return None


def _rows(payload: dict) -> tuple[list[str], list[dict]]:
    columns = ["scan_id", "status", *payload["fields"], *payload.get("extra_columns", [])]
    rows = []
    for scan in payload["scans"]:
        row = {"scan_id": scan["scan_id"], "status": scan["status"]}
        for name in [*payload["fields"], *payload.get("extra_columns", [])]:
            value = _cell(scan, name)
            row[name] = "" if value is None else value
        rows.append(row)
    return columns, rows


def save_predictions(payload: dict, output_dir: Path, *, replace: bool = False) -> None:
    """Write predictions.json and predictions.csv together, or neither.

    Both files are staged beside the destination and swapped in; if the swap
    fails midway the previous files are restored.
    """
    output_dir = Path(output_dir)
    ensure_writable(output_dir, replace=replace)
    columns, rows = _rows(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [(output_dir / PREDICTIONS_JSON), (output_dir / PREDICTIONS_CSV)]
    with tempfile.TemporaryDirectory(prefix=".predictions-", dir=output_dir) as temporary:
        staging = Path(temporary)
        (staging / "new.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with (staging / "new.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        sources = [staging / "new.json", staging / "new.csv"]
        backups, published = [], []
        try:
            for target, source in zip(targets, sources):
                if target.exists():
                    backup = staging / ("old" + target.suffix)
                    target.rename(backup)
                    backups.append((target, backup))
                source.rename(target)
                published.append(target)
        except OSError:
            for target in published:
                target.unlink(missing_ok=True)
            for target, backup in backups:
                backup.rename(target)
            raise
