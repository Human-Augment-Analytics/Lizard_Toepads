"""Score saved card predictions against manually verified field values."""

import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _matches_value(actual, expected, *, typed):
    if typed and type(actual) in (int, float) and isinstance(expected, str):
        try:
            numeric_label = Decimal(expected)
        except InvalidOperation:
            return False
        return numeric_label.is_finite() and numeric_label == Decimal(str(actual))
    return actual == expected


def _compare(scan, field, name, expected):
    """Return the outcome, delivered value, and reason for an unavailable value."""
    if scan is None:
        return "missing", None, "scan_missing"
    if scan["status"] != "ok":
        return "failed", None, scan["status"]
    if field is not None:
        if field["status"] != "ok":
            return "failed", None, field["status"]
        if field["crop_type"] == "checkbox":
            actual = field.get("checked")
            if actual is None:
                return "uncertain", None, "checkbox_uncertain"
            return ("correct" if actual == expected else "incorrect"), actual, None
    values = scan.get("values") or {}
    typed = name in values
    actual = values.get(name, field.get("text") if field is not None else None)
    if actual is None:
        if typed and field is not None and expected == "":
            return "correct", None, None
        return "missing", None, "value_missing"
    outcome = "correct" if _matches_value(actual, expected, typed=typed) else "incorrect"
    return outcome, actual, None


def _metrics(counts):
    outcomes = {key: counts[key] for key in ("correct", "incorrect", "missing", "failed", "uncertain")}
    total = sum(outcomes.values())
    return {"labeled": total, **outcomes, "accuracy": counts["correct"] / total if total else None}


def evaluate_predictions(predictions: dict, labels: dict) -> dict:
    """Score every verified value, including those on missing or failed scans."""
    if not isinstance(labels, dict):
        raise ValueError("Labels must be a JSON object keyed by scan ID")
    scans = {scan["scan_id"]: scan for scan in predictions["scans"]}
    if len(scans) != len(predictions["scans"]):
        raise ValueError("Predictions contain duplicate scan IDs")
    known_fields = set(predictions["fields"]) | set(predictions.get("extra_columns", []))
    crop_types = {
        field["name"]: field["crop_type"]
        for scan in scans.values()
        for field in scan["fields"]
    }
    totals = {"text": Counter(), "checkbox": Counter()}
    field_counts = {}
    checkbox_counts = Counter(dict.fromkeys(
        ("true_positive", "true_negative", "false_positive", "false_negative"), 0))
    issues = []
    cards_labeled = cards_correct = 0
    for scan_id, expected_fields in labels.items():
        if not isinstance(expected_fields, dict):
            raise ValueError(f"Labels for scan {scan_id} must be an object keyed by field name")
        scan = scans.get(scan_id)
        fields = {field["name"]: field for field in scan["fields"]} if scan is not None else {}
        outcomes = []
        for name, expected in expected_fields.items():
            if name not in known_fields:
                raise ValueError(f"Unknown field {name!r} in labels for scan {scan_id}")
            if expected is None:
                continue
            if type(expected) not in (str, int, float, bool):
                raise ValueError(f"Invalid label for {scan_id}/{name}: expected a string, number, or boolean")
            kind = "checkbox" if isinstance(expected, bool) else "text"
            crop_type = crop_types.get(name)
            if crop_type is not None and (crop_type == "checkbox") != (kind == "checkbox"):
                raise ValueError(f"Label type does not match prediction field {scan_id}/{name}")
            outcome, actual, reason = _compare(scan, fields.get(name), name, expected)
            totals[kind][outcome] += 1
            field_counts.setdefault(name, Counter())[outcome] += 1
            outcomes.append(outcome)
            if kind == "checkbox" and outcome in ("correct", "incorrect"):
                result = "true" if outcome == "correct" else "false"
                polarity = "positive" if actual else "negative"
                checkbox_counts[f"{result}_{polarity}"] += 1
            if outcome != "correct":
                issues.append({"scan_id": scan_id, "field": name, "outcome": outcome,
                               "expected": expected, "predicted": actual, "reason": reason})
        if outcomes:
            cards_labeled += 1
            cards_correct += all(outcome == "correct" for outcome in outcomes)
    if not cards_labeled:
        raise ValueError("Add at least one verified value before evaluating")
    return {
        "cards": {"labeled": cards_labeled, "all_labeled_fields_correct": cards_correct},
        "text": _metrics(totals["text"]),
        "checkbox": {**_metrics(totals["checkbox"]), **checkbox_counts},
        "fields": {name: _metrics(counts) for name, counts in sorted(field_counts.items())},
        "issues": issues,
    }


def evaluate_command(predictions_path: Path, labels_path: Path) -> int:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    print(json.dumps(evaluate_predictions(predictions, labels), indent=2, ensure_ascii=False))
    return 0
