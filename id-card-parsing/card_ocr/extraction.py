"""Parse raw OCR into field values using the configured types and patterns."""

from __future__ import annotations

from datetime import date, datetime
import json
import math
from pathlib import Path
import re

DEFAULT_RULES = Path(__file__).resolve().parent.parent / "template/ocr_rules.json"


def load_rules(path: Path = DEFAULT_RULES) -> dict:
    """Reject invalid extraction configuration before starting recognition."""
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or set(rules) - {"fields", "date"}:
        raise ValueError("Extraction rules must contain fields and an optional date definition")
    fields = rules.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Extraction rules need a nonempty fields mapping")
    for field, rule in fields.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", field) or not isinstance(rule, dict):
            raise ValueError(f"Invalid extraction field: {field}")
        if set(rule) - {"type", "pattern", "min", "max", "case", "input_formats", "format", "remove_characters", "retry_decimal_aliases"}:
            raise ValueError(f"Unknown extraction setting for {field}")
        if (not isinstance(rule.get("type"), str)
                or rule["type"] not in {"integer", "number", "time", "text"}):
            raise ValueError(f"Unknown extraction type for {field}")
        if not isinstance(rule.get("pattern"), str) or not rule["pattern"]:
            raise ValueError(f"A nonempty pattern is required for {field}")
        try:
            pattern = re.compile(rule["pattern"])
        except re.error as exc:
            raise ValueError(f"Invalid pattern for {field}: {exc}") from exc
        if pattern.match("") is not None:
            raise ValueError(f"Pattern must not match empty text: {field}")
        for bound in ("min", "max"):
            value = rule.get(bound)
            if bound in rule and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError(f"{bound} must be a finite number for {field}")
        if rule.get("min", -math.inf) > rule.get("max", math.inf):
            raise ValueError(f"min exceeds max for {field}")
        if (not isinstance(rule.get("case", "preserve"), str)
                or rule.get("case", "preserve") not in {"preserve", "upper", "lower"}):
            raise ValueError(f"Invalid case setting for {field}")
        if not isinstance(rule.get("remove_characters", ""), str):
            raise ValueError(f"remove_characters must be text for {field}")
        aliases = rule.get("retry_decimal_aliases", [])
        if (not isinstance(aliases, list) or any(not isinstance(alias, str) or len(alias) != 1
                or alias.isalnum() or alias.isspace() for alias in aliases)):
            raise ValueError(f"retry_decimal_aliases must contain single punctuation characters: {field}")
        if rule["type"] == "time":
            formats = rule.get("input_formats")
            if not isinstance(formats, list) or not formats or any(
                    not isinstance(value, str) or value not in {"%H:%M", "%H.%M", "%H%M"} for value in formats):
                raise ValueError(f"Unsupported input time format for {field}")
            if (not isinstance(rule.get("format"), str)
                    or rule["format"] not in {"%H:%M", "%H.%M", "%H%M"}):
                raise ValueError(f"Unsupported output time format for {field}")
    if "date" in rules:
        definition = rules["date"]
        if (not isinstance(definition, dict) or set(definition) != {"day", "month", "year", "format", "century"}
                or any(not isinstance(definition[key], str) or definition[key] not in fields
                       for key in ("day", "month", "year"))
                or type(definition["century"]) is not int
                or not 100 <= definition["century"] <= 9900 or definition["century"] % 100
                or not isinstance(definition["format"], str)
                or definition["format"] not in {"%d/%m/%y", "%d-%m-%y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"}):
            raise ValueError("Date rules need day/month/year field names and a supported format")
    return rules


def extract_value(text: str, rule: dict | None):
    """Return one unambiguous typed value; blanks or invalid values return None."""
    text = text.strip()
    if not text:
        return None
    if rule is None:
        return text
    matches = list(re.finditer(rule["pattern"], text, re.DOTALL))
    if len(matches) != 1:
        return None
    token = matches[0].group().strip()
    token = token.translate(str.maketrans("", "", rule.get("remove_characters", "")))
    kind = rule["type"]
    try:
        if kind == "integer":
            value = int(token)
        elif kind == "number":
            value = float(token)
        elif kind == "time":
            for source_format in rule["input_formats"]:
                try:
                    return datetime.strptime(token, source_format).strftime(rule["format"])
                except ValueError:
                    continue
            return None
        else:
            case = rule.get("case", "preserve")
            return token.upper() if case == "upper" else token.lower() if case == "lower" else token
    except (ValueError, OverflowError):
        return None
    if ((isinstance(value, float) and not math.isfinite(value))
            or not rule.get("min", -math.inf) <= value <= rule.get("max", math.inf)):
        return None
    return value


def extract_retry_value(primary: str, retry: str, rule: dict | None):
    """Parse a retry without accepting unchanged ambiguous text or lost digits."""
    if (rule and rule["type"] == "number"
            and sum(char.isdigit() for char in retry) < sum(char.isdigit() for char in primary)):
        return None
    value = extract_value(retry, rule)
    if value is not None or rule is None or rule["type"] != "number" or retry == primary:
        return value
    normalized = retry
    for alias in rule.get("retry_decimal_aliases", []):
        normalized = re.sub(r"(?<=[0-9])" + re.escape(alias) + r"(?=[0-9])", ".", normalized)
    return extract_value(normalized, rule) if normalized != retry else None


def extract_records(predictions: list[dict], rules: dict) -> list[dict]:
    """Build one record per card and combine valid date components without guessing."""
    records, seen = {}, set()
    for row in predictions:
        key = (row["image_id"], row["field"])
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        record = records.setdefault(row["image_id"], {"image_id": row["image_id"]})
        rule = rules["fields"].get(row["field"])
        value = extract_value(row["text"], rule) if row["status"] == "ok" else None
        if value is None and row["status"] == "ok" and row.get("retry_text"):
            value = extract_retry_value(row["text"], row["retry_text"], rule)
        record[row["field"]] = value
    if definition := rules.get("date"):
        for record in records.values():
            record["date"] = None
            components = [record.get(definition[key]) for key in ("year", "month", "day")]
            if all(type(value) is int for value in components):
                try:
                    year, month, day = components
                    calendar_year = year + definition["century"] if 0 <= year <= 99 else year
                    record["date"] = date(calendar_year, month, day).strftime(definition["format"])
                except ValueError:
                    pass
    return list(records.values())


def apply_rules(payload: dict, rules: dict) -> dict:
    """Add structured records and their rules while preserving raw prediction rows."""
    return {**payload, "extraction_rules": rules, "data": extract_records(payload["predictions"], rules)}
