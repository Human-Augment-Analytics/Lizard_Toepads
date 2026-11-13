import csv
import json

import pytest

from card_pipeline.output import PREDICTIONS_CSV, PREDICTIONS_JSON, ensure_writable, save_predictions


def _payload():
    return {
        "version": 1,
        "fields": ["day", "perch", "alive"],
        "extra_columns": ["date"],
        "scans": [
            {"scan_id": "0001", "status": "ok", "error": None,
             "fields": [
                 {"name": "day", "status": "ok", "text": "13"},
                 {"name": "perch", "status": "error", "text": None, "error": "boom"},
                 {"name": "alive", "status": "deferred", "text": None},
             ],
             "values": {"day": 13, "date": "13/05/26"}},
            {"scan_id": "0002", "status": "no_card", "error": "no detection",
             "fields": [], "values": None},
        ],
        "summary": {"scans": 2},
    }


def test_json_round_trips(tmp_path):
    save_predictions(_payload(), tmp_path)
    assert json.loads((tmp_path / PREDICTIONS_JSON).read_text("utf-8")) == _payload()


def test_csv_has_one_row_per_scan_and_a_column_per_field(tmp_path):
    save_predictions(_payload(), tmp_path)
    with (tmp_path / PREDICTIONS_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["scan_id", "status", "day", "perch", "alive", "date"]
    assert len(rows) == 2


def test_csv_prefers_typed_value_then_text_and_blanks_failures(tmp_path):
    save_predictions(_payload(), tmp_path)
    with (tmp_path / PREDICTIONS_CSV).open(encoding="utf-8", newline="") as handle:
        first, second = list(csv.DictReader(handle))
    assert first["day"] == "13" and first["date"] == "13/05/26"
    assert first["perch"] == "" and first["alive"] == ""
    assert second["status"] == "no_card" and second["day"] == ""


def test_text_used_when_no_typed_value(tmp_path):
    payload = _payload()
    payload["scans"][0]["values"] = None
    payload["scans"][0]["fields"][0]["text"] = "I3"
    save_predictions(payload, tmp_path)
    with (tmp_path / PREDICTIONS_CSV).open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["day"] == "I3"


def test_refuses_to_clobber_without_replace(tmp_path):
    save_predictions(_payload(), tmp_path)
    with pytest.raises(FileExistsError):
        save_predictions(_payload(), tmp_path)
    with pytest.raises(FileExistsError):
        ensure_writable(tmp_path, replace=False)


def test_replace_overwrites_and_leaves_no_temp_files(tmp_path):
    save_predictions(_payload(), tmp_path)
    changed = _payload()
    changed["summary"] = {"scans": 99}
    save_predictions(changed, tmp_path, replace=True)
    assert json.loads((tmp_path / PREDICTIONS_JSON).read_text("utf-8"))["summary"] == {"scans": 99}
    assert sorted(p.name for p in tmp_path.iterdir()) == [PREDICTIONS_CSV, PREDICTIONS_JSON]


def test_ensure_writable_allows_fresh_or_replace(tmp_path):
    ensure_writable(tmp_path / "new", replace=False)
    save_predictions(_payload(), tmp_path)
    ensure_writable(tmp_path, replace=True)
