import json

import pytest

from card_pipeline.__main__ import main
from card_pipeline.evaluation import evaluate_predictions


def test_evaluate_saved_run_includes_unavailable_values_in_accuracy(tmp_path, capsys):
    predictions = {
        "fields": ["day", "marked"],
        "scans": [
            {"scan_id": "0001", "status": "ok", "values": {"day": 12}, "fields": [
                {"name": "day", "crop_type": "field", "status": "ok", "text": "012"},
                {"name": "marked", "crop_type": "checkbox", "status": "ok", "checked": False},
            ]},
            {"scan_id": "0002", "status": "ok", "fields": [
                {"name": "marked", "crop_type": "checkbox", "status": "ok", "checked": None},
            ]},
            {"scan_id": "0003", "status": "alignment_failed", "fields": []},
        ],
    }
    labels = {
        "0001": {"day": 12, "marked": True},
        "0002": {"day": None, "marked": False},
        "0003": {"day": ""},
        "0004": {"day": "17"},
    }
    predictions_path, labels_path = tmp_path / "predictions.json", tmp_path / "labels.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    assert main(["evaluate", "--predictions", str(predictions_path),
                 "--labels", str(labels_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["text"]["accuracy"] == 1 / 3
    assert report["text"]["failed"] == report["text"]["missing"] == 1
    assert report["checkbox"]["false_negative"] == report["checkbox"]["uncertain"] == 1
    assert report["checkbox"]["labeled"] == 2
    assert [issue["reason"] for issue in report["issues"]] == [
        None, "checkbox_uncertain", "alignment_failed", "scan_missing",
    ]


def test_evaluate_rejects_unverified_only_labels():
    with pytest.raises(ValueError, match="at least one verified value"):
        evaluate_predictions({"fields": ["day"], "scans": []}, {"0001": {"day": None}})


def test_rules_scoring_matches_numeric_labels_and_verified_blanks():
    texts = {"day": "12", "mass": "59.07", "id": "0012", "svl": "*",
             "failed_blank": "", "unavailable": ""}
    predictions = {
        "fields": list(texts),
        "scans": [{
            "scan_id": "0001",
            "status": "ok",
            "values": {"day": 12, "mass": 59.07, "id": "0012", "svl": None,
                       "failed_blank": None, "unavailable": None},
            "fields": [
                {"name": name, "crop_type": "field", "text": value,
                 "status": "error" if name == "failed_blank" else "ok"}
                for name, value in texts.items() if name != "unavailable"
            ],
        }],
    }
    labels = {"0001": {**texts, "id": "12", "svl": ""}}
    report = evaluate_predictions(predictions, labels)
    assert report["text"]["correct"] == 3
    assert [(issue["field"], issue["outcome"]) for issue in report["issues"]] == [
        ("id", "incorrect"), ("failed_blank", "failed"), ("unavailable", "missing"),
    ]
