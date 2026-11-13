import csv
import json

from card_pipeline.__main__ import main
from tests.test_pipeline_runner import (
    DETECT,
    FakeDetector,
    FakeRecognizer,
    _detection,
    _template,
    _write_scan,
)


def _setup(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_scan(data / "0001.jpg")
    return data, _template(tmp_path), _detection(data / "0001.jpg")


def _argv(tmp_path, *extra):
    return ["run", "--weights", str(tmp_path / "w.pt"),
            "--template", str(tmp_path / "template.json"),
            "--data-dir", str(tmp_path / "data"),
            "--output", str(tmp_path / "out"),
            "--detect-size", str(DETECT), "--decode-size", "1024", "--no-align", *extra]


def test_run_writes_predictions_and_reports_summary(tmp_path, capsys):
    _data, _template_path, detection = _setup(tmp_path)
    code = main(_argv(tmp_path), detector=FakeDetector([detection]),
                recognizer=FakeRecognizer())
    assert code == 0
    payload = json.loads((tmp_path / "out" / "predictions.json").read_text("utf-8"))
    assert payload["scans"][0]["status"] == "ok"
    with (tmp_path / "out" / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["scan_id"] == "0001" and row["day"] == "red"
    output = capsys.readouterr().out
    assert "1 scans" in output and "Saved predictions" in output


def test_failures_give_nonzero_exit_but_still_save(tmp_path):
    _setup(tmp_path)
    code = main(_argv(tmp_path), detector=FakeDetector([]), recognizer=FakeRecognizer())
    assert code == 1
    payload = json.loads((tmp_path / "out" / "predictions.json").read_text("utf-8"))
    assert payload["scans"][0]["status"] == "no_card"


def test_canvas_size_and_ids_flags_are_honored(tmp_path):
    data, _t, detection = _setup(tmp_path)
    _write_scan(data / "0002.jpg")
    code = main(_argv(tmp_path, "--canvas-size", "256", "--ids", "1"),
                detector=FakeDetector([detection]), recognizer=FakeRecognizer())
    payload = json.loads((tmp_path / "out" / "predictions.json").read_text("utf-8"))
    assert code == 0
    assert payload["canvas_size"] == 256
    assert [s["scan_id"] for s in payload["scans"]] == ["0001"]     # "1" zero-filled


def test_setup_errors_exit_one_with_message(tmp_path, capsys):
    _setup(tmp_path)
    code = main(_argv(tmp_path), recognizer=FakeRecognizer())     # weights file absent
    assert code == 1
    assert "Error:" in capsys.readouterr().err


def test_existing_predictions_require_replace(tmp_path, capsys):
    _data, _t, detection = _setup(tmp_path)
    assert main(_argv(tmp_path), detector=FakeDetector([detection]),
                recognizer=FakeRecognizer()) == 0
    again = main(_argv(tmp_path), detector=FakeDetector([detection]), recognizer=FakeRecognizer())
    assert again == 1 and "--replace" in capsys.readouterr().err
    assert main(_argv(tmp_path, "--replace"), detector=FakeDetector([detection]),
                recognizer=FakeRecognizer()) == 0
