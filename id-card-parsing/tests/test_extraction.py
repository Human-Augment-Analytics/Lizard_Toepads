"""Cover typed extraction, raw-text preservation, and prediction exports."""

import copy
import csv
import json

import pytest
from PIL import Image

from card_ocr.extraction import apply_rules, extract_records, extract_retry_value, extract_value, load_rules
from card_ocr.models import Prediction
from card_ocr.runner import run_model, save_predictions


@pytest.fixture
def rules():
    """Use the checked-in extraction configuration."""
    return load_rules()


@pytest.mark.parametrize("field,text,expected", [
    ("time", "9.45", "09:45"),
    ("time", "Time: 9:51", "09:51"),
    ("time", "0945", "09:45"),
    ("time", "1333", "13:33"),
    ("time", "2400", None),
    ("time", "24:00", None),
    ("time", "9:45:30", None),
    ("time", "9:45 or 10:02", None),
    ("day", "13", 13),
    ("day", "32", None),
    ("month", "05", 5),
    ("year", "26", 26),
    ("year", "6", 6),
    ("year", "1999", None),
    ("year", "2.6", 26),
    ("day", "1.3", 13),
    ("month", "0.5", 5),
    ("year", "2.026", None),
    ("year", "123456", None),
    ("mass", "mass: 4.87", 4.87),
    ("mass", "2.37", 2.37),
    ("mass", "-4.87", None),
    ("mass", "2.11 / 4.87", None),
    ("tb", "3-1", None),
    ("tb", "-3.1", -3.1),
    ("perch", "trunk", "TRUNK"),
    ("recapture_id", "L180.", "L180"),
    ("flag", "B10", "B10"),
    ("initials", "jts", "JTS"),
    ("initials", "x . y.", "X.Y."),
    ("initials", "X.Y.", "X.Y."),
    ("initials", "S . C .", "S.C."),
    ("svl", "", None),
])
def test_typed_values(rules, field, text, expected):
    """Normalize explicit values without choosing among ambiguous candidates."""
    assert extract_value(text, rules["fields"][field]) == expected


def test_date_components_and_raw_text(rules):
    """Combine real calendar dates and leave original OCR rows unchanged."""
    rows = [dict(image_id="0010", field=field, text=text, status="ok", error=None)
            for field, text in [("day", " 13 "), ("month", "05"), ("year", "26")]]
    payload = {"predictions": rows}
    original = copy.deepcopy(payload)
    result = apply_rules(payload, rules)
    assert result["data"][0]["date"] == "13/05/26"
    assert result["data"][0]["month"] == 5
    assert result["data"][0]["year"] == 26
    assert payload == original
    assert result["predictions"] == original["predictions"]
    rows[0]["text"], rows[1]["text"] = "31", "02"
    assert extract_records(rows, rules)[0]["date"] is None
    rows[0]["status"], rows[0]["text"] = "error", None
    assert extract_records(rows, rules)[0]["day"] is None


def test_configurable_formats_and_bounds(rules):
    """Take output formats and numeric bounds from JSON rather than defaults."""
    rules["fields"]["time"]["format"] = "%H.%M"
    rules["fields"]["mass"]["max"] = 3
    assert extract_value("9:45", rules["fields"]["time"]) == "09.45"
    assert extract_value("4.87", rules["fields"]["mass"]) is None


@pytest.mark.parametrize("setting,value", [
    ("pattern", "["), ("pattern", ".*"), ("type", "unknown"),
    ("min", "one"), ("min", True), ("case", "guess"),
])
def test_invalid_rules_rejected(tmp_path, rules, setting, value):
    """Fail before recognition when a rule has an invalid setting."""
    rules["fields"]["mass"][setting] = value
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules))
    with pytest.raises(ValueError):
        load_rules(path)


def test_recognition_and_exports(tmp_path):
    """Save per-card extracted CSV data while retaining raw OCR in JSON."""
    class Adapter:
        """Supply recognizer text without loading a model."""

        model_id = "fixture"

        def recognize_batch(self, images):
            """Return a time in the handwritten decimal notation."""
            return [Prediction(" 9.45 ")]

    Image.new("RGB", (20, 20), "white").save(tmp_path / "time.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [
        dict(image_id="0010", field="time", crop="time.png")]}))
    payload = run_model(manifest, adapter=Adapter())
    assert payload["predictions"][0]["text"] == " 9.45 "
    assert payload["data"] == [dict(image_id="0010", time="09:45", date=None)]
    output = tmp_path / "results.json"
    save_predictions(payload, output)
    assert json.loads(output.read_text())["predictions"][0]["text"] == " 9.45 "
    with output.with_suffix(".csv").open() as handle:
        assert next(csv.DictReader(handle)) == dict(image_id="0010", time="09:45", date="")
    with pytest.raises(FileExistsError):
        save_predictions(payload, output)


def test_bad_rules_do_not_load_model(tmp_path, monkeypatch):
    """Configuration errors must not initialize Paddle or overwrite saved output."""
    path = tmp_path / "rules.json"
    path.write_text('{"fields": {}}')

    def fail_if_loaded(*args, **kwargs):
        """Make accidental model initialization fail the test."""
        pytest.fail("Model loaded before configuration validation")

    monkeypatch.setattr("card_ocr.runner.create_model", fail_if_loaded)
    with pytest.raises(ValueError):
        run_model(tmp_path / "manifest.json", rules_path=path)


@pytest.mark.parametrize("retry_text,expected", [("30.1", 30.1), ("30-1", 30.1), ("3-1", None)])
def test_invalid_number_retries_raw_crop(tmp_path, retry_text, expected):
    """Recover digits through recognition of the raw crop, never by inventing them."""
    class Adapter:
        """Return different results for cleaned and raw crop inputs."""

        def __init__(self):
            """Track the two recognition attempts."""
            self.responses = iter(["3-1", retry_text, retry_text])

        def recognize_batch(self, images):
            """Return the next observed OCR result."""
            return [Prediction(next(self.responses))]

    for name in ("tb.png", "tb_raw.png"):
        Image.new("RGB", (20, 20), "white").save(tmp_path / name)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [dict(
        image_id="0014", field="tb", crop="tb.png", raw_crop="tb_raw.png")]}))
    payload = run_model(manifest, adapter=Adapter())
    assert payload["predictions"][0]["text"] == "3-1"
    assert payload["predictions"][0]["retry_text"] == retry_text
    assert payload["data"][0]["tb"] == expected
    save_predictions(payload, tmp_path / "results.json")


def test_primary_value_wins_over_retry(rules):
    """Keep valid primary text rather than replacing it with an alternate reading."""
    rows = [dict(image_id="0014", field="tb", text="30.5", retry_text="30.1", status="ok")]
    assert extract_records(rows, rules)[0]["tb"] == 30.5


@pytest.mark.parametrize("primary,retry,expected", [
    ("3-1", "30-1", 30.1), ("3-1", "3-1", None), ("30-1", "3-1", None), ("30-1", "3.1", None),
    ("3-1", "-30-1", -30.1),
])
def test_decimal_aliases_require_a_changed_retry(rules, primary, retry, expected):
    """Normalize retry decimal marks without inserting digits or changing signs."""
    assert extract_retry_value(primary, retry, rules["fields"]["tb"]) == expected


def test_binary_retry_recovers_visible_zero(tmp_path):
    """Use actual recognizer output from a masked crop to recover a missing digit."""
    class Adapter:
        """Emulate the observed cleaned, raw, and binary crop readings."""

        def __init__(self):
            """Retain the sequence observed on card 0014."""
            self.responses = iter(["3-1", "3-1", "30-1"])

        def recognize_batch(self, images):
            """Return the next crop reading."""
            return [Prediction(next(self.responses))]

    for name in ("tb.png", "tb_raw.png"):
        Image.new("RGB", (20, 20), "white").save(tmp_path / name)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [dict(
        image_id="0014", field="tb", crop="tb.png", raw_crop="tb_raw.png")]}))
    payload = run_model(manifest, adapter=Adapter())
    assert payload["predictions"][0]["text"] == "3-1"
    assert payload["predictions"][0]["retry_text"] == "30-1"
    assert payload["predictions"][0]["retry_processing"] == "binary_140"
    assert payload["data"][0]["tb"] == 30.1
