"""End-to-end runner tests with fake detector/recognizer (no weights, no Paddle)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from annotator.geometry import apply_affine, build_transform
from annotator.image_loader import ImageLoader
from card_pipeline.config import PipelineConfig
from card_pipeline.detector import Detection, WeightsLoadError
from card_pipeline.output import save_predictions
from card_pipeline.runner import run_pipeline
from template_builder.template_record import FieldSpec, TemplateRecord, save_record

DETECT = 512
# Card quad in original scan pixels (TL, TR, BR, BL): 1200 x 700 inside 1600 x 1000.
CARD = np.array([[200, 150], [1400, 150], [1400, 850], [200, 850]], float)


def _write_scan(path):
    image = Image.new("RGB", (1600, 1000), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 150, 799, 849), fill=(220, 30, 30))      # left half red
    draw.rectangle((800, 150, 1399, 849), fill=(30, 30, 220))     # right half blue
    image.save(path, "JPEG", quality=95)


def _preview_quad(path, quad):
    preview = ImageLoader(DETECT).load(path)
    return apply_affine(build_transform(preview.scale, preview.offset), quad)


class FakeDetector:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _detection(path, quad=CARD, confidence=0.9, order=(0, 1, 2, 3)):
    corners = _preview_quad(path, quad)[list(order)]
    return Detection(corners=corners, confidence=confidence, area=1.0)


class FakeRecognizer:
    """Scripted outcomes in call order; falls back to naming the crop's color."""

    model_id = "fake-recognizer"

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes) if outcomes is not None else None
        self.calls = 0

    def recognize(self, crop):
        self.calls += 1
        if self.outcomes is not None:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        red, _green, blue = np.asarray(crop.convert("RGB")).reshape(-1, 3).mean(axis=0)
        return "red" if red > blue else "blue"


def _template(tmp_path, day_dictionary="alpha"):
    record = TemplateRecord(
        canvas_size=512, pad_mode="solid", fill=(255, 255, 255),
        card_obb_corners=CARD.tolist(),
        fields=[
            FieldSpec("day", (0.05, 0.4, 0.2, 0.6), "integer", day_dictionary),
            FieldSpec("perch", (0.8, 0.4, 0.95, 0.6), "text", None),
            FieldSpec("alive", (0.45, 0.4, 0.55, 0.6), crop_type="checkbox"),
        ])
    return save_record(record, tmp_path / "template.json")


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_scan(data / "0001.jpg")
    return tmp_path


def _config(workspace, **overrides):
    # Synthetic two-color scans have no printed-form features to register, so
    # alignment is off by default here and exercised with injected aligners.
    options = dict(weights=workspace / "w.pt",
                   data_dir=workspace / "data", output_dir=workspace / "out",
                   detect_size=DETECT, decode_size=1024, align=False)
    options.update(overrides)
    # Write the default template only if the test did not supply its own; both
    # use the same path, so an eager default would clobber a custom one.
    options.setdefault("template_path", None)
    if options["template_path"] is None:
        options["template_path"] = _template(workspace)
    return PipelineConfig(**options)


def _run(workspace, detector, recognizer=None, scan_ids=None, aligner=None, **overrides):
    return run_pipeline(_config(workspace, **overrides), detector=detector,
                        recognizer=recognizer or FakeRecognizer(), aligner=aligner,
                        scan_ids=scan_ids, log=lambda message: None)


class FakeAligner:
    """Rotates every card 180 degrees, standing in for a registration that found a flip."""

    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def align(self, card):
        from card_pipeline.align import AlignResult
        self.calls += 1
        if self.error:
            raise self.error
        return AlignResult(card.rotate(180), 180.0, {"inliers": 99, "median_error_px": 0.5})


def _good_detector(workspace, order=(0, 1, 2, 3), count=1):
    path = workspace / "data" / "0001.jpg"
    return FakeDetector(*[[_detection(path, order=order)] for _ in range(count)])


# --- alignment and routing --------------------------------------------------

def test_crops_land_where_the_template_says(workspace):
    payload = _run(workspace, _good_detector(workspace))
    scan = payload["scans"][0]
    assert scan["status"] == "ok"
    by_name = {f["name"]: f for f in scan["fields"]}
    assert by_name["day"]["raw_text"] == "red"       # left-half box
    assert by_name["perch"]["raw_text"] == "blue"    # right-half box


def test_shuffled_detector_corner_order_still_aligns(workspace):
    payload = _run(workspace, _good_detector(workspace, order=(2, 0, 3, 1)))
    by_name = {f["name"]: f for f in payload["scans"][0]["fields"]}
    assert by_name["day"]["raw_text"] == "red"
    assert by_name["perch"]["raw_text"] == "blue"


def test_checkbox_is_cropped_and_deferred_but_never_recognized(workspace):
    recognizer = FakeRecognizer()
    payload = _run(workspace, _good_detector(workspace), recognizer)
    scan = payload["scans"][0]
    alive = next(f for f in scan["fields"] if f["name"] == "alive")
    assert alive["status"] == "deferred" and alive["crop_type"] == "checkbox"
    assert (workspace / "out" / alive["crop"]).is_file()
    assert recognizer.calls == 2          # day + perch only


def test_field_order_follows_the_template(workspace):
    payload = _run(workspace, _good_detector(workspace))
    assert [f["name"] for f in payload["scans"][0]["fields"]] == ["day", "perch", "alive"]
    assert payload["fields"] == ["day", "perch", "alive"]


def test_artifacts_are_saved(workspace):
    _run(workspace, _good_detector(workspace))
    scan_dir = workspace / "out" / "scans" / "0001"
    assert (scan_dir / "card.png").is_file()
    assert (scan_dir / "overlay.png").is_file()
    assert (scan_dir / "crops" / "day.png").is_file()


# --- dictionary -------------------------------------------------------------

def test_dictionary_filters_text_and_keeps_raw(workspace):
    payload = _run(workspace, _good_detector(workspace), FakeRecognizer(["I3x", "free text"]),
                   template_path=_template(workspace, day_dictionary="numeric"))
    by_name = {f["name"]: f for f in payload["scans"][0]["fields"]}
    assert by_name["day"]["raw_text"] == "I3x"
    assert by_name["day"]["text"] == "3"
    assert by_name["day"]["dictionary"] == "numeric"
    assert by_name["perch"]["text"] == "free text"      # no dictionary -> unchanged


def test_everything_filtered_is_an_empty_value_not_a_failure(workspace):
    payload = _run(workspace, _good_detector(workspace), FakeRecognizer(["abc", "ok"]),
                   template_path=_template(workspace, day_dictionary="numeric"))
    day = payload["scans"][0]["fields"][0]
    assert day["status"] == "ok" and day["text"] == "" and day["raw_text"] == "abc"


# --- canvas override --------------------------------------------------------

def test_default_canvas_comes_from_the_template(workspace):
    payload = _run(workspace, _good_detector(workspace))
    assert payload["canvas_size"] == 512
    with Image.open(workspace / "out" / "scans" / "0001" / "card.png") as card:
        assert card.size == (512, 512)


def test_canvas_override_gives_larger_aligned_crops(workspace):
    payload = _run(workspace, _good_detector(workspace), canvas_size=1024)
    assert payload["canvas_size"] == 1024
    by_name = {f["name"]: f for f in payload["scans"][0]["fields"]}
    assert by_name["day"]["raw_text"] == "red" and by_name["perch"]["raw_text"] == "blue"
    with Image.open(workspace / "out" / "scans" / "0001" / "crops" / "day.png") as crop:
        assert crop.size == (154, 206)       # 0.15 x 0.2 of 1024, rounded outward


# --- typed extraction -------------------------------------------------------

def test_rules_produce_typed_values(workspace):
    rules = workspace / "rules.json"
    rules.write_text(json.dumps({"fields": {"day": {
        "type": "integer", "pattern": "[0-9]{1,2}", "min": 1, "max": 31}}}), encoding="utf-8")
    payload = _run(workspace, _good_detector(workspace), FakeRecognizer(["13", "x"]),
                   template_path=_template(workspace, day_dictionary="numeric"),
                   rules_path=rules)
    scan = payload["scans"][0]
    assert scan["values"]["day"] == 13
    assert scan["fields"][0]["text"] == "13"


# --- failure isolation ------------------------------------------------------

def test_recognition_failure_is_isolated_to_one_field(workspace):
    payload = _run(workspace, _good_detector(workspace),
                   FakeRecognizer([RuntimeError("boom"), "fine"]))
    scan = payload["scans"][0]
    day, perch = scan["fields"][0], scan["fields"][1]
    assert scan["status"] == "ok"
    assert day["status"] == "error" and "RuntimeError: boom" in day["error"]
    assert perch["status"] == "ok" and perch["text"] == "fine"
    assert payload["summary"]["field_failures"] == 1


def test_no_detection_does_not_stop_later_scans(workspace):
    _write_scan(workspace / "data" / "0002.jpg")
    second = _detection(workspace / "data" / "0002.jpg")
    payload = _run(workspace, FakeDetector([], [second]))
    statuses = {s["scan_id"]: s["status"] for s in payload["scans"]}
    assert statuses == {"0001": "no_card", "0002": "ok"}
    assert payload["scans"][0]["fields"] == []
    assert payload["summary"]["no_card"] == 1


def test_low_confidence_counts_as_no_card(workspace):
    path = workspace / "data" / "0001.jpg"
    payload = _run(workspace, FakeDetector([_detection(path, confidence=0.1)]))
    assert payload["scans"][0]["status"] == "no_card"


def test_degenerate_detection_is_recorded(workspace):
    line = Detection(corners=np.array([[10, 10], [20, 20], [30, 30], [40, 40]], float),
                     confidence=0.9, area=0.0)
    payload = _run(workspace, FakeDetector([line]))
    assert payload["scans"][0]["status"] == "degenerate_obb"
    assert payload["summary"]["degenerate_obb"] == 1


def test_detector_exception_is_isolated(workspace):
    payload = _run(workspace, FakeDetector(RuntimeError("model fell over")))
    scan = payload["scans"][0]
    assert scan["status"] == "detector_error" and "model fell over" in scan["error"]


def test_unreadable_scan_does_not_stop_the_run(workspace):
    (workspace / "data" / "0000.jpg").write_bytes(b"not an image")
    payload = _run(workspace, _good_detector(workspace))
    statuses = {s["scan_id"]: s["status"] for s in payload["scans"]}
    assert statuses == {"0000": "unreadable_scan", "0001": "ok"}


def test_summary_counts(workspace):
    _write_scan(workspace / "data" / "0002.jpg")
    path = workspace / "data" / "0001.jpg"
    payload = _run(workspace, FakeDetector([_detection(path)], []))
    summary = payload["summary"]
    assert summary["scans"] == 2 and summary["normalized"] == 1 and summary["no_card"] == 1
    assert summary["fields_recognized"] == 2 and summary["fields_deferred"] == 1


# --- selection and fail-fast ------------------------------------------------

def test_scan_ids_limit_the_run(workspace):
    _write_scan(workspace / "data" / "0002.jpg")
    detector = FakeDetector([_detection(workspace / "data" / "0002.jpg")])
    payload = _run(workspace, detector, scan_ids=["0002"])
    assert [s["scan_id"] for s in payload["scans"]] == ["0002"]
    assert detector.calls == 1


def test_unknown_scan_id_fails_before_processing(workspace):
    detector = FakeDetector()
    with pytest.raises(ValueError):
        _run(workspace, detector, scan_ids=["9999"])
    assert detector.calls == 0


def test_missing_template_fails_fast(workspace):
    with pytest.raises(FileNotFoundError):
        _run(workspace, FakeDetector(), template_path=workspace / "absent.json")


def test_invalid_template_fails_fast(workspace):
    bad = workspace / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        _run(workspace, FakeDetector(), template_path=bad)


def test_unloadable_weights_fail_before_any_scan(workspace):
    config = _config(workspace)
    with pytest.raises(WeightsLoadError):
        run_pipeline(config, recognizer=FakeRecognizer(), log=lambda message: None)


def test_existing_predictions_block_the_run_before_work(workspace):
    save_predictions({"fields": [], "scans": []}, workspace / "out")
    detector = FakeDetector()
    with pytest.raises(FileExistsError):
        _run(workspace, detector)
    assert detector.calls == 0


def test_replace_allows_rerun(workspace):
    save_predictions({"fields": [], "scans": []}, workspace / "out")
    payload = _run(workspace, _good_detector(workspace), replace=True)
    assert payload["scans"][0]["status"] == "ok"


def test_payload_is_json_serializable(workspace):
    payload = _run(workspace, _good_detector(workspace))
    json.dumps(payload)


# --- alignment --------------------------------------------------------------

def test_aligned_card_is_what_gets_cropped(workspace):
    # The fake aligner flips the card, so the left/right colors swap.
    aligner = FakeAligner()
    payload = _run(workspace, _good_detector(workspace), aligner=aligner, align=True,
                   template_image=workspace / "unused.jpg")
    scan = payload["scans"][0]
    by_name = {f["name"]: f for f in scan["fields"]}
    assert by_name["day"]["raw_text"] == "blue" and by_name["perch"]["raw_text"] == "red"
    assert scan["alignment"] == {"rotation_degrees": 180.0, "inliers": 99, "median_error_px": 0.5}
    assert (workspace / "out" / "scans" / "0001" / "card_unaligned.png").is_file()


def test_alignment_failure_is_recorded_and_skips_crops(workspace):
    recognizer = FakeRecognizer()
    payload = _run(workspace, _good_detector(workspace), recognizer,
                   aligner=FakeAligner(ValueError("Insufficient printed form matches")),
                   align=True, template_image=workspace / "unused.jpg")
    scan = payload["scans"][0]
    assert scan["status"] == "alignment_failed" and "Insufficient" in scan["error"]
    assert scan["fields"] == [] and recognizer.calls == 0
    assert (workspace / "out" / "scans" / "0001" / "card_unaligned.png").is_file()
    assert payload["summary"]["alignment_failed"] == 1


def test_missing_template_image_fails_before_any_scan(workspace):
    detector = FakeDetector()
    with pytest.raises(OSError):
        _run(workspace, detector, align=True, template_image=workspace / "absent.jpg")
    assert detector.calls == 0


def test_corners_follow_the_template_card_aspect(workspace, monkeypatch):
    # The template card is landscape (1200 x 700), so canonicalization is asked for that.
    import card_pipeline.runner as runner
    seen = []
    real = runner.canonicalize_corners
    monkeypatch.setattr(runner, "canonicalize_corners",
                        lambda points, aspect=None: seen.append(aspect) or real(points, aspect=aspect))
    _run(workspace, _good_detector(workspace))
    assert seen == [pytest.approx(1200 / 700)]
