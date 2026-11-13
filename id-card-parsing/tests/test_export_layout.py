import json

from card_ocr.fields import load_checkboxes, load_layout

from template_builder.export_layout import (
    to_checkboxes_layout,
    to_fields_layout,
    to_ocr_rules,
)
from template_builder.template_record import FieldSpec, TemplateRecord


def _record():
    return TemplateRecord(
        canvas_size=512,
        pad_mode="solid",
        fill=(255, 255, 255),
        card_obb_corners=[[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]],
        fields=[
            FieldSpec("day", (0.218, 0.143, 0.373, 0.198), "integer", "0123456789"),
            FieldSpec("perch", (0.40, 0.55, 0.72, 0.60), "text", None),
            FieldSpec("svl", (0.10, 0.70, 0.30, 0.75), "number", None),
        ],
    )


def _mixed_record():
    return TemplateRecord(
        canvas_size=512,
        pad_mode="solid",
        fill=(255, 255, 255),
        card_obb_corners=[[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]],
        fields=[
            FieldSpec("day", (0.218, 0.143, 0.373, 0.198), "integer", "0123456789"),
            FieldSpec("alive", (0.80, 0.10, 0.88, 0.16), crop_type="checkbox"),
            FieldSpec("gravid", (0.80, 0.20, 0.88, 0.26), crop_type="checkbox"),
        ],
    )


def test_fields_layout_shape():
    layout = to_fields_layout(_record())
    assert set(layout) >= {"fields"}
    names = [item["name"] for item in layout["fields"]]
    assert names == ["day", "perch", "svl"]
    day = layout["fields"][0]
    assert day["box"] == [0.218, 0.143, 0.373, 0.198]


def test_fields_layout_loads_via_card_ocr(tmp_path):
    # The whole point of Req 6.3: card_ocr.fields.load_layout must accept it.
    layout = to_fields_layout(_record())
    path = tmp_path / "fields.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    loaded = load_layout(path)
    assert [f.name for f in loaded] == ["day", "perch", "svl"]
    assert loaded[0].box == (0.218, 0.143, 0.373, 0.198)


def test_fields_layout_excludes_checkboxes():
    # Only OCR fields belong in the fields layout.
    layout = to_fields_layout(_mixed_record())
    names = [item["name"] for item in layout["fields"]]
    assert names == ["day"]


def test_checkboxes_layout_shape():
    layout = to_checkboxes_layout(_mixed_record())
    names = [item["name"] for item in layout["checkboxes"]]
    assert names == ["alive", "gravid"]
    assert layout["checkboxes"][0]["box"] == [0.80, 0.10, 0.88, 0.16]


def test_checkboxes_layout_loads_via_card_ocr(tmp_path):
    layout = to_checkboxes_layout(_mixed_record())
    path = tmp_path / "checkboxes.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    loaded = load_checkboxes(path)
    assert [c.name for c in loaded] == ["alive", "gravid"]


def test_ocr_rules_excludes_checkboxes():
    rules = to_ocr_rules(_mixed_record())
    assert set(rules["fields"]) == {"day"}


def test_ocr_rules_type_per_field():
    rules = to_ocr_rules(_record())
    assert rules["fields"]["day"]["type"] == "integer"
    assert rules["fields"]["perch"]["type"] == "text"
    assert rules["fields"]["svl"]["type"] == "number"


def test_ocr_rules_carries_dictionary_when_present():
    rules = to_ocr_rules(_record())
    # Dictionary is carried through for a later OCR stage to resolve (Req 6.4).
    assert rules["fields"]["day"].get("dictionary") == "0123456789"
    assert "dictionary" not in rules["fields"]["perch"] or rules["fields"]["perch"]["dictionary"] is None
