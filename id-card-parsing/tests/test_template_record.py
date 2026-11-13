import json

import pytest

from template_builder.template_record import (
    FieldSpec,
    TemplateRecord,
    load_record,
    save_record,
)


def _field(name="day", box=(0.1, 0.1, 0.3, 0.2), datatype="integer", dictionary=None,
           crop_type="field"):
    return FieldSpec(name=name, box=box, datatype=datatype, dictionary=dictionary,
                     crop_type=crop_type)


def _record(fields=None):
    return TemplateRecord(
        canvas_size=512,
        pad_mode="solid",
        fill=(255, 255, 255),
        card_obb_corners=[[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]],
        fields=fields if fields is not None else [_field()],
    )


# --- FieldSpec validation ---

def test_fieldspec_accepts_valid():
    spec = _field(name="svl_mm", datatype="number", dictionary="0123456789.")
    assert spec.name == "svl_mm"
    assert spec.datatype == "number"
    assert spec.dictionary == "0123456789."


def test_fieldspec_optional_dictionary_defaults_none():
    assert _field().dictionary is None


@pytest.mark.parametrize("bad", ["Day", "1day", "day-1", "day ", "", "_day"])
def test_fieldspec_rejects_bad_name(bad):
    with pytest.raises(ValueError):
        _field(name=bad)


@pytest.mark.parametrize("box", [
    (0.3, 0.1, 0.1, 0.2),   # left >= right
    (0.1, 0.5, 0.3, 0.2),   # top >= bottom
    (-0.1, 0.1, 0.3, 0.2),  # left < 0
    (0.1, 0.1, 1.2, 0.2),   # right > 1
    (0.1, 0.1, 0.3),        # wrong length
])
def test_fieldspec_rejects_bad_box(box):
    with pytest.raises(ValueError):
        _field(box=box)


def test_fieldspec_rejects_unknown_datatype():
    with pytest.raises(ValueError):
        _field(datatype="boolean")


def test_fieldspec_rejects_non_str_dictionary():
    with pytest.raises(ValueError):
        _field(dictionary=123)


# --- crop_type ---

def test_fieldspec_crop_type_defaults_to_field():
    spec = FieldSpec(name="day", box=(0.1, 0.1, 0.3, 0.2), datatype="integer")
    assert spec.crop_type == "field"


def test_fieldspec_accepts_checkbox_crop_type():
    spec = _field(name="alive", datatype="text", crop_type="checkbox")
    assert spec.crop_type == "checkbox"


def test_fieldspec_checkbox_ignores_datatype_requirement():
    # A checkbox crop is not OCR'd, so an unset/irrelevant datatype is fine.
    spec = FieldSpec(name="alive", box=(0.1, 0.1, 0.2, 0.2), crop_type="checkbox")
    assert spec.crop_type == "checkbox"


def test_fieldspec_rejects_unknown_crop_type():
    with pytest.raises(ValueError):
        _field(crop_type="radio")


def test_field_crop_type_requires_valid_datatype():
    # When crop_type is 'field', a valid OCR datatype is still required.
    with pytest.raises(ValueError):
        FieldSpec(name="day", box=(0.1, 0.1, 0.2, 0.2), datatype="boolean", crop_type="field")


def test_from_json_backfills_missing_crop_type(tmp_path):
    # A legacy template written before crop_type existed must load as 'field'.
    legacy = {
        "canvas_size": 512,
        "pad_mode": "solid",
        "fill": [255, 255, 255],
        "card_obb_corners": [[0, 0], [100, 0], [100, 60], [0, 60]],
        "fields": [
            {"name": "day", "box": [0.2, 0.2, 0.4, 0.3], "datatype": "integer", "dictionary": "0123456789"}
        ],
    }
    record = TemplateRecord.from_json(json.dumps(legacy))
    assert record.fields[0].crop_type == "field"


# --- TemplateRecord validation ---

def test_record_rejects_empty_fields():
    with pytest.raises(ValueError):
        _record(fields=[]).validate()


def test_record_rejects_duplicate_names():
    dupe = [_field(name="day"), _field(name="day", box=(0.4, 0.4, 0.6, 0.5))]
    with pytest.raises(ValueError):
        _record(fields=dupe).validate()


def test_record_rejects_tiny_canvas():
    rec = TemplateRecord(canvas_size=1, pad_mode="solid", fill=(0, 0, 0),
                         card_obb_corners=[[0, 0], [1, 0], [1, 1], [0, 1]], fields=[_field()])
    with pytest.raises(ValueError):
        rec.validate()


def test_valid_record_passes_validate():
    _record().validate()  # should not raise


# --- Serialization ---

def test_json_round_trip():
    rec = _record(fields=[
        _field(name="day", datatype="integer", dictionary="0123456789"),
        _field(name="perch", box=(0.4, 0.5, 0.7, 0.6), datatype="text", dictionary=None),
    ])
    restored = TemplateRecord.from_json(rec.to_json())
    assert restored == rec


def test_to_json_is_valid_json_with_expected_keys():
    payload = json.loads(_record().to_json())
    assert set(payload) == {"canvas_size", "pad_mode", "fill", "card_obb_corners", "fields"}
    assert set(payload["fields"][0]) == {"name", "box", "datatype", "dictionary", "crop_type"}


def test_save_and_load_round_trip(tmp_path):
    rec = _record()
    path = tmp_path / "nested" / "field_template.json"
    saved = save_record(rec, path)
    assert saved == path
    assert path.is_file()
    assert load_record(path) == rec


def test_load_missing_returns_none(tmp_path):
    assert load_record(tmp_path / "absent.json") is None


def test_save_atomic_no_partial_on_existing(tmp_path):
    path = tmp_path / "field_template.json"
    save_record(_record(), path)
    # Overwrite with a different record; result must be the new one, fully written.
    other = _record(fields=[_field(name="mass", datatype="number")])
    save_record(other, path, overwrite=True)
    assert load_record(path) == other


def test_save_without_overwrite_rejects_existing(tmp_path):
    path = tmp_path / "field_template.json"
    save_record(_record(), path)
    with pytest.raises(FileExistsError):
        save_record(_record(), path, overwrite=False)
