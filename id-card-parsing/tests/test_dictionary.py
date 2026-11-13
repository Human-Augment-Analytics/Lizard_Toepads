import pytest

from card_pipeline.dictionary import NAMED_SETS, apply_dictionary, resolve_dictionary


@pytest.mark.parametrize("spec", [None, "", "all", "ALL", " all "])
def test_unconstrained_specs_resolve_to_none(spec):
    assert resolve_dictionary(spec) is None


def test_numeric_named_set():
    assert resolve_dictionary("numeric") == set("0123456789")


def test_alpha_and_alphanumeric_named_sets():
    alpha = resolve_dictionary("alpha")
    assert "a" in alpha and "Z" in alpha and "1" not in alpha
    alnum = resolve_dictionary("alphanumeric")
    assert alpha < alnum and "7" in alnum


def test_named_set_lookup_is_case_insensitive():
    assert resolve_dictionary("Numeric") == resolve_dictionary("numeric")


def test_other_strings_are_literal_allow_lists():
    assert resolve_dictionary("0123456789/.-") == set("0123456789/.-")


def test_literal_allow_list_keeps_whitespace():
    assert " " in resolve_dictionary("0123456789 ")


def test_named_sets_table_has_the_documented_names():
    assert set(NAMED_SETS) == {"numeric", "alphanumeric", "alpha", "all"}


def test_apply_filters_to_allowed_characters():
    assert apply_dictionary("I3o5", "numeric") == "35"
    assert apply_dictionary("12.5 mm", "0123456789.") == "12.5"


def test_apply_leaves_text_alone_when_unconstrained():
    assert apply_dictionary("A-1 /x", None) == "A-1 /x"
    assert apply_dictionary("A-1 /x", "all") == "A-1 /x"


def test_apply_can_filter_everything_out():
    assert apply_dictionary("abc", "numeric") == ""


def test_apply_preserves_character_order():
    assert apply_dictionary("a1b2c3", "alpha") == "abc"
