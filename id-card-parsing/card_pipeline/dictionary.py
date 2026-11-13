"""Per-field character constraints applied after recognition.

A template ``dictionary`` is either a named set (``numeric``, ``alphanumeric``,
``alpha``, ``all``) or an explicit allow-list of literal characters. ``all``,
an empty string, or None means no constraint. The constraint filters the
recognized string; callers keep the raw text so a reviewer can see what changed.
"""

from __future__ import annotations

import string

_DIGITS = string.digits
_ALPHA = string.ascii_letters

# None means "no constraint".
NAMED_SETS: dict[str, str | None] = {
    "numeric": _DIGITS,
    "alphanumeric": _DIGITS + _ALPHA,
    "alpha": _ALPHA,
    "all": None,
}


def resolve_dictionary(spec: str | None) -> set[str] | None:
    """Return the allowed character set, or None when the field is unconstrained."""
    if spec is None or spec == "":
        return None
    name = spec.strip().lower()
    if name in NAMED_SETS:
        chars = NAMED_SETS[name]
        return None if chars is None else set(chars)
    if not name:
        # Whitespace-only spec is a literal allow-list of whitespace, not a name.
        return set(spec)
    return set(spec)


def apply_dictionary(text: str, spec: str | None) -> str:
    """Keep only characters the dictionary allows, preserving their order."""
    allowed = resolve_dictionary(spec)
    if allowed is None:
        return text
    return "".join(char for char in text if char in allowed)
