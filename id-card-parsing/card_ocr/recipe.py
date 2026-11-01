"""Fingerprint crop settings so old inputs are not mistaken for corrected ones."""

import hashlib
import json
from pathlib import Path

# Bump when alignment or cleanup behavior changes without a layout change.
PIPELINE_VERSION = 'registered-form-print-mask-v3'


def preparation_recipe(template: Path, fields, *, card_width=1000, decode_size=4096,
                       clean_print=True, checkboxes=()) -> str:
    """Identify the template, text/checkbox regions, margins, and preprocessing."""
    payload = dict(version=PIPELINE_VERSION,
                   template_sha256=hashlib.sha256(Path(template).read_bytes()).hexdigest(),
                   fields=[dict(name=f.name, box=f.box, margin=f.margin, review_note=f.review_note)
                           for f in sorted(fields, key=lambda f: f.name)],
                   checkboxes=[dict(name=f.name, box=f.box, margin=f.margin)
                               for f in sorted(checkboxes, key=lambda f: f.name)],
                   card_width=card_width, decode_size=decode_size, clean_print=clean_print)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
