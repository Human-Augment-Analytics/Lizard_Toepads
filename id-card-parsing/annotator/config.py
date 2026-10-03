"""Configuration for the annotator.

Defaults resolve relative to the ``id-card-parsing/`` directory (the parent of
this package) so the app works when launched from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# id-card-parsing/  (parent of the annotator package)
_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AnnotatorConfig:
    data_dir: Path = _ROOT / "data"
    out_dir: Path = _ROOT / "annotations"
    target_resolution: int = 1024

    def __post_init__(self) -> None:
        if self.target_resolution <= 0:
            raise ValueError("target_resolution must be positive")
