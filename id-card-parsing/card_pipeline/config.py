"""Run configuration for the card OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Inputs for one pipeline run.

    ``canvas_size`` overrides the template's canvas size. Template boxes are
    fractions, so any size aligns; larger sizes keep more pixels per field.
    """

    weights: Path
    template_path: Path
    data_dir: Path
    output_dir: Path
    confidence: float = 0.25
    canvas_size: int | None = None
    detect_size: int = 1280
    decode_size: int = 4096
    template_image: Path | None = None   # blank card crop used to register scans
    align: bool = True                   # resolve 180-degree flips and fine-align
    device: str = "cpu"
    rules_path: Path | None = None
    replace: bool = False

    def __post_init__(self) -> None:
        for name in ("weights", "template_path", "data_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("rules_path", "template_image"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Path(getattr(self, name)))
        if self.align and self.template_image is None:
            raise ValueError("alignment needs template_image (the blank card crop); "
                             "pass one or disable alignment")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.device != "cpu":
            raise ValueError("this pipeline runs on CPU; choose cpu")
        if self.canvas_size is not None and (
                not isinstance(self.canvas_size, int) or self.canvas_size < 2):
            raise ValueError("canvas_size must be an integer >= 2 or None")
        if not 32 <= self.detect_size <= 8192:
            raise ValueError("detect_size must be between 32 and 8192")
        if not 100 <= self.decode_size <= 8192 or self.decode_size < self.detect_size:
            raise ValueError("decode_size must be 100-8192 and at least detect_size")
