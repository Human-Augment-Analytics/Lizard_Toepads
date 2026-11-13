"""Configuration for the template builder.

Defaults resolve relative to the ``id-card-parsing/`` directory (the parent of
this package) so the app works when launched from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# id-card-parsing/  (parent of the template_builder package)
_ROOT = Path(__file__).resolve().parent.parent

PAD_MODES = ("solid",)


@dataclass(frozen=True)
class TemplateBuilderConfig:
    """Settings for normalizing a card and saving its field template.

    ``canvas_size`` is the square side length of the Normalized_Card (equal
    width and height). ``pad_mode``/``fill`` control how non-square rectified
    cards are padded. ``output_path`` is where the Template_Record JSON is
    written. ``preview_resolution`` is the display downscale for stage-1
    cropping; it does not affect rectification, which uses original pixels.
    """

    canvas_size: int = 512
    pad_mode: str = "solid"
    fill: tuple[int, int, int] = (255, 255, 255)
    output_path: Path = field(default_factory=lambda: _ROOT / "template" / "field_template.json")
    preview_resolution: int = 1536

    def __post_init__(self) -> None:
        if not isinstance(self.canvas_size, int) or self.canvas_size < 2:
            raise ValueError("canvas_size must be an integer >= 2")
        if self.pad_mode not in PAD_MODES:
            raise ValueError(f"pad_mode must be one of {PAD_MODES}")
        if (len(self.fill) != 3
                or any(not isinstance(channel, int) or not 0 <= channel <= 255
                       for channel in self.fill)):
            raise ValueError("fill must be three integers in 0..255")
        if not isinstance(self.preview_resolution, int) or self.preview_resolution <= 0:
            raise ValueError("preview_resolution must be a positive integer")
        # Normalize output_path to a Path without mutating a frozen dataclass field.
        object.__setattr__(self, "output_path", Path(self.output_path))
