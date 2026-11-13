"""Single-crop text recognition over the existing CPU PaddleOCR wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

# Matches card_ocr.prepare, which pads every field crop with a white border.
_BORDER = 8


class RecognizerUnavailable(RuntimeError):
    """Raised when the PaddleOCR recognizer cannot be created."""


class FieldRecognizer(Protocol):
    def recognize(self, crop: Image.Image) -> str: ...


class PaddleFieldRecognizer:
    """Recognize one field crop at a time with ``card_ocr.models.Paddle``.

    Pass ``adapter`` to inject any object with ``recognize_batch`` (tests do
    this); otherwise the real PaddleOCR recognizer is created lazily.
    """

    def __init__(self, *, adapter=None, cache: Path | None = None, model_id: str | None = None):
        if adapter is None:
            from card_ocr.models import DEFAULT_CACHE, create_model
            try:
                adapter = create_model("paddle", cache=cache or DEFAULT_CACHE,
                                       device="cpu", model_id=model_id)
            except ImportError as exc:
                raise RecognizerUnavailable(
                    f"PaddleOCR is not installed in this environment ({exc}); install "
                    "id-card-parsing/requirements.txt into the environment you run from") from exc
        self._adapter = adapter

    @property
    def model_id(self) -> str | None:
        return getattr(self._adapter, "model_id", None)

    def recognize(self, crop: Image.Image) -> str:
        padded = ImageOps.expand(crop.convert("RGB"), border=_BORDER, fill="white")
        results = self._adapter.recognize_batch([padded])
        if len(results) != 1:
            raise ValueError("Recognizer must return one prediction per crop")
        return results[0].text
