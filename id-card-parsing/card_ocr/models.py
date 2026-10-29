"""Recognize aligned handwriting crops with PaddleOCR on CPU."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_ID = 'en_PP-OCRv5_mobile_rec'
DEFAULT_CACHE = Path(sys.prefix) / '.cache' / 'card-ocr'


@dataclass(frozen=True)
class Prediction:
    """Raw text returned by the recognizer."""
    text: str


class Paddle:
    """Run the English PP-OCRv5 recognizer on known field crops using CPU."""

    def __init__(self, cache: Path, *, model_id=None, device='cpu'):
        """Load a CPU recognizer with checkpoint files stored in the supplied cache."""
        if device != 'cpu':
            raise ValueError('PaddleOCR in this workflow uses CPU; choose cpu')
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['PADDLE_PDX_CACHE_HOME'] = str(cache / 'paddlex')
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
        from paddleocr import TextRecognition
        self.model_id = model_id or MODEL_ID
        self.model = TextRecognition(model_name=self.model_id, device='cpu')

    def recognize_batch(self, images: list[Image.Image]) -> list[Prediction]:
        """Convert RGB crops to Paddle's BGR input and return unchanged text."""
        inputs = [np.asarray(image.convert('RGB'))[:, :, ::-1].copy() for image in images]
        results = self.model.predict(input=inputs, batch_size=1)
        return [Prediction(str(result['rec_text'])) for result in results]


def create_model(name='paddle', *, cache=DEFAULT_CACHE, device='cpu', model_id=None):
    """Create the Paddle recognizer."""
    if name != 'paddle':
        raise ValueError('Only PaddleOCR is supported; choose paddle')
    return Paddle(cache, model_id=model_id, device=device)
