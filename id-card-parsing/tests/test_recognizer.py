import pytest
from PIL import Image

from card_ocr.models import Prediction
from card_pipeline.recognizer import PaddleFieldRecognizer


class _Adapter:
    model_id = "fake-model"

    def __init__(self, texts):
        self.texts = list(texts)
        self.seen = []

    def recognize_batch(self, images):
        self.seen.extend(images)
        return [Prediction(self.texts.pop(0)) for _ in images]


def test_recognize_returns_adapter_text():
    recognizer = PaddleFieldRecognizer(adapter=_Adapter(["12.5"]))
    assert recognizer.recognize(Image.new("RGB", (40, 20), "white")) == "12.5"


def test_crop_gets_white_border_like_card_ocr_prepare():
    adapter = _Adapter(["x"])
    PaddleFieldRecognizer(adapter=adapter).recognize(Image.new("RGB", (40, 20), "black"))
    padded = adapter.seen[0]
    assert padded.size == (40 + 16, 20 + 16)
    assert padded.getpixel((0, 0)) == (255, 255, 255)


def test_model_id_comes_from_adapter():
    assert PaddleFieldRecognizer(adapter=_Adapter([])).model_id == "fake-model"


def test_missing_paddle_raises_clear_error(monkeypatch):
    import card_ocr.models as models
    from card_pipeline.recognizer import RecognizerUnavailable

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'paddleocr'")

    monkeypatch.setattr(models, "create_model", missing)
    with pytest.raises(RecognizerUnavailable, match="requirements.txt"):
        PaddleFieldRecognizer()


def test_wrong_prediction_count_raises():
    class Bad:
        def recognize_batch(self, images):
            return []

    with pytest.raises(ValueError):
        PaddleFieldRecognizer(adapter=Bad()).recognize(Image.new("RGB", (4, 4)))
