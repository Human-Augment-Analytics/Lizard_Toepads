import threading
from types import SimpleNamespace

from annotator.__main__ import main


def test_keyboard_interrupt_uses_safe_close(monkeypatch):
    closed = threading.Event()
    calls = []
    def loop():
        raise KeyboardInterrupt
    def close():
        calls.append('close')
        closed.set()
    monkeypatch.setattr('annotator.__main__.App', lambda: SimpleNamespace(
        _closed=closed, mainloop=loop, _close=close))
    main()
    assert calls == ['close']
