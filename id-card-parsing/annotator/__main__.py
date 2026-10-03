"""Entry point: ``python -m annotator`` from the ``id-card-parsing/`` directory."""

from __future__ import annotations

from .backend import AnnotatorBackend
from .config import AnnotatorConfig
from .ui.app import App


def main() -> None:
    config = AnnotatorConfig()
    backend = AnnotatorBackend(config)
    App(backend).mainloop()


if __name__ == "__main__":
    main()
