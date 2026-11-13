"""Entry point: ``python -m template_builder`` from the ``id-card-parsing/`` dir."""

from __future__ import annotations

from .backend import TemplateBuilderBackend
from .config import TemplateBuilderConfig
from .ui.app import App


def main() -> None:
    backend = TemplateBuilderBackend(TemplateBuilderConfig())
    App(backend).mainloop()


if __name__ == "__main__":
    main()
