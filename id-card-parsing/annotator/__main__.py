"""Entry point: ``python -m annotator`` from the ``id-card-parsing/`` directory."""

from __future__ import annotations

from .ui.app import App


def main() -> None:
    app = App()
    while not app._closed.is_set():
        try:
            app.mainloop()
            break
        except KeyboardInterrupt:
            app._close()


if __name__ == "__main__":
    main()
