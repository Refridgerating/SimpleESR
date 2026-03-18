"""Application entrypoint for the next-generation Qt GUI."""

from __future__ import annotations

import sys


def run(argv: list[str] | None = None) -> int:
    """Run the Qt GUI application."""

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        print(
            "PySide6 is required for gui2. Install with: "
            "python -m pip install -r requirements-gui2.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    try:
        import pyqtgraph  # noqa: F401
    except Exception as exc:
        print(
            "pyqtgraph is required for gui2. Install with: "
            "python -m pip install -r requirements-gui2.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    from .main_window import MainWindow

    app = QApplication(argv or sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

