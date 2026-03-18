"""Next-generation Qt GUI scaffold for SimpleESR."""

from __future__ import annotations

from typing import Sequence

__all__ = ["run"]


def run(argv: Sequence[str] | None = None) -> int:
    """Proxy entry point that avoids importing ``app`` at module import time."""

    from .app import run as _run

    return _run(list(argv) if argv is not None else None)
