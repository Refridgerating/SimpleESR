"""Typed state containers for the Qt GUI layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TraceState:
    """In-memory state for one plotted trace."""

    label: str
    field: np.ndarray
    intensity: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    analysis_rows: list[dict[str, Any]] = field(default_factory=list)
    fit_rows: list[dict[str, Any]] = field(default_factory=list)
    auto_peaks: list[tuple[int, int]] = field(default_factory=list)
    absorption_peaks: list[int] = field(default_factory=list)


@dataclass
class AppState:
    """Top-level GUI state."""

    traces: list[TraceState] = field(default_factory=list)
    active_index: int = 0

    @property
    def has_data(self) -> bool:
        return bool(self.traces)


@dataclass
class AnalysisOptions:
    """User-configurable knobs for the analysis wizard."""

    expected_peaks: int = 4
    peak_method: str = "auto"
    show_dhpp: bool = True
    show_fwhm: bool = True

    def sanitized_expected(self) -> int:
        """Return a valid even count suitable for the peak finder."""

        value = max(2, int(self.expected_peaks))
        if value % 2 != 0:
            value += 1
        return value

    def allowed_labels(self) -> set[str]:
        """Return normalized analysis labels that remain visible."""

        allowed: set[str] = set()
        if self.show_dhpp:
            allowed.add("dh_pp")
        if self.show_fwhm:
            allowed.add("fwhm")
        return {label.lower() for label in allowed}
