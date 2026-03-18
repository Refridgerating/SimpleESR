"""View-model layer for the Qt GUI scaffold."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.fitting import (
    lorentzian_absorption_model,
    lorentzian_derivative_model,
)
from .types import AppState, TraceState


class ESRViewModel:
    """Mutable state manager independent from concrete Qt widgets."""

    def __init__(self) -> None:
        self.state = AppState()

    def add_trace(
        self,
        *,
        label: str,
        field: np.ndarray,
        intensity: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        trace = TraceState(
            label=label,
            field=np.asarray(field, dtype=float),
            intensity=np.asarray(intensity, dtype=float),
            metadata=dict(metadata or {}),
        )
        self.state.traces.append(trace)
        self.state.active_index = len(self.state.traces) - 1

    def active_trace(self) -> TraceState | None:
        if not self.state.traces:
            return None
        if not (0 <= self.state.active_index < len(self.state.traces)):
            self.state.active_index = 0
        return self.state.traces[self.state.active_index]

    def set_active(self, index: int) -> None:
        if 0 <= index < len(self.state.traces):
            self.state.active_index = index

    def set_visible(self, index: int, visible: bool) -> None:
        if 0 <= index < len(self.state.traces):
            self.state.traces[index].visible = bool(visible)

    def visible_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.state.traces) if bool(t.visible)]

    def trace_labels(self) -> list[str]:
        return [t.label for t in self.state.traces]

    def apply_pipeline_payload(self, index: int, payload: dict[str, Any]) -> None:
        if not (0 <= index < len(self.state.traces)):
            return
        trace = self.state.traces[index]

        peaks_raw = payload.get("peaks", [])
        trace.auto_peaks = []
        trace.absorption_peaks = []
        if isinstance(peaks_raw, list):
            if peaks_raw and isinstance(peaks_raw[0], (tuple, list)):
                for item in peaks_raw:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        try:
                            trace.auto_peaks.append((int(item[0]), int(item[1])))
                        except Exception:
                            continue
            else:
                for item in peaks_raw:
                    try:
                        trace.absorption_peaks.append(int(item))
                    except Exception:
                        continue

        trace.analysis_rows = []
        widths = payload.get("widths", [])
        if isinstance(widths, list):
            for item in widths:
                if not isinstance(item, dict):
                    continue
                try:
                    peak = int(item.get("peak", 0))
                    pos_idx = int(item.get("pos_idx", -1))
                    neg_idx = int(item.get("neg_idx", -1))
                    dhpp = float(item.get("peak_to_peak", 0.0))
                    fwhm = float(item.get("fwhm", 0.0))
                except Exception:
                    continue
                if not (0 <= pos_idx < len(trace.field) and 0 <= neg_idx < len(trace.field)):
                    continue
                pos_x = float(trace.field[pos_idx])
                neg_x = float(trace.field[neg_idx])
                pos_y = float(trace.intensity[pos_idx])
                neg_y = float(trace.intensity[neg_idx])
                trace.analysis_rows.append(
                    {
                        "analysis": "dH_pp",
                        "peak": peak,
                        "pos_x": pos_x,
                        "pos_y": pos_y,
                        "neg_x": neg_x,
                        "neg_y": neg_y,
                        "width": dhpp,
                    }
                )
                trace.analysis_rows.append(
                    {
                        "analysis": "FWHM",
                        "peak": peak,
                        "pos_x": pos_x,
                        "pos_y": pos_y,
                        "neg_x": neg_x,
                        "neg_y": neg_y,
                        "width": fwhm,
                    }
                )

        fit_rows = payload.get("fits", [])
        trace.fit_rows = fit_rows if isinstance(fit_rows, list) else []

    def fit_overlay(self, index: int) -> np.ndarray | None:
        """Return a summed fit overlay for the selected trace if available."""

        if not (0 <= index < len(self.state.traces)):
            return None
        trace = self.state.traces[index]
        if not trace.fit_rows:
            return None

        overlay = np.zeros_like(trace.field, dtype=float)
        used = False
        for row in trace.fit_rows:
            if not isinstance(row, dict):
                continue
            try:
                h_res = float(row.get("h_res", 0.0))
                delta = float(row.get("delta", 0.0))
                amp = float(row.get("A", 0.0))
                b_or_c = float(row.get("B", 0.0))
                kind = str(row.get("kind", "derivative")).lower()
            except Exception:
                continue
            if kind == "absorption":
                overlay += lorentzian_absorption_model(trace.field, h_res, delta, amp, b_or_c)
            else:
                overlay += lorentzian_derivative_model(trace.field, h_res, delta, amp, b_or_c)
            used = True
        return overlay if used else None


def normalize_analysis_label(label: str | None) -> str:
    """Normalize analysis labels for filtering."""

    if label is None:
        return ""
    return str(label).strip().lower()


def filter_analysis_rows(
    rows: list[dict[str, Any]],
    allowed_labels: set[str],
) -> list[dict[str, Any]]:
    """Return only analysis rows whose labels are enabled."""

    if not allowed_labels:
        return []
    normalized = {item.lower() for item in allowed_labels}
    return [
        row
        for row in rows
        if normalize_analysis_label(row.get("analysis")) in normalized
    ]
