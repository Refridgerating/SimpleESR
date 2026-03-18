"""High-performance plotting widget based on pyqtgraph."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class LivePlotWidget(QWidget):
    """Wrapper around a pyqtgraph plot with lab-friendly defaults."""

    _RAW_COLORS = [
        "#53b9ff",
        "#ffd166",
        "#7ae582",
        "#ff7f7f",
        "#cdb4ff",
        "#90e0ef",
        "#f4a261",
        "#a3b18a",
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._theme = "dark"
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setBackground("#121417")
        self._plot.setLabel("bottom", "Magnetic Field", units="mT")
        self._plot.setLabel("left", "Signal", units="a.u.")
        self._legend = self._plot.addLegend(offset=(8, 8))

        self._raw_curves: dict[int, pg.PlotDataItem] = {}
        self._fit_curves: dict[int, pg.PlotDataItem] = {}
        self._crit_pos = pg.ScatterPlotItem(
            pen=pg.mkPen(color="#ff6b6b"),
            brush=pg.mkBrush("#ff6b6b"),
            size=9,
            symbol="o",
        )
        self._crit_neg = pg.ScatterPlotItem(
            pen=pg.mkPen(color="#4ecdc4"),
            brush=pg.mkBrush("#4ecdc4"),
            size=9,
            symbol="t",
        )
        self._plot.addItem(self._crit_pos)
        self._plot.addItem(self._crit_neg)
        self._crit_pos.hide()
        self._crit_neg.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    def _pen_for(self, index: int, *, active: bool) -> pg.mkPen:
        color = self._RAW_COLORS[index % len(self._RAW_COLORS)]
        width = 2.3 if active else 1.3
        return pg.mkPen(color=color, width=width)

    def _fit_pen_for(self, index: int) -> pg.mkPen:
        color = self._RAW_COLORS[index % len(self._RAW_COLORS)]
        return pg.mkPen(color=color, width=1.2, style=pg.QtCore.Qt.DashLine)

    def set_theme(self, theme: str) -> None:
        """Apply simple plot styling for the selected theme."""

        self._theme = "light" if theme == "light" else "dark"
        if self._theme == "light":
            self._plot.setBackground("#f6f8fb")
            axis_pen = pg.mkPen(color="#22313f")
        else:
            self._plot.setBackground("#121417")
            axis_pen = pg.mkPen(color="#d8dee9")
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)

    def set_traces(
        self,
        traces: list[dict[str, Any]],
        *,
        active_index: int,
        fit_overlays: dict[int, Any] | None = None,
    ) -> None:
        """Render many traces and optional fit overlays."""

        fit_overlays = fit_overlays or {}
        live_ids: set[int] = set()
        for entry in traces:
            try:
                idx = int(entry["index"])
            except Exception:
                continue
            live_ids.add(idx)

            field = entry.get("field")
            intensity = entry.get("intensity")
            label = str(entry.get("label", f"Trace {idx + 1}"))
            visible = bool(entry.get("visible", True))
            if field is None or intensity is None:
                continue

            raw_curve = self._raw_curves.get(idx)
            if raw_curve is None:
                raw_curve = self._plot.plot(name=label)
                self._raw_curves[idx] = raw_curve
            raw_curve.setData(field, intensity)
            raw_curve.setDownsampling(auto=True, method="peak")
            raw_curve.setClipToView(True)
            raw_curve.setPen(self._pen_for(idx, active=(idx == active_index)))
            raw_curve.setVisible(visible)

            overlay = fit_overlays.get(idx)
            fit_curve = self._fit_curves.get(idx)
            if overlay is not None and visible:
                if fit_curve is None:
                    fit_curve = self._plot.plot(name=f"{label} fit")
                    self._fit_curves[idx] = fit_curve
                fit_curve.setData(field, overlay)
                fit_curve.setDownsampling(auto=True, method="peak")
                fit_curve.setClipToView(True)
                fit_curve.setPen(self._fit_pen_for(idx))
                fit_curve.setVisible(True)
            elif fit_curve is not None:
                fit_curve.setVisible(False)

        stale_raw = [idx for idx in self._raw_curves.keys() if idx not in live_ids]
        for idx in stale_raw:
            curve = self._raw_curves.pop(idx, None)
            if curve is not None:
                self._plot.removeItem(curve)
        stale_fit = [idx for idx in self._fit_curves.keys() if idx not in live_ids]
        for idx in stale_fit:
            curve = self._fit_curves.pop(idx, None)
            if curve is not None:
                self._plot.removeItem(curve)

    def set_critical_points(
        self,
        positives: list[tuple[float, float]] | None,
        negatives: list[tuple[float, float]] | None,
    ) -> None:
        """Display user-selected peak markers."""

        self._set_scatter(self._crit_pos, positives)
        self._set_scatter(self._crit_neg, negatives)

    def _set_scatter(
        self,
        item: pg.ScatterPlotItem,
        points: list[tuple[float, float]] | None,
    ) -> None:
        if points:
            xs, ys = zip(*points)
            item.setData(xs, ys)
            item.setVisible(True)
        else:
            item.clear()
            item.setVisible(False)

    def clear(self) -> None:
        for curve in self._raw_curves.values():
            self._plot.removeItem(curve)
        for curve in self._fit_curves.values():
            self._plot.removeItem(curve)
        self._raw_curves.clear()
        self._fit_curves.clear()
        self._crit_pos.clear()
        self._crit_neg.clear()
        self._crit_pos.hide()
        self._crit_neg.hide()
