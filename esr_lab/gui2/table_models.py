"""Qt table models for analysis and fit result views."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from .formulas import (
    ANALYSIS_HEADER_TOOLTIPS,
    FIT_HEADER_TOOLTIPS,
    GROUP_HEADER_TOOLTIPS,
    analysis_cell_tooltip,
    fit_cell_tooltip,
)


class DictRowsTableModel(QAbstractTableModel):
    """Simple reusable table model backed by list[dict]."""

    def __init__(
        self,
        columns: list[tuple[str, str]],
        parent=None,
        *,
        header_tooltips: dict[str, str] | None = None,
        cell_tooltip_builder: Callable[[dict[str, Any], str], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._columns = columns
        self._rows: list[dict[str, Any]] = []
        self._header_tooltips = dict(header_tooltips or {})
        self._cell_tooltip_builder = cell_tooltip_builder

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        key, _header = self._columns[index.column()]
        if role == Qt.ToolTipRole:
            if callable(self._cell_tooltip_builder):
                tip = self._cell_tooltip_builder(row, key)
                if tip:
                    return tip
            return self._header_tooltips.get(key)
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        val = row.get(key, "")
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ) -> Any:  # noqa: N802
        if orientation == Qt.Horizontal and 0 <= section < len(self._columns):
            key, header = self._columns[section]
            if role == Qt.DisplayRole:
                return header
            if role == Qt.ToolTipRole:
                return self._header_tooltips.get(key)
            return None
        if role == Qt.DisplayRole:
            return str(section + 1)
        return None


def analysis_table_model(parent=None) -> DictRowsTableModel:
    return DictRowsTableModel(
        [
            ("analysis", "Analysis"),
            ("peak", "Peak"),
            ("pos_x", "Pos X"),
            ("neg_x", "Neg X"),
            ("width", "Value"),
        ],
        parent=parent,
        header_tooltips=ANALYSIS_HEADER_TOOLTIPS,
        cell_tooltip_builder=analysis_cell_tooltip,
    )


def fit_table_model(parent=None) -> DictRowsTableModel:
    return DictRowsTableModel(
        [
            ("peak", "Peak"),
            ("kind", "Kind"),
            ("h_res", "H_res"),
            ("delta", "Delta"),
            ("A", "A"),
            ("B", "B/C"),
            ("g", "g"),
            ("g_err", "g err"),
            ("g_err_pct", "g err %"),
            ("area", "Area"),
            ("area_err", "Area err"),
            ("area_err_pct", "Area err %"),
            ("chi2", "chi2"),
            ("stderr", "stderr"),
        ],
        parent=parent,
        header_tooltips=FIT_HEADER_TOOLTIPS,
        cell_tooltip_builder=fit_cell_tooltip,
    )


def group_table_model(parent=None) -> DictRowsTableModel:
    return DictRowsTableModel(
        [
            ("group", "Group"),
            ("peak", "Peak"),
            ("kind", "Kind"),
            ("n_total", "n total"),
            ("n_used", "n used"),
            ("h_res_mean", "H_res mean"),
            ("h_res_std", "H_res std"),
            ("h_res_sem", "H_res sem"),
            ("h_res_wmean", "H_res wmean"),
            ("h_res_wsem", "H_res wsem"),
            ("h_res_err_total", "H_res err"),
            ("delta_mean", "Delta mean"),
            ("delta_std", "Delta std"),
            ("g_mean", "g mean"),
            ("g_std", "g std"),
            ("area_mean", "Area mean"),
            ("area_std", "Area std"),
            ("chi2_mean", "chi2 mean"),
            ("included", "Included"),
            ("rejected", "Rejected"),
        ],
        parent=parent,
        header_tooltips=GROUP_HEADER_TOOLTIPS,
    )
