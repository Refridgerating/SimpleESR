"""Main window scaffold for the next-generation ESR GUI."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QThread, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..spectrum import ESRSpectrum
from .analysis_wizard import AnalysisWizard
from .formulas import CONTROL_TOOLTIPS
from .plot_widget import LivePlotWidget
from .table_models import analysis_table_model, fit_table_model, group_table_model
from .theme import normalize_theme, stylesheet_for
from .types import AnalysisOptions
from .viewmodels import ESRViewModel, filter_analysis_rows
from .workers import FileLoadWorker, PipelineWorker
from ..services import summarize_replicate_fits


class MainWindow(QMainWindow):
    """Qt scaffold focused on fast plotting + async analysis."""

    ORG_NAME = "SimpleESR"
    APP_NAME = "SimpleESRNext"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SimpleESR Next")
        self.resize(1500, 900)

        self._settings = QSettings(self.ORG_NAME, self.APP_NAME)
        self._theme = normalize_theme(str(self._settings.value("ui/theme", "dark")))
        self._last_dir = str(self._settings.value("paths/last_dir", ""))

        self._vm = ESRViewModel()
        self._busy = False
        self._analysis_options = self._load_analysis_options()

        self._pipeline_thread: QThread | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._loader_thread: QThread | None = None
        self._loader_worker: FileLoadWorker | None = None
        self._pipeline_errors: list[str] = []
        self._load_errors: list[str] = []
        self._updating_trace_list = False
        self._group_rows: list[dict[str, Any]] = []

        self._build_ui()
        self._apply_theme(self._theme)
        self._restore_window_settings()
        self._update_controls()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        controls = QWidget(root)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.open_btn = QPushButton("Open CSV")
        self.analyze_btn = QPushButton("Analyze Active")
        self.batch_btn = QPushButton("Analyze All")
        self.export_btn = QPushButton("Export CSV")
        self.theme_btn = QPushButton("Theme")
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()

        controls_layout.addWidget(self.open_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.batch_btn)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addWidget(self.theme_btn)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.progress)

        self.main_splitter = QSplitter(root)
        self.main_splitter.setChildrenCollapsible(False)

        trace_panel = QWidget(self.main_splitter)
        trace_layout = QVBoxLayout(trace_panel)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_layout.setSpacing(6)
        trace_layout.addWidget(QLabel("Traces"))
        self.trace_list = QListWidget(trace_panel)
        self.trace_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        trace_layout.addWidget(self.trace_list, stretch=1)
        trace_btns = QWidget(trace_panel)
        trace_btns_layout = QHBoxLayout(trace_btns)
        trace_btns_layout.setContentsMargins(0, 0, 0, 0)
        trace_btns_layout.setSpacing(6)
        self.show_selected_btn = QPushButton("Show Selected", trace_btns)
        self.hide_selected_btn = QPushButton("Hide Selected", trace_btns)
        trace_btns_layout.addWidget(self.show_selected_btn)
        trace_btns_layout.addWidget(self.hide_selected_btn)
        trace_layout.addWidget(trace_btns)

        self.plot_widget = LivePlotWidget(self.main_splitter)

        panel = QWidget(self.main_splitter)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)
        panel_layout.addWidget(QLabel("Analysis"))
        self.analysis_model = analysis_table_model(panel)
        self.analysis_table = QTableView(panel)
        self.analysis_table.setModel(self.analysis_model)
        panel_layout.addWidget(self.analysis_table, stretch=2)
        self.analysis_filter_widget = QWidget(panel)
        analysis_filter_layout = QHBoxLayout(self.analysis_filter_widget)
        analysis_filter_layout.setContentsMargins(0, 0, 0, 0)
        analysis_filter_layout.setSpacing(6)
        analysis_filter_layout.addWidget(QLabel("Critical points:"))
        self.analysis_dhpp_toggle = QCheckBox("Delta H_pp", self.analysis_filter_widget)
        self.analysis_dhpp_toggle.setChecked(self._analysis_options.show_dhpp)
        self.analysis_dhpp_toggle.toggled.connect(self._on_analysis_filter_changed)
        analysis_filter_layout.addWidget(self.analysis_dhpp_toggle)
        self.analysis_fwhm_toggle = QCheckBox("FWHM", self.analysis_filter_widget)
        self.analysis_fwhm_toggle.setChecked(self._analysis_options.show_fwhm)
        self.analysis_fwhm_toggle.toggled.connect(self._on_analysis_filter_changed)
        analysis_filter_layout.addWidget(self.analysis_fwhm_toggle)
        analysis_filter_layout.addStretch(1)
        panel_layout.addWidget(self.analysis_filter_widget)
        panel_layout.addWidget(QLabel("Fits"))
        self.fit_model = fit_table_model(panel)
        self.fit_table = QTableView(panel)
        self.fit_table.setModel(self.fit_model)
        panel_layout.addWidget(self.fit_table, stretch=2)
        panel_layout.addWidget(QLabel("Grouped Fits (R# triplicates)"))
        self.group_model = group_table_model(panel)
        self.group_table = QTableView(panel)
        self.group_table.setModel(self.group_model)
        panel_layout.addWidget(self.group_table, stretch=2)

        self.main_splitter.setSizes([260, 900, 520])
        root_layout.addWidget(controls)
        root_layout.addWidget(self.main_splitter, stretch=1)

        self.open_btn.clicked.connect(self._open_files)
        self.analyze_btn.clicked.connect(self._analyze_active)
        self.batch_btn.clicked.connect(self._analyze_all)
        self.export_btn.clicked.connect(self._export_results)
        self.theme_btn.clicked.connect(self._toggle_theme)
        self.trace_list.currentRowChanged.connect(self._on_trace_selected)
        self.trace_list.itemChanged.connect(self._on_trace_item_changed)
        self.show_selected_btn.clicked.connect(lambda: self._set_selected_visibility(True))
        self.hide_selected_btn.clicked.connect(lambda: self._set_selected_visibility(False))
        self._apply_tooltips()

    def _apply_tooltips(self) -> None:
        self.open_btn.setToolTip(CONTROL_TOOLTIPS["open_btn"])
        self.analyze_btn.setToolTip(CONTROL_TOOLTIPS["analyze_btn"])
        self.batch_btn.setToolTip(CONTROL_TOOLTIPS["batch_btn"])
        self.export_btn.setToolTip(CONTROL_TOOLTIPS["export_btn"])
        self.theme_btn.setToolTip(CONTROL_TOOLTIPS["theme_btn"])
        self.trace_list.setToolTip(CONTROL_TOOLTIPS["trace_list"])
        self.show_selected_btn.setToolTip(CONTROL_TOOLTIPS["show_selected_btn"])
        self.hide_selected_btn.setToolTip(CONTROL_TOOLTIPS["hide_selected_btn"])
        self.progress.setToolTip(CONTROL_TOOLTIPS["progress"])
        self.analysis_table.setToolTip(CONTROL_TOOLTIPS["analysis_table"])
        self.analysis_filter_widget.setToolTip(CONTROL_TOOLTIPS["analysis_filters"])
        self.analysis_dhpp_toggle.setToolTip(CONTROL_TOOLTIPS["analysis_filter_dhpp"])
        self.analysis_fwhm_toggle.setToolTip(CONTROL_TOOLTIPS["analysis_filter_fwhm"])
        self.fit_table.setToolTip(CONTROL_TOOLTIPS["fit_table"])
        self.group_table.setToolTip(CONTROL_TOOLTIPS["group_table"])
        self.plot_widget.setToolTip(CONTROL_TOOLTIPS["plot_widget"])

    def _on_analysis_filter_changed(self) -> None:
        self._analysis_options.show_dhpp = self.analysis_dhpp_toggle.isChecked()
        self._analysis_options.show_fwhm = self.analysis_fwhm_toggle.isChecked()
        self._save_analysis_options()
        self._refresh_active_views()

    def _sync_filter_toggles_from_options(self) -> None:
        toggles = [
            (self.analysis_dhpp_toggle, self._analysis_options.show_dhpp),
            (self.analysis_fwhm_toggle, self._analysis_options.show_fwhm),
        ]
        for widget, value in toggles:
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        if busy:
            self.progress.show()
            self.progress.setValue(0)
        else:
            self.progress.hide()
        if message:
            self.status_label.setText(message)
        self._update_controls()

    def _update_controls(self) -> None:
        has_data = self._vm.state.has_data
        can_action = has_data and not self._busy
        self.analyze_btn.setEnabled(can_action)
        self.batch_btn.setEnabled(can_action)
        self.export_btn.setEnabled(can_action)
        self.show_selected_btn.setEnabled(can_action)
        self.hide_selected_btn.setEnabled(can_action)
        self.open_btn.setEnabled(not self._busy)
        self.trace_list.setEnabled(not self._busy)

    def _refresh_trace_list(self, selected_index: int | None = None) -> None:
        selected_rows = {idx.row() for idx in self.trace_list.selectedIndexes()}
        labels = self._vm.trace_labels()
        self.trace_list.blockSignals(True)
        self._updating_trace_list = True
        self.trace_list.clear()
        for idx, label in enumerate(labels):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            visible = self._vm.state.traces[idx].visible
            item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
            self.trace_list.addItem(item)

        if not labels:
            self._updating_trace_list = False
            self.trace_list.blockSignals(False)
            return
        target = self._vm.state.active_index if selected_index is None else selected_index
        if not (0 <= target < len(labels)):
            target = len(labels) - 1
        self.trace_list.setCurrentRow(target)
        for row in sorted(selected_rows):
            if 0 <= row < len(labels):
                item = self.trace_list.item(row)
                if item is not None:
                    item.setSelected(True)
        self._updating_trace_list = False
        self.trace_list.blockSignals(False)

    def _refresh_active_views(self) -> None:
        trace = self._vm.active_trace()
        if trace is None:
            self.analysis_model.set_rows([])
            self.fit_model.set_rows([])
            self.group_model.set_rows([])
            self.plot_widget.clear()
            self.status_label.setText("Ready")
            return

        fit_overlays: dict[int, Any] = {}
        traces_payload: list[dict[str, Any]] = []
        for idx, row in enumerate(self._vm.state.traces):
            traces_payload.append(
                {
                    "index": idx,
                    "label": row.label,
                    "field": row.field,
                    "intensity": row.intensity,
                    "visible": row.visible,
                }
            )
            overlay = self._vm.fit_overlay(idx)
            if overlay is not None:
                fit_overlays[idx] = overlay
        self.plot_widget.set_traces(
            traces_payload,
            active_index=self._vm.state.active_index,
            fit_overlays=fit_overlays,
        )
        allowed = self._analysis_options.allowed_labels()
        filtered_rows = filter_analysis_rows(trace.analysis_rows, allowed)
        self.analysis_model.set_rows(filtered_rows)
        positives, negatives = self._collect_critical_points(filtered_rows)
        self.plot_widget.set_critical_points(positives, negatives)
        self.fit_model.set_rows(trace.fit_rows)
        self.group_model.set_rows(self._group_rows)
        visible_count = len(self._vm.visible_indices())
        self.status_label.setText(f"Active: {trace.label} ({visible_count} visible)")

    def _open_files(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Open", "Wait for the current task to finish.")
            return
        start_dir = self._last_dir if self._last_dir else ""
        paths, _filter = QFileDialog.getOpenFileNames(
            self,
            "Open ESR CSV files",
            start_dir,
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not paths:
            return
        first = Path(paths[0]).resolve()
        self._last_dir = str(first.parent)
        self._start_file_load(paths)

    def _start_file_load(self, paths: list[str]) -> None:
        if self._loader_thread is not None and self._loader_thread.isRunning():
            QMessageBox.information(self, "Open", "File loading is already running.")
            return
        self._load_errors = []
        self._set_busy(True, "Loading files...")

        self._loader_thread = QThread(self)
        self._loader_worker = FileLoadWorker(paths=paths)
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.progress.connect(self._on_loader_progress)
        self._loader_worker.finished.connect(self._on_loader_finished)
        self._loader_worker.failed.connect(self._on_loader_failed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.failed.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._cleanup_loader_worker)
        self._loader_thread.start()

    def _on_loader_progress(self, current: int, total: int, label: str) -> None:
        pct = int((current / max(total, 1)) * 100.0)
        self.progress.setValue(pct)
        self.status_label.setText(f"Loading {label} ({current}/{total})")

    def _on_loader_failed(self, message: str) -> None:
        self._set_busy(False, "Load failed")
        QMessageBox.warning(self, "Open", message)

    def _on_loader_finished(self, loaded: object, errors: object) -> None:
        loaded_rows = loaded if isinstance(loaded, list) else []
        error_rows = errors if isinstance(errors, list) else []

        loaded_count = 0
        for item in loaded_rows:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            label, spectrum = item
            if not isinstance(label, str) or not isinstance(spectrum, ESRSpectrum):
                continue
            self._vm.add_trace(
                label=label,
                field=spectrum.field,
                intensity=spectrum.intensity,
                metadata=spectrum.metadata or {},
            )
            loaded_count += 1

        self._set_busy(False, f"Loaded {loaded_count} trace(s)")
        self._group_rows = []
        self.group_model.set_rows([])
        self._refresh_trace_list()
        self._refresh_active_views()
        self._update_controls()

        if error_rows:
            msg = "\n".join(str(e) for e in error_rows)
            QMessageBox.warning(self, "Open", msg)

    def _cleanup_loader_worker(self) -> None:
        if self._loader_worker is not None:
            self._loader_worker.deleteLater()
            self._loader_worker = None
        if self._loader_thread is not None:
            self._loader_thread.deleteLater()
            self._loader_thread = None

    def _start_pipeline(self, indices: list[int], *, options: AnalysisOptions | None = None) -> None:
        if self._busy:
            QMessageBox.information(self, "Analyze", "Wait for the current task to finish.")
            return
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            QMessageBox.information(self, "Analyze", "Analysis is already running.")
            return

        opts = options or self._analysis_options
        expected = opts.sanitized_expected()
        method = self._normalized_peak_method(opts.peak_method)

        spectra = [
            ESRSpectrum(field=t.field, intensity=t.intensity, metadata=t.metadata)
            for t in self._vm.state.traces
        ]

        self._pipeline_errors = []
        self._set_busy(True, "Analyzing...")

        self._pipeline_thread = QThread(self)
        self._pipeline_worker = PipelineWorker(
            spectra=spectra,
            indices=indices,
            expected_peaks=expected,
            peak_method=method,
        )
        self._pipeline_worker.moveToThread(self._pipeline_thread)
        self._pipeline_thread.started.connect(self._pipeline_worker.run)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.failed.connect(self._on_pipeline_failed)
        self._pipeline_worker.finished.connect(self._pipeline_thread.quit)
        self._pipeline_worker.failed.connect(self._pipeline_thread.quit)
        self._pipeline_thread.finished.connect(self._cleanup_pipeline_worker)
        self._pipeline_thread.start()

    def _analyze_active(self) -> None:
        if not self._vm.state.has_data:
            return
        self._start_pipeline([self._vm.state.active_index], options=self._analysis_options)

    def _analyze_all(self) -> None:
        if not self._vm.state.has_data:
            return
        wizard = AnalysisWizard(parent=self, initial=self._analysis_options)
        if wizard.exec() != QDialog.Accepted:
            return
        self._analysis_options = wizard.selected_options()
        self._sync_filter_toggles_from_options()
        self._on_analysis_filter_changed()
        self._start_pipeline(list(range(len(self._vm.state.traces))), options=self._analysis_options)

    def _on_pipeline_progress(self, current: int, total: int, label: str) -> None:
        pct = int((current / max(total, 1)) * 100.0)
        self.progress.setValue(pct)
        self.status_label.setText(f"Analyzing {label} ({current}/{total})")

    def _on_pipeline_failed(self, message: str) -> None:
        self._pipeline_errors.append(message)
        self._set_busy(False, "Analysis failed")
        QMessageBox.warning(self, "Analyze", message)

    def _on_pipeline_finished(self, results: object) -> None:
        rows = results if isinstance(results, list) else []
        for item in rows:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            idx, payload = item
            try:
                self._vm.apply_pipeline_payload(int(idx), payload)
            except Exception:
                continue

        self.progress.setValue(100)
        self._rebuild_group_rows()
        self._set_busy(False, "Analysis complete")
        self._refresh_active_views()
        if self._pipeline_errors:
            QMessageBox.warning(self, "Analyze", "\n".join(self._pipeline_errors))

    def _rebuild_group_rows(self) -> None:
        labels = self._vm.trace_labels()
        payloads = [{"fits": trace.fit_rows} for trace in self._vm.state.traces]
        self._group_rows = summarize_replicate_fits(
            labels,
            payloads,
            min_replicates=2,
            max_chi2=25.0,
            outlier_z=3.5,
        )
        if self._vm.active_trace() is not None:
            self.group_model.set_rows(self._group_rows)

    def _cleanup_pipeline_worker(self) -> None:
        if self._pipeline_worker is not None:
            self._pipeline_worker.deleteLater()
            self._pipeline_worker = None
        if self._pipeline_thread is not None:
            self._pipeline_thread.deleteLater()
            self._pipeline_thread = None

    def _on_trace_selected(self, index: int) -> None:
        if self._busy:
            return
        self._vm.set_active(index)
        self._refresh_active_views()

    def _on_trace_item_changed(self, item: QListWidgetItem) -> None:
        if self._busy or self._updating_trace_list:
            return
        index = self.trace_list.row(item)
        if index < 0:
            return
        self._vm.set_visible(index, item.checkState() == Qt.Checked)
        self._refresh_active_views()

    def _set_selected_visibility(self, visible: bool) -> None:
        if self._busy or not self._vm.state.has_data:
            return
        rows = sorted({idx.row() for idx in self.trace_list.selectedIndexes()})
        if not rows:
            current = self.trace_list.currentRow()
            if current >= 0:
                rows = [current]
        if not rows:
            return
        for row in rows:
            self._vm.set_visible(row, visible)
        self._refresh_trace_list(selected_index=self._vm.state.active_index)
        self._refresh_active_views()

    def _export_results(self) -> None:
        if not self._vm.state.has_data or self._busy:
            return

        default_name = "simpleesr_export.csv"
        start_path = str(Path(self._last_dir) / default_name) if self._last_dir else default_name
        out_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Analysis",
            start_path,
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not out_path:
            return
        self._last_dir = str(Path(out_path).resolve().parent)

        metadata_keys: set[str] = set()
        for trace in self._vm.state.traces:
            metadata_keys.update(str(k) for k in trace.metadata.keys())
        meta_fields = [f"meta_{k}" for k in sorted(metadata_keys)]

        fieldnames = [
            "trace",
            "category",
            "group",
            "analysis",
            "peak",
            "pos_x",
            "pos_y",
            "neg_x",
            "neg_y",
            "width",
            "h_res",
            "delta",
            "A",
            "B",
            "g",
            "g_err",
            "g_err_pct",
            "area",
            "area_err",
            "area_err_pct",
            "chi2",
            "stderr",
            "kind",
            "n_total",
            "n_used",
            "h_res_mean",
            "h_res_std",
            "h_res_sem",
            "h_res_wmean",
            "h_res_wsem",
            "h_res_err_total",
            "delta_mean",
            "delta_std",
            "g_mean",
            "g_std",
            "area_mean",
            "area_std",
            "chi2_mean",
            "included",
            "rejected",
        ] + meta_fields

        rows: list[dict[str, Any]] = []
        for trace in self._vm.state.traces:
            trace_meta = {f"meta_{k}": trace.metadata.get(k, "") for k in metadata_keys}

            for item in trace.analysis_rows:
                row = {key: "" for key in fieldnames}
                row["trace"] = trace.label
                row["category"] = "analysis"
                row["group"] = ""
                row["analysis"] = item.get("analysis", "")
                row["peak"] = item.get("peak", "")
                row["pos_x"] = item.get("pos_x", "")
                row["pos_y"] = item.get("pos_y", "")
                row["neg_x"] = item.get("neg_x", "")
                row["neg_y"] = item.get("neg_y", "")
                row["width"] = item.get("width", "")
                row.update(trace_meta)
                rows.append(row)

            for item in trace.fit_rows:
                row = {key: "" for key in fieldnames}
                row["trace"] = trace.label
                row["category"] = "fit"
                row["group"] = ""
                row["analysis"] = item.get("analysis", "Lorentzian")
                row["peak"] = item.get("peak", "")
                row["h_res"] = item.get("h_res", "")
                row["delta"] = item.get("delta", "")
                row["A"] = item.get("A", "")
                row["B"] = item.get("B", "")
                row["g"] = item.get("g", "")
                row["g_err"] = item.get("g_err", "")
                row["g_err_pct"] = item.get("g_err_pct", "")
                row["area"] = item.get("area", "")
                row["area_err"] = item.get("area_err", "")
                row["area_err_pct"] = item.get("area_err_pct", "")
                row["chi2"] = item.get("chi2", "")
                row["stderr"] = item.get("stderr", "")
                row["kind"] = item.get("kind", "")
                row.update(trace_meta)
                rows.append(row)

        for item in self._group_rows:
            row = {key: "" for key in fieldnames}
            row["trace"] = ""
            row["category"] = "group_fit"
            row["group"] = item.get("group", "")
            row["analysis"] = "replicate_summary"
            row["peak"] = item.get("peak", "")
            row["kind"] = item.get("kind", "")
            row["n_total"] = item.get("n_total", "")
            row["n_used"] = item.get("n_used", "")
            row["h_res_mean"] = item.get("h_res_mean", "")
            row["h_res_std"] = item.get("h_res_std", "")
            row["h_res_sem"] = item.get("h_res_sem", "")
            row["h_res_wmean"] = item.get("h_res_wmean", "")
            row["h_res_wsem"] = item.get("h_res_wsem", "")
            row["h_res_err_total"] = item.get("h_res_err_total", "")
            row["delta_mean"] = item.get("delta_mean", "")
            row["delta_std"] = item.get("delta_std", "")
            row["g_mean"] = item.get("g_mean", "")
            row["g_std"] = item.get("g_std", "")
            row["area_mean"] = item.get("area_mean", "")
            row["area_std"] = item.get("area_std", "")
            row["chi2_mean"] = item.get("chi2_mean", "")
            row["included"] = item.get("included", "")
            row["rejected"] = item.get("rejected", "")
            rows.append(row)

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Failed to export:\n{exc}")
            return
        self.status_label.setText(f"Exported {len(rows)} rows")

    def _load_analysis_options(self) -> AnalysisOptions:
        options = AnalysisOptions()
        options.expected_peaks = self._settings_int("analysis/expected_peaks", options.expected_peaks)
        options.peak_method = str(self._settings.value("analysis/peak_method", options.peak_method))
        options.show_dhpp = self._settings_bool("analysis/show_dhpp", options.show_dhpp)
        options.show_fwhm = self._settings_bool("analysis/show_fwhm", options.show_fwhm)
        return options

    def _save_analysis_options(self) -> None:
        opts = self._analysis_options
        self._settings.setValue("analysis/expected_peaks", opts.sanitized_expected())
        self._settings.setValue("analysis/peak_method", opts.peak_method)
        self._settings.setValue("analysis/show_dhpp", opts.show_dhpp)
        self._settings.setValue("analysis/show_fwhm", opts.show_fwhm)

    def _settings_bool(self, key: str, default: bool) -> bool:
        val = self._settings.value(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        return str(val).strip().lower() in {"1", "true", "yes", "on"}

    def _settings_int(self, key: str, default: int) -> int:
        val = self._settings.value(key)
        if val is None:
            return default
        try:
            return int(val)
        except Exception:
            return default

    def _normalized_peak_method(self, method: str) -> str:
        method = str(method).strip().lower()
        if method not in {"auto", "zero", "curvature"}:
            return "auto"
        return method

    def _collect_critical_points(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        pos_points: set[tuple[float, float]] = set()
        neg_points: set[tuple[float, float]] = set()
        for row in rows:
            try:
                pos_x = float(row.get("pos_x"))
                pos_y = float(row.get("pos_y", 0.0))
                neg_x = float(row.get("neg_x"))
                neg_y = float(row.get("neg_y", 0.0))
            except (TypeError, ValueError):
                continue
            pos_points.add((pos_x, pos_y))
            neg_points.add((neg_x, neg_y))
        return (sorted(pos_points), sorted(neg_points))

    def _apply_theme(self, theme: str) -> None:
        self._theme = normalize_theme(theme)
        self.setStyleSheet(stylesheet_for(self._theme))
        self.plot_widget.set_theme(self._theme)
        self.theme_btn.setText(f"Theme: {self._theme.title()}")

    def _toggle_theme(self) -> None:
        self._apply_theme("light" if self._theme == "dark" else "dark")

    def _restore_window_settings(self) -> None:
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            try:
                self.restoreGeometry(geometry)
            except Exception:
                pass
        state = self._settings.value("window/state")
        if state is not None:
            try:
                self.restoreState(state)
            except Exception:
                pass
        splitter_sizes = self._settings.value("window/main_splitter_sizes")
        if isinstance(splitter_sizes, list):
            try:
                self.main_splitter.setSizes([int(v) for v in splitter_sizes])
            except Exception:
                pass

    def _save_window_settings(self) -> None:
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())
        self._settings.setValue("window/main_splitter_sizes", self.main_splitter.sizes())
        self._settings.setValue("ui/theme", self._theme)
        self._settings.setValue("paths/last_dir", self._last_dir)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_window_settings()
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            self._pipeline_thread.quit()
            self._pipeline_thread.wait(1500)
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(1500)
        return super().closeEvent(event)
