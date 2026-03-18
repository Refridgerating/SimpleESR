"""Qt dialog that collects parameters before running batch analysis."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from .types import AnalysisOptions


class AnalysisWizard(QDialog):
    """Simple wizard that lets the user tweak analysis parameters."""

    def __init__(self, *, parent=None, initial: AnalysisOptions | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analysis Wizard")
        self.setModal(True)
        self._options: AnalysisOptions = replace(initial) if initial else AnalysisOptions()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QLabel(
            "Adjust the automatic analysis to match your spectra.\n"
            "Expected peaks and visibility toggles help guide the pipeline."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)  # type: ignore[name-defined]
        layout.addLayout(form)

        self._peaks_spin = QSpinBox(self)
        self._peaks_spin.setRange(2, 24)
        self._peaks_spin.setSingleStep(2)
        self._peaks_spin.setValue(self._options.sanitized_expected())
        form.addRow("Expected extrema", self._peaks_spin)

        self._method_combo = QComboBox(self)
        self._method_combo.addItem("Auto (choose best)", "auto")
        self._method_combo.addItem("Derivative (zero crossings)", "zero")
        self._method_combo.addItem("Absorption (curvature)", "curvature")
        current_method = self._options.peak_method.lower()
        idx = max(0, self._method_combo.findData(current_method))
        self._method_combo.setCurrentIndex(idx)
        form.addRow("Peak finder", self._method_combo)

        self._dhpp_toggle = QCheckBox("Show Delta H_pp results")
        self._dhpp_toggle.setChecked(self._options.show_dhpp)
        form.addRow("Delta H_pp", self._dhpp_toggle)

        self._fwhm_toggle = QCheckBox("Show FWHM results")
        self._fwhm_toggle.setChecked(self._options.show_fwhm)
        form.addRow("FWHM", self._fwhm_toggle)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._update_summary()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._peaks_spin.valueChanged.connect(lambda _: self._update_summary())
        self._method_combo.currentIndexChanged.connect(lambda _: self._update_summary())

    def _update_summary(self) -> None:
        method_label = self._method_combo.currentText()
        peaks = self._peaks_spin.value()
        self._summary.setText(
            f"Running analysis with {peaks} extrema ({peaks // 2} peak pairs) "
            f"using {method_label.lower()} detection."
        )

    def _on_accept(self) -> None:
        self._options.expected_peaks = self._peaks_spin.value()
        self._options.peak_method = str(self._method_combo.currentData())
        self._options.show_dhpp = self._dhpp_toggle.isChecked()
        self._options.show_fwhm = self._fwhm_toggle.isChecked()
        self.accept()

    def selected_options(self) -> AnalysisOptions:
        return self._options
