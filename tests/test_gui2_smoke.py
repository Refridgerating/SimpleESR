import numpy as np
import pytest

from esr_lab.gui2.types import AnalysisOptions
from esr_lab.gui2.viewmodels import ESRViewModel, filter_analysis_rows


def test_gui2_viewmodel_trace_labels_and_switching():
    vm = ESRViewModel()
    field = np.linspace(-1.0, 1.0, 5)
    vm.add_trace(label="a.csv", field=field, intensity=np.zeros_like(field), metadata={})
    vm.add_trace(label="b.csv", field=field, intensity=np.ones_like(field), metadata={})

    assert vm.trace_labels() == ["a.csv", "b.csv"]
    assert vm.state.active_index == 1
    vm.set_active(0)
    active = vm.active_trace()
    assert active is not None
    assert active.label == "a.csv"


def test_gui2_viewmodel_visibility_toggles():
    vm = ESRViewModel()
    field = np.linspace(-1.0, 1.0, 5)
    vm.add_trace(label="a.csv", field=field, intensity=np.zeros_like(field), metadata={})
    vm.add_trace(label="b.csv", field=field, intensity=np.ones_like(field), metadata={})
    vm.add_trace(label="c.csv", field=field, intensity=np.full_like(field, 2.0), metadata={})

    assert vm.visible_indices() == [0, 1, 2]
    vm.set_visible(0, False)
    vm.set_visible(2, False)
    assert vm.visible_indices() == [1]
    vm.set_visible(2, True)
    assert vm.visible_indices() == [1, 2]


def test_gui2_viewmodel_fit_overlay_from_payload():
    vm = ESRViewModel()
    field = np.linspace(-5.0, 5.0, 1001)
    vm.add_trace(label="trace.csv", field=field, intensity=np.zeros_like(field), metadata={})
    payload = {
        "peaks": [(450, 550)],
        "widths": [
            {"peak": 1, "pos_idx": 450, "neg_idx": 550, "peak_to_peak": 1.0, "fwhm": 1.732}
        ],
        "fits": [
            {
                "peak": 1,
                "kind": "derivative",
                "h_res": 0.0,
                "delta": 1.0,
                "A": 2.0,
                "B": 0.5,
                "chi2": 0.01,
            }
        ],
    }
    vm.apply_pipeline_payload(0, payload)
    overlay = vm.fit_overlay(0)
    assert overlay is not None
    assert overlay.shape == field.shape
    assert np.any(np.abs(overlay) > 0.0)


def test_analysis_options_and_filters():
    opts = AnalysisOptions(expected_peaks=3, show_dhpp=True, show_fwhm=False)
    assert opts.sanitized_expected() == 4
    allowed = opts.allowed_labels()
    assert allowed == {"dh_pp"}
    rows = [
        {"analysis": "dH_pp", "pos_x": 1.0, "pos_y": 2.0, "neg_x": -1.0, "neg_y": -2.0},
        {"analysis": "FWHM", "pos_x": 1.2, "pos_y": 1.8, "neg_x": -1.1, "neg_y": -1.9},
    ]
    filtered = filter_analysis_rows(rows, allowed)
    assert len(filtered) == 1
    assert filtered[0]["analysis"] == "dH_pp"

    opts.show_dhpp = False
    opts.show_fwhm = False
    assert opts.allowed_labels() == set()
    assert filter_analysis_rows(rows, opts.allowed_labels()) == []


def test_gui2_analysis_table_tooltip_formulas():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt

    from esr_lab.gui2.table_models import analysis_table_model

    model = analysis_table_model()
    model.set_rows(
        [
            {
                "analysis": "dH_pp",
                "peak": 1,
                "pos_x": 1.0,
                "neg_x": -1.0,
                "width": 2.0,
            }
        ]
    )
    tip = model.data(model.index(0, 4), Qt.ToolTipRole)
    assert isinstance(tip, str)
    assert "dH_pp" in tip

    model.set_rows(
        [
            {
                "analysis": "FWHM",
                "peak": 1,
                "pos_x": 1.0,
                "neg_x": -1.0,
                "width": 3.464,
            }
        ]
    )
    tip = model.data(model.index(0, 4), Qt.ToolTipRole)
    assert isinstance(tip, str)
    assert "sqrt(3)" in tip


def test_gui2_fit_table_tooltip_formulas():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt

    from esr_lab.gui2.table_models import fit_table_model

    model = fit_table_model()
    header_tip = model.headerData(12, Qt.Horizontal, Qt.ToolTipRole)
    assert isinstance(header_tip, str)
    assert "MAD" in header_tip

    model.set_rows(
        [
            {
                "peak": 1,
                "kind": "derivative",
                "h_res": 0.0,
                "delta": 1.0,
                "A": 2.0,
                "B": 0.1,
                "g": 2.0,
                "g_err": 0.01,
                "g_err_pct": 0.5,
                "chi2": 1.1,
                "stderr": (0.1, 0.1, 0.1, 0.1),
            }
        ]
    )
    cell_tip = model.data(model.index(0, 8), Qt.ToolTipRole)
    assert isinstance(cell_tip, str)
    assert "100" in cell_tip


def test_pipeline_worker_contract(monkeypatch):
    pytest.importorskip("PySide6")
    from esr_lab.gui2 import workers
    from esr_lab.spectrum import ESRSpectrum

    field = np.linspace(-1.0, 1.0, 5)
    spectra = [ESRSpectrum(field=field, intensity=np.zeros_like(field), metadata={"Frequency": 9.5})]

    def fake_analyze_spectrum(*args, **kwargs):
        return {"peaks": [], "widths": [], "fits": [{"peak": 1, "kind": "derivative", "h_res": 0.0}]}

    monkeypatch.setattr(workers, "analyze_spectrum", fake_analyze_spectrum)

    worker = workers.PipelineWorker(
        spectra=spectra,
        indices=[0],
        expected_peaks=4,
        peak_method="auto",
    )
    captured: dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("finished", payload))
    worker.failed.connect(lambda message: captured.setdefault("failed", message))
    worker.run()

    assert "failed" not in captured
    assert "finished" in captured
    rows = captured["finished"]
    assert isinstance(rows, list)
    assert rows and rows[0][0] == 0


def test_file_load_worker_contract(monkeypatch):
    pytest.importorskip("PySide6")
    from esr_lab.gui2 import workers
    from esr_lab.spectrum import ESRSpectrum

    field = np.linspace(-1.0, 1.0, 3)

    def fake_load_csv(path):
        return ESRSpectrum(field=field, intensity=np.zeros_like(field), metadata={"src": str(path)})

    monkeypatch.setattr(workers.ESRLoader, "load_csv", fake_load_csv)

    worker = workers.FileLoadWorker(paths=["a.csv", "b.csv"])
    captured: dict[str, object] = {}
    worker.finished.connect(lambda loaded, errors: captured.setdefault("done", (loaded, errors)))
    worker.failed.connect(lambda message: captured.setdefault("failed", message))
    worker.run()

    assert "failed" not in captured
    assert "done" in captured
    loaded, errors = captured["done"]
    assert len(loaded) == 2
    assert errors == []
