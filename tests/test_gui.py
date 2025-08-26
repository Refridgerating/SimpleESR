import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from esr_lab import gui
from esr_lab.spectrum import ESRSpectrum


def test_gui_main(monkeypatch, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    class DummyTk:
        def withdraw(self):
            pass

    monkeypatch.setattr(gui.tk, "Tk", lambda: DummyTk())
    monkeypatch.setattr(gui.filedialog, "askopenfilenames", lambda **kwargs: [str(csv_file)])

    with patch("esr_lab.gui.SpanPeakSelector") as selector_mock:
        gui.main()
        selector_mock.assert_called_once()
        selector_mock.return_value.show.assert_called_once()


def test_span_selector_analysis():
    spectrum = ESRSpectrum(field=np.arange(10.0), intensity=np.zeros(10))
    selector = gui.SpanPeakSelector(spectrum)

    with patch("esr_lab.gui.find_peak", return_value=(1, 3)) as fp, \
        patch("esr_lab.gui.calc_fwhm", return_value=0.5) as cf, \
        patch("esr_lab.gui.messagebox.showinfo") as info:
        selector.analysis_func = cf
        selector.onselect(1.0, 2.0)

        assert selector.ranges == [(1.0, 2.0)]
        fp.assert_called_once()
        cf.assert_called_once()
        info.assert_called_once()
        assert selector.results == [
            {
                "analysis": "FWHM",
                "pos_x": 1.0,
                "pos_y": 0.0,
                "neg_x": 3.0,
                "neg_y": 0.0,
                "width": 0.5,
            }
        ]


def test_peak_to_peak_analysis():
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)

    fig, selector.ax = plt.subplots()
    with patch("esr_lab.gui.find_peak", return_value=(1, 3)) as fp, \
        patch("esr_lab.gui.calc_peak_to_peak", return_value=2.0) as cpp, \
        patch("esr_lab.gui.messagebox.showinfo") as info:
        selector.start_peak_to_peak()
        selector.onselect(0.0, 4.0)
        fp.assert_called_once()
        cpp.assert_called_once()
        info.assert_called_once()
        assert selector.results == [
            {
                "analysis": "\u0394H_pp",
                "pos_x": 1.0,
                "pos_y": 0.0,
                "neg_x": 3.0,
                "neg_y": 0.0,
                "width": 2.0,
            }
        ]
    plt.close(fig)


def test_results_persist_across_analyses():
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()

    with patch("esr_lab.gui.find_peak", return_value=(1, 3)) as fp, \
        patch("esr_lab.gui.calc_fwhm", return_value=0.5) as cf, \
        patch("esr_lab.gui.calc_peak_to_peak", return_value=2.0) as cpp, \
        patch("esr_lab.gui.messagebox.showinfo"):
        selector.start_analysis(analysis_func=cf)
        selector.onselect(1.0, 2.0)
        selector.start_peak_to_peak()
        selector.onselect(0.0, 4.0)
        assert fp.call_count == 2
        cf.assert_called_once()
        cpp.assert_called_once()
        assert selector.results == [
            {
                "analysis": "FWHM",
                "pos_x": 1.0,
                "pos_y": 0.0,
                "neg_x": 3.0,
                "neg_y": 0.0,
                "width": 0.5,
            },
            {
                "analysis": "\u0394H_pp",
                "pos_x": 1.0,
                "pos_y": 0.0,
                "neg_x": 3.0,
                "neg_y": 0.0,
                "width": 2.0,
            },
        ]
    plt.close(fig)


def test_multi_trace_results_isolated():
    spec1 = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    spec2 = ESRSpectrum(field=np.arange(5.0), intensity=np.ones(5))
    selector = gui.SpanPeakSelector([spec1, spec2], labels=["one", "two"])
    fig, selector.ax = plt.subplots()

    tree = MagicMock()
    tree.get_children.return_value = []
    selector.tree = tree
    ltree = MagicMock()
    ltree.get_children.return_value = []
    selector.lorentz_tree = ltree

    selector.peak_slider = MagicMock()

    with patch("esr_lab.gui.find_peak", return_value=(1, 3)), \
        patch("esr_lab.gui.calc_fwhm", return_value=0.5), \
        patch("esr_lab.gui.messagebox.showinfo"):
        selector.start_analysis()
        selector.onselect(1.0, 2.0)

        # switch to second trace
        selector.trace_var = MagicMock(get=lambda: "two")
        selector._on_trace_change()
        selector.start_analysis()
        selector.onselect(1.0, 2.0)

    assert len(selector.results_all[0]) == 1
    assert len(selector.results_all[1]) == 1
    plt.close(fig)


def test_lorentzian_fit_overlay():
    spectrum = ESRSpectrum(field=np.linspace(-1, 1, 5), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.selected_peak = -0.5
    with patch(
        "esr_lab.gui.fit_lorentzian_derivative", return_value=(0.0, 1.0, 1.0, 0.0)
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=True) as ask:
        selector.fit_lorentzian()
        fit.assert_called_once()
        ask.assert_called_once()
        assert len(selector.ax.lines) == 2
    plt.close(fig)

    fig, selector.ax = plt.subplots()
    selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.selected_peak = 0.5
    with patch(
        "esr_lab.gui.fit_lorentzian_derivative", return_value=(0.0, 1.0, 1.0, 0.0)
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=False) as ask:
        selector.fit_lorentzian()
        fit.assert_called_once()
        ask.assert_called_once()
        assert len(selector.ax.lines) == 1
    plt.close(fig)


def test_lorentzian_fit_results_tabulated():
    spectrum = ESRSpectrum(field=np.linspace(-1, 1, 5), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.selected_peak = 0.0
    selector.lorentz_tree = MagicMock()
    with patch(
        "esr_lab.gui.fit_lorentzian_derivative",
        return_value=(0.0, 1.0, 2.0, 3.0),
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=True):
        selector.fit_lorentzian()
        fit.assert_called_once()
    assert selector.lorentz_results == [
        {"peak": 0.0, "h_res": 0.0, "delta": 1.0, "A": 2.0, "B": 3.0}
    ]
    selector.lorentz_tree.insert.assert_called_once()
    plt.close(fig)


def test_toolbar_has_default_tools_without_subplots():
    tools = [item[0] for item in gui.NavigationToolbarNoSubplots.toolitems if item]
    assert "Subplots" not in tools
    assert "Pan" in tools and "Zoom" in tools
    assert "Edit" in tools


def test_filter_ticks_respects_limits():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_xticks(gui._filter_ticks([-50, 0, 10], 0, 10))
    ax.set_ylim(0, 1)
    ax.set_yticks(gui._filter_ticks([-1, 0, 1], 0, 1))
    assert ax.get_xlim() == (0.0, 10.0)
    assert ax.get_ylim() == (0.0, 1.0)
    plt.close(fig)


def test_table_columns_centered(monkeypatch):
    """Ensure tabulated values are centred for readability."""

    spectrum = ESRSpectrum(field=np.linspace(-1, 1, 5), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)

    column_cfg: dict[str, dict] = {}

    class DummyTree:
        def heading(self, *args, **kwargs):
            pass

        def column(self, col, **cfg):
            column_cfg[col] = cfg

        def pack(self, *args, **kwargs):
            pass

        def get_children(self):
            return []

    class DummyFrame:
        def pack(self, *args, **kwargs):
            pass

    class DummyTk:
        def title(self, *args, **kwargs):
            pass

        def mainloop(self):
            pass

    monkeypatch.setattr(gui.tk, "Tk", lambda: DummyTk())
    monkeypatch.setattr(gui.tk, "Frame", lambda *a, **k: DummyFrame())
    monkeypatch.setattr(gui.tk, "Button", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.tk, "Scale", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.tk, "Label", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.ttk, "Treeview", lambda *a, **k: DummyTree())
    monkeypatch.setattr(gui, "FigureCanvasTkAgg", MagicMock())
    monkeypatch.setattr(gui, "NavigationToolbarNoSubplots", MagicMock())

    selector.show()

    # All columns created in show should be centred
    assert column_cfg
    assert all(cfg.get("anchor") == gui.tk.CENTER for cfg in column_cfg.values())

