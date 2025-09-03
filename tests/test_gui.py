import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import sympy as sp

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
                "peak": 1,
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
        patch("esr_lab.gui.messagebox.showinfo") as info, \
        patch("esr_lab.gui.simpledialog.askinteger", return_value=1):
        selector.start_peak_to_peak()
        selector.onselect(0.0, 4.0)
        fp.assert_called_once()
        cpp.assert_called_once()
        info.assert_called_once()
        assert selector.results == [
            {
                "analysis": "\u0394H_pp",
                "peak": 1,
                "pos_x": 1.0,
                "pos_y": 0.0,
                "neg_x": 3.0,
                "neg_y": 0.0,
                "width": 2.0,
            }
        ]
    plt.close(fig)


def test_fwhm_differs_from_peak_to_peak():
    field = np.linspace(-5, 5, 10001)
    intensity = field * np.exp(-field**2 / 2)
    spec = ESRSpectrum(field=field, intensity=intensity)
    selector = gui.SpanPeakSelector(spec)

    with patch("esr_lab.gui.messagebox.showinfo"):
        selector.onselect(-2.0, 2.0)
    fwhm = selector.results[0]["width"]

    selector.results.clear()
    selector.analysis_func = gui.calc_peak_to_peak
    selector.analysis_label = "\u0394H_pp"
    with patch("esr_lab.gui.messagebox.showinfo"):
        selector.onselect(-2.0, 2.0)
    dhpp = selector.results[0]["width"]

    assert not np.isclose(fwhm, dhpp)
    assert np.isclose(fwhm, np.sqrt(3.0) * dhpp, atol=1e-3)


def test_peak_finder_marks_peaks():
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.array([0, 1, 0, -1, 0]))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()

    markers: list[MagicMock] = []

    def fake_plot(*args, **kwargs):
        m = MagicMock()
        markers.append(m)
        return [m]

    selector.ax.plot = MagicMock(side_effect=fake_plot)

    with patch("esr_lab.gui.auto_peak_finder", return_value=[(1, 3)]), \
        patch("esr_lab.gui.simpledialog.askinteger", return_value=2), \
        patch("esr_lab.gui.messagebox.askyesno", return_value=True), \
        patch("esr_lab.gui.messagebox.showinfo"):
        selector.peak_finder()

    assert selector.ax.plot.call_count == 2
    assert all(m.remove.call_count == 1 for m in markers)
    assert selector.auto_peaks == [(1, 3)]
    plt.close(fig)


def test_peak_finder_tabulates_positions():
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.array([0, 1, 0, -1, 0]))
    selector = gui.SpanPeakSelector(spectrum)
    selector.ax = None
    tree = MagicMock()
    tree.get_children.return_value = []
    selector.peak_tree = tree
    with patch("esr_lab.gui.auto_peak_finder", return_value=[(1, 3)]), \
        patch("esr_lab.gui.simpledialog.askinteger", return_value=2), \
        patch("esr_lab.gui.messagebox.askyesno", return_value=True), \
        patch("esr_lab.gui.messagebox.showinfo"):
        selector.peak_finder()
    tree.insert.assert_called_once()
    assert tree.insert.call_args.kwargs["values"] == (
        selector.labels[0],
        "1",
        "1.000",
        "3.000",
    )


def test_peak_finder_uses_auto_method():
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.array([0, 1, 0, -1, 0]))
    selector = gui.SpanPeakSelector(spectrum)
    selector.ax = None
    captured: dict[str, str] = {}

    def fake_peak_finder(field, intensity, expected=4, width=15.0, method="zero"):
        captured["method"] = method
        return [(1, 3)]

    with patch("esr_lab.gui.auto_peak_finder", new=fake_peak_finder), \
        patch("esr_lab.gui.simpledialog.askinteger", return_value=2), \
        patch("esr_lab.gui.messagebox.askyesno", return_value=True), \
        patch("esr_lab.gui.messagebox.showinfo"):
        selector.peak_finder()

    assert captured["method"] == "auto"


def test_results_persist_across_analyses():
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()

    with patch("esr_lab.gui.find_peak", return_value=(1, 3)) as fp, \
        patch("esr_lab.gui.calc_fwhm", return_value=0.5) as cf, \
        patch("esr_lab.gui.calc_peak_to_peak", return_value=2.0) as cpp, \
        patch("esr_lab.gui.messagebox.showinfo"), \
        patch("esr_lab.gui.simpledialog.askinteger", return_value=1):
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
                "peak": 1,
                "pos_x": 1.0,
                "pos_y": 0.0,
                "neg_x": 3.0,
                "neg_y": 0.0,
                "width": 0.5,
            },
            {
                "analysis": "\u0394H_pp",
                "peak": 1,
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
        patch("esr_lab.gui.messagebox.showinfo"), \
        patch("esr_lab.gui.simpledialog.askinteger", return_value=1):
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


def test_baseline_correction_manual():
    spectrum = ESRSpectrum(
        field=np.array([0.0, 1.0, 2.0]), intensity=np.array([0.0, 100.0, 0.0])
    )
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    (line,) = selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.trace_lines = [line]
    selector.ax.figure.canvas.draw_idle = MagicMock()

    corrected = np.zeros(3)
    with patch(
        "esr_lab.gui.SpanPeakSelector._get_baseline_options",
        return_value=(True, False),
    ), patch(
        "esr_lab.gui.plt.ginput",
        return_value=[(1.9, 100.0), (0.1, -50.0)],
    ), patch(
        "esr_lab.gui.baseline_correct", return_value=(corrected, corrected)
    ) as bc:
        selector.baseline_correction()
        bc.assert_called_once()
        # ensure closest x-coordinate is used regardless of y
        assert bc.call_args.kwargs["points"] == [(2.0, 0.0), (0.0, 0.0)]
        assert len(selector.spectra) == 2
        assert np.allclose(selector.spectra[-1].intensity, corrected)
        assert len(selector.trace_lines) == 2

    plt.close(fig)


def test_baseline_correction_auto_confirm():
    spectrum = ESRSpectrum(
        field=np.array([0.0, 1.0, 2.0]), intensity=np.array([1.0, 2.0, 3.0])
    )
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    (line,) = selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.trace_lines = [line]
    selector.ax.figure.canvas.draw_idle = MagicMock()

    corrected = np.zeros(3)
    baseline = np.ones(3)
    with patch(
        "esr_lab.gui.SpanPeakSelector._get_baseline_options",
        return_value=(True, True),
    ), patch(
        "esr_lab.gui.baseline_correct", return_value=(corrected, baseline)
    ) as bc, patch(
        "esr_lab.gui.messagebox.askyesno", return_value=True
    ) as ask:
        selector.baseline_correction()
        bc.assert_called_once()
        ask.assert_called_once()
        assert len(selector.spectra) == 2
        assert np.allclose(selector.spectra[-1].intensity, corrected)
        assert len(selector.ax.lines) == 2

    plt.close(fig)


def test_baseline_correction_auto_cancel():
    spectrum = ESRSpectrum(
        field=np.array([0.0, 1.0, 2.0]), intensity=np.array([1.0, 2.0, 3.0])
    )
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    (line,) = selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.trace_lines = [line]
    selector.ax.figure.canvas.draw_idle = MagicMock()

    corrected = np.zeros(3)
    baseline = np.ones(3)
    with patch(
        "esr_lab.gui.SpanPeakSelector._get_baseline_options",
        return_value=(False, True),
    ), patch(
        "esr_lab.gui.baseline_correct", return_value=(corrected, baseline)
    ) as bc, patch(
        "esr_lab.gui.messagebox.askyesno", return_value=False
    ) as ask:
        selector.baseline_correction()
        bc.assert_called_once()
        ask.assert_called_once()
        assert len(selector.spectra) == 1
        assert len(selector.ax.lines) == 1

    plt.close(fig)


def test_baseline_correction_manual_cancel():
    spectrum = ESRSpectrum(
        field=np.array([0.0, 1.0, 2.0]), intensity=np.array([1.0, 2.0, 3.0])
    )
    selector = gui.SpanPeakSelector(spectrum)
    selector.root = object()
    fig, selector.ax = plt.subplots()
    (line,) = selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.trace_lines = [line]
    selector.ax.figure.canvas.draw_idle = MagicMock()

    editor = MagicMock()
    editor.get_points.return_value = [(0.0, 1.0), (2.0, 3.0)]
    editor.clear_artists = MagicMock()
    editor.disconnect = MagicMock()

    class DummyVar:
        def __init__(self, value=None):
            self.value = value

        def set(self, value):
            self.value = value

        def get(self):
            return self.value

    class DummyWidget:
        def pack(self, *args, **kwargs):
            pass

    class DummyDialog:
        def __init__(self, master=None):
            pass

        def title(self, *args, **kwargs):
            pass

        def protocol(self, *args, **kwargs):
            pass

        def destroy(self):
            pass

        def wait_window(self):
            pass

    with patch(
        "esr_lab.gui.SpanPeakSelector._get_baseline_options", return_value=(True, False)
    ), patch(
        "esr_lab.gui.BaselinePointEditor", return_value=editor
    ), patch(
        "esr_lab.gui.tk.StringVar", side_effect=lambda *a, **k: DummyVar()
    ), patch(
        "esr_lab.gui.tk.BooleanVar", side_effect=lambda *a, **k: DummyVar()
    ), patch(
        "esr_lab.gui.tk.Label", side_effect=lambda *a, **k: DummyWidget()
    ), patch(
        "esr_lab.gui.tk.Button", side_effect=lambda *a, **k: DummyWidget()
    ), patch(
        "esr_lab.gui.tk.Frame", side_effect=lambda *a, **k: DummyWidget()
    ), patch(
        "esr_lab.gui.tk.Toplevel", side_effect=lambda *a, **k: DummyDialog()
    ), patch(
        "esr_lab.gui.messagebox.showwarning"
    ) as warn, patch(
        "esr_lab.gui.baseline_correct"
    ) as bc:
        selector.baseline_correction()
        bc.assert_not_called()
        editor.clear_artists.assert_called_once()
        warn.assert_not_called()

    plt.close(fig)


def test_baseline_editor_clear():
    field = np.linspace(0.0, 2.0, 3)
    intensity = np.zeros(3)
    fig, ax = plt.subplots()
    editor = gui.BaselinePointEditor(ax, field, intensity, degree=1)

    class Event:
        def __init__(self, x):
            self.inaxes = ax
            self.xdata = x
            self.ydata = 0.0
            self.x, self.y = ax.transData.transform((x, 0.0))

    editor.on_click(Event(field[0]))
    editor.on_click(Event(field[-1]))
    assert len(editor.get_points()) == 2
    baseline_line = editor.baseline_line
    markers = editor.point_artists.copy()
    assert baseline_line in ax.lines
    editor.clear_artists()
    assert baseline_line not in ax.lines
    for m in markers:
        assert m not in ax.lines
    plt.close(fig)


def test_lorentzian_fit_overlay():
    spectrum = ESRSpectrum(field=np.linspace(-1, 1, 5), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    (line0,) = selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.trace_lines = [line0]
    selector.selected_peak = -0.5
    with patch(
        "esr_lab.gui.fit_lorentzian_derivative",
        return_value=((0.0, 1.0, 1.0, 0.0), {"chi2": 0.0, "stderr": (0, 0, 0, 0), "residuals": np.zeros(5)}),
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=True) as ask, patch(
        "esr_lab.gui.simpledialog.askinteger", return_value=1
    ), patch("esr_lab.gui.plot_residuals") as plot_res:
        selector.fit_lorentzian()
        fit.assert_called_once()
        ask.assert_called_once()
        plot_res.assert_called_once()
        assert len(selector.ax.lines) == 2
        assert len(selector.trace_lines) == 2
        line = selector.trace_lines[1]
        selector._toggle_trace(1, False)
        assert not line.get_visible()
    plt.close(fig)

    fig, selector.ax = plt.subplots()
    (line0,) = selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.trace_lines = [line0]
    selector.selected_peak = 0.5
    with patch(
        "esr_lab.gui.fit_lorentzian_derivative",
        return_value=((0.0, 1.0, 1.0, 0.0), {"chi2": 0.0, "stderr": (0, 0, 0, 0), "residuals": np.zeros(5)}),
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=False) as ask, patch(
        "esr_lab.gui.simpledialog.askinteger", return_value=1
    ), patch("esr_lab.gui.plot_residuals"):
        selector.fit_lorentzian()
        fit.assert_called_once()
        ask.assert_called_once()
        assert len(selector.ax.lines) == 1
        # No new trace added when fit is rejected
        assert len(selector.trace_lines) == 1
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
        return_value=((0.0, 1.0, 2.0, 3.0), {"chi2": 0.0, "stderr": (0, 0, 0, 0), "residuals": np.zeros(5)}),
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=True), patch(
        "esr_lab.gui.simpledialog.askinteger", return_value=1
    ), patch("esr_lab.gui.plot_residuals"):
        selector.fit_lorentzian()
        fit.assert_called_once()
    assert selector.lorentz_results == [
        {"analysis": "Lorentzian", "peak": 1, "h_res": 0.0, "delta": 1.0, "A": 2.0, "B": 3.0}
    ]
    selector.lorentz_tree.insert.assert_called_once()
    plt.close(fig)


def test_toolbar_has_default_tools_without_subplots():
    tools = [item[0] for item in gui.NavigationToolbarNoSubplots.toolitems if item]
    assert "Subplots" not in tools
    assert "Pan" in tools and "Zoom" in tools


def test_toolbar_selects_active_line():
    fig, ax = plt.subplots()
    line1, = ax.plot([0, 1], [0, 1])
    line2, = ax.plot([0, 1], [1, 0])

    toolbar = gui.NavigationToolbarNoSubplots.__new__(gui.NavigationToolbarNoSubplots)
    toolbar.get_active_index = lambda: 1
    assert toolbar._get_selected_line(ax) is line2
    toolbar.get_active_index = lambda: 0
    assert toolbar._get_selected_line(ax) is line1
    plt.close(fig)


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


def test_window_maximized_on_show(monkeypatch):
    spectrum = ESRSpectrum(field=np.linspace(-1, 1, 5), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)

    class DummyTree:
        def heading(self, *args, **kwargs):
            pass

        def column(self, *args, **kwargs):
            pass

        def pack(self, *args, **kwargs):
            pass

        def get_children(self):
            return []

    class DummyFrame:
        def pack(self, *args, **kwargs):
            pass

    class DummyTk:
        def __init__(self):
            self.state_called = False

        def title(self, *args, **kwargs):
            pass

        def state(self, *_args):
            self.state_called = True

        def update_idletasks(self):
            pass

        def mainloop(self):
            pass

    root = DummyTk()
    monkeypatch.setattr(gui.tk, "Tk", lambda: root)
    monkeypatch.setattr(gui.tk, "Frame", lambda *a, **k: DummyFrame())
    monkeypatch.setattr(gui.tk, "Button", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.tk, "Scale", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.tk, "Label", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.tk, "Canvas", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.tk, "Scrollbar", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gui.ttk, "Treeview", lambda *a, **k: DummyTree())
    monkeypatch.setattr(gui, "FigureCanvasTkAgg", MagicMock())
    monkeypatch.setattr(gui, "NavigationToolbarNoSubplots", MagicMock())

    selector.show()

    assert root.state_called


def test_spectra_comparison_tabulated():
    spec1 = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    spec2 = ESRSpectrum(field=np.arange(5.0), intensity=np.ones(5))
    selector = gui.SpanPeakSelector([spec1, spec2])

    selector.compare_tree = MagicMock()
    selector.compare_tree.get_children.return_value = []

    selector.results_all = [
        [
            {"analysis": "FWHM", "peak": 1, "pos_x": 0, "pos_y": 0, "neg_x": 0, "neg_y": 0, "width": 1.0},
            {"analysis": "\u0394H_pp", "peak": 1, "pos_x": 0, "pos_y": 0, "neg_x": 0, "neg_y": 0, "width": 3.0},
        ],
        [
            {"analysis": "FWHM", "peak": 1, "pos_x": 0, "pos_y": 0, "neg_x": 0, "neg_y": 0, "width": 2.0},
            {"analysis": "\u0394H_pp", "peak": 1, "pos_x": 0, "pos_y": 0, "neg_x": 0, "neg_y": 0, "width": 5.0},
        ],
    ]
    selector.lorentz_all = [
        [
            {"analysis": "Lorentzian", "peak": 1, "h_res": 10.0, "delta": 0, "A": 0, "B": 0}
        ],
        [
            {"analysis": "Lorentzian", "peak": 1, "h_res": 12.0, "delta": 0, "A": 0, "B": 0}
        ],
    ]

    with patch.object(selector, "_prompt_traces", return_value=(0, 1)), patch(
        "esr_lab.gui.messagebox.showinfo"
    ):
        selector.compare_spectra()

    calls = selector.compare_tree.insert.call_args_list
    assert calls
    values = [c.kwargs["values"] for c in calls]
    assert ("FWHM P1", "1.000", "2.000", "-1.000") in values
    assert ("\u0394H_pp P1", "3.000", "5.000", "-2.000") in values
    assert ("H_res P1", "10.000", "12.000", "-2.000") in values


def test_integrate_trace_plots_absorption():
    field = np.linspace(0, 4, 5)
    intensity = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    spec = ESRSpectrum(field=field, intensity=intensity)
    selector = gui.SpanPeakSelector(spec)
    fig, selector.ax = plt.subplots()

    # Initial derivative trace
    (line0,) = selector.ax.plot(field, intensity)
    selector.trace_lines = [line0]
    selector.integrate_trace()

    from scipy.integrate import cumulative_trapezoid

    expected = cumulative_trapezoid(intensity, field, initial=0)
    expected -= np.mean(expected)

    line = selector.trace_lines[-1]
    assert np.allclose(line.get_ydata(), expected)
    assert line.get_label() == "Trace 1 (absorption)"
    assert selector.labels[-1] == "Trace 1 (absorption)"
    assert len(selector.spectra) == 2
    plt.close(fig)


def test_integrate_trace_adds_toggle():
    field = np.linspace(0, 4, 5)
    intensity = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    spec = ESRSpectrum(field=field, intensity=intensity)
    selector = gui.SpanPeakSelector(spec)
    fig, selector.ax = plt.subplots()
    (line0,) = selector.ax.plot(field, intensity)
    selector.trace_lines = [line0]
    selector.toggle_frame = MagicMock()

    with patch("esr_lab.gui.tk.BooleanVar") as MockVar, patch(
        "esr_lab.gui.tk.Checkbutton"
    ) as MockChk:
        selector.integrate_trace()
        MockVar.assert_called_once_with(value=True)
        MockChk.assert_called_once()
        MockChk.return_value.pack.assert_called_once_with(anchor="w")
        assert selector.trace_vars[-1] is MockVar.return_value

    plt.close(fig)


def test_integrated_trace_toggle():
    field = np.linspace(0, 4, 5)
    intensity = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    spec = ESRSpectrum(field=field, intensity=intensity)
    selector = gui.SpanPeakSelector(spec)
    fig, selector.ax = plt.subplots()
    (line0,) = selector.ax.plot(field, intensity)
    selector.trace_lines = [line0]
    selector.integrate_trace()
    line = selector.trace_lines[1]
    selector.update_legend = MagicMock()
    selector.ax.figure.canvas.draw_idle = MagicMock()
    selector._toggle_trace(1, False)
    assert not line.get_visible()
    selector.update_legend.assert_called_once()
    assert selector.ax.figure.canvas.draw_idle.call_count == 2
    plt.close(fig)


def test_trace_visibility_toggle_updates_legend():
    spec1 = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    spec2 = ESRSpectrum(field=np.arange(5.0), intensity=np.ones(5))
    selector = gui.SpanPeakSelector([spec1, spec2], labels=["one", "two"])
    fig, selector.ax = plt.subplots()
    selector.trace_lines = []
    for spec in selector.spectra:
        line, = selector.ax.plot(spec.field, spec.intensity)
        selector.trace_lines.append(line)

    selector.update_legend()
    legend = selector.ax.get_legend()
    assert legend.get_draggable()
    assert len(legend.get_texts()) == 2

    selector._toggle_trace(1, False)
    assert not selector.trace_lines[1].get_visible()
    assert selector.ax.get_legend().get_draggable()
    assert len(selector.ax.get_legend().get_texts()) == 1

    selector._toggle_trace(1, True)
    assert selector.trace_lines[1].get_visible()
    assert selector.ax.get_legend().get_draggable()
    assert len(selector.ax.get_legend().get_texts()) == 2
    plt.close(fig)


def test_set_label_updates_legend():
    spec1 = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    spec2 = ESRSpectrum(field=np.arange(5.0), intensity=np.ones(5))
    selector = gui.SpanPeakSelector([spec1, spec2], labels=["one", "two"])
    fig, selector.ax = plt.subplots()
    selector.trace_lines = []
    for spec in selector.spectra:
        line, = selector.ax.plot(spec.field, spec.intensity)
        selector.trace_lines.append(line)

    selector.update_legend()
    assert [t.get_text() for t in selector.ax.get_legend().get_texts()] == ["one", "two"]

    selector._set_label(1, "new")
    selector.update_legend()
    assert selector.labels[1] == "new"
    assert [t.get_text() for t in selector.ax.get_legend().get_texts()] == ["one", "new"]
    plt.close(fig)


def test_help_menu_created(monkeypatch):
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)

    class DummyRoot:
        def __init__(self):
            self.menu = None

        def config(self, **kwargs):
            self.menu = kwargs.get("menu")

    root = DummyRoot()
    selector.root = root

    class DummyMenu:
        def __init__(self, master=None, tearoff=0):
            self.items = []
            self.cascade = None

        def add_command(self, label, command):
            self.items.append(("command", label, command))

        def add_cascade(self, label, menu):
            self.cascade = (label, menu)

    monkeypatch.setattr(gui.tk, "Menu", DummyMenu)

    selector._create_menu()

    assert isinstance(root.menu, DummyMenu)
    assert root.menu.cascade[0] == "Help"
    help_menu = root.menu.cascade[1]
    expected = [
        ("command", "Readme", selector._show_readme),
        ("command", "Workflow", selector._show_workflow),
        ("command", "Functions", selector._show_functions),
    ]
    assert help_menu.items == expected


def test_help_dialogs(monkeypatch):
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)

    info = MagicMock()
    monkeypatch.setattr(gui.messagebox, "showinfo", info)

    monkeypatch.setattr(gui.Path, "read_text", lambda self, encoding="utf-8": "readme text")
    selector._show_readme()
    info.assert_called_with("README", "readme text")

    info.reset_mock()
    selector._show_workflow()
    assert info.call_args[0][0] == "Workflow"

    info.reset_mock()
    monkeypatch.setattr(gui, "FUNCTION_DETAILS", {"f": ("desc", sp.Symbol("x") ** 2)})
    selector._show_functions()
    call = info.call_args
    assert call[0][0] == "Functions"
    lines = call[0][1].splitlines()
    assert lines[0] == "f"
    assert lines[1].strip() == "desc"
    expected = sp.pretty(sp.Symbol("x") ** 2, use_unicode=True).splitlines()
    for idx, part in enumerate(expected, start=2):
        assert lines[idx].strip() == part.strip()


def test_axes_rescale_on_trace_change_and_toggle():
    spec1 = ESRSpectrum(field=np.arange(2.0), intensity=np.array([0.0, 1.0]))
    spec2 = ESRSpectrum(field=np.arange(2.0), intensity=np.array([0.0, 5.0]))
    selector = gui.SpanPeakSelector([spec1, spec2])
    fig, selector.ax = plt.subplots()
    line1, = selector.ax.plot(spec1.field, spec1.intensity)
    line2, = selector.ax.plot(spec2.field, spec2.intensity)
    selector.trace_lines = [line1, line2]

    trace_var = MagicMock()
    trace_var.get.return_value = selector.labels[1]
    selector.trace_var = trace_var

    selector.ax.set_ylim(-1, 1)
    selector._on_trace_change()
    assert selector.ax.get_ylim()[1] > 4.0

    selector._toggle_trace(1, False)
    assert selector.ax.get_ylim()[1] <= 1.5

    plt.close(fig)

