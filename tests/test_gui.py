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


def test_lorentzian_fit_overlay():
    spectrum = ESRSpectrum(field=np.linspace(-1, 1, 5), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    selector.ax.plot(spectrum.field, spectrum.intensity)
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
    plt.close(fig)

    fig, selector.ax = plt.subplots()
    selector.ax.plot(spectrum.field, spectrum.intensity)
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
    selector.ax.plot(field, intensity)
    selector.integrate_trace()

    from scipy.integrate import cumulative_trapezoid

    expected = cumulative_trapezoid(intensity, field, initial=0)
    expected -= np.mean(expected)

    line = selector.ax.lines[-1]
    assert np.allclose(line.get_ydata(), expected)
    assert line.get_label() == "Trace 1 (absorption)"
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
    assert len(selector.ax.get_legend().get_texts()) == 2

    selector._toggle_trace(1, False)
    assert not selector.trace_lines[1].get_visible()
    assert len(selector.ax.get_legend().get_texts()) == 1

    selector._toggle_trace(1, True)
    assert selector.trace_lines[1].get_visible()
    assert len(selector.ax.get_legend().get_texts()) == 2
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
    monkeypatch.setattr(gui, "FUNCTION_DETAILS", {"f": ("desc", r"$x^2$")})
    selector._show_functions()
    call = info.call_args
    assert call[0][0] == "Functions"
    lines = call[0][1].splitlines()
    assert lines[0] == "f"
    assert lines[1].strip() == "desc"
    assert lines[2].strip() == r"$x^2$"

