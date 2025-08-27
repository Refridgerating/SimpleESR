"""Simple GUI utilities for visualizing and analysing ESR spectra.

This module now provides an interactive plot accompanied by an analysis panel.
Users can press a button to activate two span selections that determine the
positive and negative extrema of an absorption line.  The x and y values of the
identified peaks together with the calculated full width at half maximum (FWHM)
are shown in a small table on the right hand side of the window.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser, simpledialog

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector, Cursor
from typing import Callable
from scipy.integrate import cumulative_trapezoid
import sympy as sp

from .analysis import (
    calc_fwhm,
    find_peak,
    fit_lorentzian_derivative,
    calc_peak_to_peak,
    peak_finder as auto_peak_finder,
    baseline_correct,
    FUNCTION_DETAILS,
)
from .io import ESRLoader
from .plotter import plot_residuals


def _filter_ticks(ticks: list[float], lower: float, upper: float) -> list[float]:
    """Return only tick locations within the provided axis limits."""

    return [t for t in ticks if lower <= t <= upper]


class NavigationToolbarNoSubplots(NavigationToolbar2Tk):
    """Tk toolbar without the subplot configuration tool.

    The default Matplotlib toolbar includes a *Configure Subplots* button which
    launches a dialog with sliders for adjusting subplot parameters.  The
    application does not rely on this functionality and its presence spawns an
    unnecessary window.  A small subclass of ``NavigationToolbar2Tk`` removes the
    corresponding tool so that users are left with the standard navigation
    controls (home, pan, zoom, save) only.  The toolbar additionally knows about
    the currently selected spectrum through a ``get_active_index`` callback so
    that line editing operates on the user‑chosen graph.
    """

    # Filter out the "Subplots" entry from the base class' tool items and add
    # custom tools for editing axes and updating the legend.  This allows users
    # to tweak axis limits and refresh legend entries directly from the embedded
    # toolbar.
    toolitems = [item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots"]

    # Insert the edit tool and the legend updater just before the standard
    # "Save" action.  The ``qt4_editor_options`` icon is shipped with Matplotlib
    # and provides a sensible default even for the Tk backend.
    _names = [item[0] for item in toolitems if item]
    if "Save" in _names:
        idx = _names.index("Save")
        toolitems.insert(
            idx,
            ("Edit", "Edit axis, curve and image parameters", "qt4_editor_options", "edit_parameters"),
        )
        toolitems.insert(
            idx + 1,
            ("Legend", "Update legend to reflect line styles", "qt4_editor_options", "update_legend"),
        )

    def __init__(
        self,
        canvas,
        window,
        get_active_index: Callable[[], int] | None = None,
        update_legend: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(canvas, window, **kwargs)
        self.get_active_index = get_active_index or (lambda: 0)
        self.update_legend_callback = update_legend

    def update_legend(self) -> None:
        if self.update_legend_callback:
            self.update_legend_callback()

    def _get_selected_line(self, ax):
        idx = self.get_active_index()
        if ax.lines and 0 <= idx < len(ax.lines):
            return ax.lines[idx]
        return ax.lines[0] if ax.lines else None

    def edit_parameters(self) -> None:
        """Open a small Tk dialog to edit basic plot parameters.

        The dialog exposes a subset of Matplotlib's axis and line configuration
        options.  Users can modify axis limits, labels, titles, tick locations,
        scales and line width.  Invalid inputs are ignored in order to keep the
        interface straightforward and robust.
        """

        figure = self.canvas.figure
        axes = figure.axes
        if not axes:
            messagebox.showwarning("Edit Plot", "No axes to configure")
            return

        ax = axes[0]

        dialog = tk.Toplevel(self)
        dialog.title("Edit Plot")

        # Helper to create a labelled entry
        def add_entry(row: int, label: str, initial: str) -> tk.Entry:
            tk.Label(dialog, text=label).grid(row=row, column=0, sticky="e")
            var = tk.Entry(dialog)
            var.insert(0, initial)
            var.grid(row=row, column=1, padx=5, pady=2)
            return var

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        title_ent = add_entry(0, "Title", ax.get_title())
        xlabel_ent = add_entry(1, "X label", ax.get_xlabel())
        ylabel_ent = add_entry(2, "Y label", ax.get_ylabel())
        xmin_ent = add_entry(3, "X min", f"{xmin}")
        xmax_ent = add_entry(4, "X max", f"{xmax}")
        ymin_ent = add_entry(5, "Y min", f"{ymin}")
        ymax_ent = add_entry(6, "Y max", f"{ymax}")
        line = self._get_selected_line(ax)
        lw_init = line.get_linewidth() if line is not None else 1.0
        lw_ent = add_entry(7, "Line width", f"{lw_init}")

        color_init = line.get_color() if line is not None else ""
        tk.Label(dialog, text="Line color").grid(row=8, column=0, sticky="e")
        color_frame = tk.Frame(dialog)
        color_frame.grid(row=8, column=1, padx=5, pady=2, sticky="w")
        color_ent = tk.Entry(color_frame)
        color_ent.insert(0, color_init)
        color_ent.grid(row=0, column=0)

        preview = tk.Canvas(color_frame, width=20, height=20, bg=color_init)
        preview.grid(row=0, column=1, padx=5)

        def choose_color() -> None:
            color = colorchooser.askcolor(color_ent.get())[1]
            if color:
                color_ent.delete(0, tk.END)
                color_ent.insert(0, color)
                preview.config(bg=color)

        try:
            ttk.Button(
                color_frame,
                text="Pick",
                command=choose_color,
                style="Compact.TButton",
            ).grid(row=0, column=2, padx=5)
        except Exception:
            tk.Button(color_frame, text="Pick", command=choose_color).grid(
                row=0, column=2, padx=5
            )

        def _update_preview(*_args: object) -> None:
            color_val = color_ent.get().strip()
            try:
                preview.config(bg=color_val)
            except tk.TclError:
                pass

        color_ent.bind("<KeyRelease>", _update_preview)

        tk.Label(dialog, text="Marker").grid(row=9, column=0, sticky="e")
        marker_init = line.get_marker() if line is not None else "None"
        marker_var = tk.StringVar(value=marker_init)
        markers = ["None", "o", "s", "^", "D", "*", "x", "+"]
        tk.OptionMenu(dialog, marker_var, *markers).grid(row=9, column=1, sticky="w")

        # Scale selection
        tk.Label(dialog, text="X scale").grid(row=10, column=0, sticky="e")
        xscale_var = tk.StringVar(value=ax.get_xscale())
        tk.OptionMenu(dialog, xscale_var, "linear", "log").grid(row=10, column=1, sticky="w")

        tk.Label(dialog, text="Y scale").grid(row=11, column=0, sticky="e")
        yscale_var = tk.StringVar(value=ax.get_yscale())
        tk.OptionMenu(dialog, yscale_var, "linear", "log").grid(row=11, column=1, sticky="w")

        xticks_ent = add_entry(12, "X ticks", ", ".join(map(str, ax.get_xticks())))
        yticks_ent = add_entry(13, "Y ticks", ", ".join(map(str, ax.get_yticks())))

        major_var = tk.BooleanVar(value=ax.xaxis._major_tick_kw.get("gridOn", False))
        tk.Checkbutton(dialog, text="Major grid", variable=major_var).grid(row=14, column=0, columnspan=2, sticky="w")
        minor_var = tk.BooleanVar(value=ax.xaxis._minor_tick_kw.get("gridOn", False))
        tk.Checkbutton(dialog, text="Minor grid", variable=minor_var).grid(row=15, column=0, columnspan=2, sticky="w")

        def apply() -> None:
            try:
                xmin_val = float(xmin_ent.get())
                xmax_val = float(xmax_ent.get())
                ax.set_xlim(xmin_val, xmax_val)
                xmin_val, xmax_val = ax.get_xlim()
            except ValueError:
                xmin_val, xmax_val = ax.get_xlim()
            try:
                ymin_val = float(ymin_ent.get())
                ymax_val = float(ymax_ent.get())
                ax.set_ylim(ymin_val, ymax_val)
                ymin_val, ymax_val = ax.get_ylim()
            except ValueError:
                ymin_val, ymax_val = ax.get_ylim()

            ax.set_title(title_ent.get())
            ax.set_xlabel(xlabel_ent.get())
            ax.set_ylabel(ylabel_ent.get())

            line = self._get_selected_line(ax)
            if line is not None:
                try:
                    lw = float(lw_ent.get())
                    line.set_linewidth(lw)
                except ValueError:
                    pass

                color_val = color_ent.get().strip()
                if color_val:
                    line.set_color(color_val)

                marker_val = marker_var.get()
                line.set_marker("" if marker_val == "None" else marker_val)

            ax.set_xscale(xscale_var.get())
            ax.set_yscale(yscale_var.get())

            try:
                ticks = [float(v) for v in xticks_ent.get().split(",") if v.strip()]
                ticks = _filter_ticks(ticks, xmin_val, xmax_val)
                ax.set_xticks(ticks)
            except ValueError:
                pass
            try:
                ticks = [float(v) for v in yticks_ent.get().split(",") if v.strip()]
                ticks = _filter_ticks(ticks, ymin_val, ymax_val)
                ax.set_yticks(ticks)
            except ValueError:
                pass

            ax.grid(major_var.get(), which="major")
            if minor_var.get():
                ax.minorticks_on()
            else:
                ax.minorticks_off()
            ax.grid(minor_var.get(), which="minor")

            self.canvas.draw_idle()

        try:
            ttk.Button(dialog, text="Apply", command=apply, style="Compact.TButton").grid(
                row=16, column=0, pady=5
            )
            ttk.Button(
                dialog, text="Close", command=dialog.destroy, style="Compact.TButton"
            ).grid(row=16, column=1, pady=5)
        except Exception:
            tk.Button(dialog, text="Apply", command=apply).grid(row=16, column=0, pady=5)
            tk.Button(dialog, text="Close", command=dialog.destroy).grid(
                row=16, column=1, pady=5
            )



class SpanPeakSelector:
    """Interactive peak analysis with an optional Tk GUI.

    The class keeps a small backwards-compatible API for the unit tests.  When
    :meth:`show` is invoked a full Tkinter application with an analysis panel is
    created.  Peak positions and the resulting FWHM are listed in a table on the
    side.  Without calling :meth:`show` the class can still be used in a headless
    manner where selections are fed directly to :meth:`onselect` and the results
    are reported via a message box, mirroring the behaviour of previous
    versions.
    """

    def __init__(self, spectrum, labels: list[str] | None = None) -> None:
        """Initialise the selector with one or more spectra.

        Parameters
        ----------
        spectrum:
            Either a single :class:`~esr_lab.spectrum.ESRSpectrum` instance or a
            list of spectra to overlay.  The previous API accepted only a single
            spectrum which is still supported for backwards compatibility.
        """

        # Normalise ``spectrum`` to a list so the rest of the implementation can
        # treat all inputs uniformly.  Existing tests pass a single spectrum
        # which results in a one-element list here.
        if isinstance(spectrum, list):
            self.spectra = spectrum
        else:
            self.spectra = [spectrum]

        if labels is not None and len(labels) == len(self.spectra):
            self.labels = labels
        else:
            self.labels = [f"Trace {i + 1}" for i in range(len(self.spectra))]
        self.current = 0
        self.spectrum = self.spectra[self.current]

        # Keep analysis results for each loaded spectrum separately.  ``results``
        # and ``lorentz_results`` always refer to the currently selected trace so
        # the public API remains unchanged.
        self.results_all: list[list[dict[str, float | str | int]]] = [
            [] for _ in self.spectra
        ]
        self.lorentz_all: list[list[dict[str, float | str | int]]] = [
            [] for _ in self.spectra
        ]
        self.results = self.results_all[self.current]
        self.lorentz_results = self.lorentz_all[self.current]

        # Individual span selections per trace
        self.ranges_all: list[list[tuple[float, float]]] = [
            [] for _ in self.spectra
        ]
        self.ranges = self.ranges_all[self.current]

        # Automatically detected peak indices per trace
        self.auto_peaks_all: list[list[tuple[int, int]]] = [
            [] for _ in self.spectra
        ]
        self.auto_peaks = self.auto_peaks_all[self.current]

        # Line objects of integrated absorption spectra per trace
        self.abs_lines: list[Line2D | None] = [None for _ in self.spectra]

        # Plot lines and visibility state for each spectrum
        self.trace_lines: list[Line2D] = []
        self.trace_vars: list[tk.BooleanVar] = []

        # GUI related attributes are initialised lazily in ``show`` so that the
        # class can be instantiated in environments without a display (e.g. the
        # test suite).
        self.root: tk.Tk | None = None
        self.ax = None
        self.peak_tree: ttk.Treeview | None = None
        self.tree: ttk.Treeview | None = None
        self.lorentz_tree: ttk.Treeview | None = None
        self.analyse_btn: tk.Button | ttk.Button | None = None
        self.dhpp_btn: tk.Button | ttk.Button | None = None
        self.find_btn: tk.Button | ttk.Button | None = None
        self.fit_btn: tk.Button | ttk.Button | None = None
        self.integrate_btn: tk.Button | ttk.Button | None = None
        self.baseline_btn: tk.Button | ttk.Button | None = None
        self.compare_btn: tk.Button | ttk.Button | None = None
        self.compare_tree: ttk.Treeview | None = None
        self.trace_combo: ttk.Combobox | None = None
        self.plot_frame: tk.Frame | None = None
        self.extra_canvases: list[FigureCanvasTkAgg] = []
        self.trace_var: tk.StringVar | None = None
        self.peak_slider: tk.Scale | None = None
        self.selected_peak: float | None = None
        self.meta_label: tk.Label | None = None
        self.metadata_text: str = ""
        # Keep track of which peak (1 or 2) the user is analysing.
        # Default to the first peak so headless usage remains functional
        # without invoking the interactive prompt.
        self.current_peak: int = 1
        self.selector: SpanSelector | None = None
        self.analysis_func: Callable[[np.ndarray, np.ndarray, int, int], float] = calc_fwhm
        self.analysis_label: str = "FWHM"

    # ------------------------------------------------------------------
    def _prompt_peak(self) -> int | None:
        """Ask the user which peak should be analysed.

        Returns
        -------
        int | None
            ``1`` or ``2`` depending on the user's choice.  ``None`` is
            returned if the dialog is cancelled.  In headless environments
            where the dialog cannot be shown, the function falls back to
            the first peak to keep scripted use working.
        """

        try:
            return simpledialog.askinteger(
                "Select Peak", "Analyse peak 1 or 2?", minvalue=1, maxvalue=2
            )
        except Exception:
            # When running without a display (e.g. during tests) Tk may raise
            # errors.  Defaulting to the first peak keeps the API usable.
            return 1

    # ------------------------------------------------------------------
    def _prompt_traces(self) -> tuple[int, int] | None:
        """Ask the user which two traces should be compared.

        Returns
        -------
        tuple[int, int] | None
            Zero-based indices of the selected traces or ``None`` if the dialog
            is cancelled. In headless environments the first two traces are
            chosen by default when available.
        """

        if len(self.spectra) < 2:
            return None

        try:
            class _TraceDialog(simpledialog.Dialog):
                """Dialog with two drop-downs to choose spectra."""

                def __init__(self, parent, labels: list[str]):
                    self.labels = labels
                    self.first_var = tk.StringVar()
                    self.second_var = tk.StringVar()
                    super().__init__(parent, title="Compare Spectra")

                def body(self, master):  # type: ignore[override]
                    ttk.Label(master, text="First trace:").grid(row=0, column=0, padx=5, pady=5)
                    first = ttk.Combobox(
                        master,
                        values=self.labels,
                        textvariable=self.first_var,
                        state="readonly",
                    )
                    first.grid(row=0, column=1, padx=5, pady=5)
                    if self.labels:
                        first.current(0)
                        self.first_var.set(self.labels[0])

                    ttk.Label(master, text="Second trace:").grid(row=1, column=0, padx=5, pady=5)
                    second = ttk.Combobox(
                        master,
                        values=self.labels,
                        textvariable=self.second_var,
                        state="readonly",
                    )
                    second.grid(row=1, column=1, padx=5, pady=5)
                    if len(self.labels) > 1:
                        second.current(1)
                        self.second_var.set(self.labels[1])
                    elif self.labels:
                        second.current(0)
                        self.second_var.set(self.labels[0])
                    return first

                def apply(self) -> None:  # type: ignore[override]
                    self.result = (
                        self.labels.index(self.first_var.get()),
                        self.labels.index(self.second_var.get()),
                    )

            parent = self.root
            temp_root = None
            if parent is None:
                temp_root = tk.Tk()
                temp_root.withdraw()
                parent = temp_root

            dialog = _TraceDialog(parent, self.labels)
            if temp_root is not None:
                temp_root.destroy()
            return dialog.result
        except Exception:
            if len(self.spectra) >= 2:
                return 0, 1
            return None

    # ------------------------------------------------------------------
    def start_analysis(
        self,
        analysis_func: Callable[[np.ndarray, np.ndarray, int, int], float] = calc_fwhm,
        label: str = "FWHM",
    ) -> None:
        """Enable span selection and prepare for analysis.

        Previously this method cleared any existing analysis results each time a
        new analysis was started.  This behaviour made it impossible to perform
        multiple analyses in succession without losing earlier data.  The method
        now preserves ``self.results`` and any existing table entries so that
        users can accumulate measurements across different analyses.
        """

        if self.tree is not None and "width" in self.tree["columns"]:
            # The tree keeps previously analysed data; only the analysis label
            # column distinguishes between different result types so the width
            # heading can remain unchanged.
            pass

        peak_choice = self._prompt_peak()
        if peak_choice is None:
            return
        self.current_peak = int(peak_choice)
        self.analysis_func = analysis_func
        self.analysis_label = label

        if len(self.auto_peaks) >= self.current_peak:
            pos_idx, neg_idx = self.auto_peaks[self.current_peak - 1]
            width = self.analysis_func(
                self.spectrum.field, self.spectrum.intensity, pos_idx, neg_idx
            )
            pos_field = self.spectrum.field[pos_idx]
            pos_y = self.spectrum.intensity[pos_idx]
            neg_field = self.spectrum.field[neg_idx]
            neg_y = self.spectrum.intensity[neg_idx]
            result = {
                "analysis": self.analysis_label,
                "peak": int(self.current_peak),
                "pos_x": float(pos_field),
                "pos_y": float(pos_y),
                "neg_x": float(neg_field),
                "neg_y": float(neg_y),
                "width": float(width),
            }
            self.results.append(result)
            if self.tree is not None:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        self.analysis_label,
                        f"{self.current_peak}",
                        f"{pos_field:.3f}",
                        f"{pos_y:.3f}",
                        f"{neg_field:.3f}",
                        f"{neg_y:.3f}",
                        f"{width:.3f}",
                    ),
                )
            messagebox.showinfo(
                "Peak analysis",
                f"Peak {self.current_peak}: pos={pos_field:.3f}, neg={neg_field:.3f}, {self.analysis_label}={width:.3f}",
            )
            return

        self.ranges.clear()
        if self.selector is not None:
            self.selector.disconnect_events()
        assert self.ax is not None
        self.selector = SpanSelector(
            self.ax, self.onselect, "horizontal", useblit=True
        )
        if self.analyse_btn is not None:
            self.analyse_btn.config(state=tk.DISABLED)
        if self.dhpp_btn is not None:
            self.dhpp_btn.config(state=tk.DISABLED)
        if self.find_btn is not None:
            self.find_btn.config(state=tk.DISABLED)

    def start_peak_to_peak(self) -> None:
        """Start interactive \u0394H_pp analysis using span selection."""

        self.start_analysis(calc_peak_to_peak, "\u0394H_pp")

    def peak_finder(self) -> None:
        """Automatically detect peak pairs and store them for analysis.

        Temporary markers are drawn on the plot to aid the user in verifying
        the detected peak positions.  The markers are removed once the user
        decides whether to accept the peaks.
        """

        # Always operate on the currently selected trace
        self.spectrum = self.spectra[self.current]
        self.auto_peaks = self.auto_peaks_all[self.current]

        try:
            num = simpledialog.askinteger(
                "Peak Finder", "How many peaks to expect?", initialvalue=4, minvalue=2
            )
        except Exception:
            num = 4
        if num is None:
            return
        try:
            pairs = auto_peak_finder(
                self.spectrum.field, self.spectrum.intensity, expected=int(num)
            )
        except ValueError as exc:
            messagebox.showerror("Peak Finder", str(exc))
            return

        markers: list[Line2D] = []
        if self.ax is not None:
            for p, n in pairs:
                (pos_marker,) = self.ax.plot(
                    self.spectrum.field[p],
                    self.spectrum.intensity[p],
                    marker="o",
                    color="red",
                )
                (neg_marker,) = self.ax.plot(
                    self.spectrum.field[n],
                    self.spectrum.intensity[n],
                    marker="o",
                    color="blue",
                )
                markers.extend([pos_marker, neg_marker])
            self.ax.figure.canvas.draw_idle()

        lines = [
            (
                f"Peak {i + 1}: pos={self.spectrum.field[p]:.3f}, "
                f"neg={self.spectrum.field[n]:.3f}"
            )
            for i, (p, n) in enumerate(pairs)
        ]
        accept = messagebox.askyesno(
            "Peak Finder", "\n".join(lines) + "\nAccept peaks?"
        )

        for m in markers:
            m.remove()
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()

        if not accept:
            return

        self.auto_peaks.clear()
        self.auto_peaks.extend(pairs)
        self._refresh_tables()
        messagebox.showinfo("Peak Finder", "Peaks stored for analysis")

    # ------------------------------------------------------------------
    def onselect(self, xmin: float, xmax: float) -> None:
        """Handle span selections and display peak data."""

        start, end = sorted((xmin, xmax))
        self.ranges.append((start, end))

        if self.selector is not None:
            self.selector.disconnect_events()

        pos_idx, neg_idx = find_peak(
            self.spectrum.field, self.spectrum.intensity, start, end
        )
        width = self.analysis_func(
            self.spectrum.field, self.spectrum.intensity, pos_idx, neg_idx
        )
        pos_field = self.spectrum.field[pos_idx]
        pos_y = self.spectrum.intensity[pos_idx]
        neg_field = self.spectrum.field[neg_idx]
        neg_y = self.spectrum.intensity[neg_idx]

        result = {
            "analysis": self.analysis_label,
            "peak": int(self.current_peak),
            "pos_x": float(pos_field),
            "pos_y": float(pos_y),
            "neg_x": float(neg_field),
            "neg_y": float(neg_y),
            "width": float(width),
        }
        self.results.append(result)

        if self.tree is not None:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    self.analysis_label,
                    f"{self.current_peak}",
                    f"{pos_field:.3f}",
                    f"{pos_y:.3f}",
                    f"{neg_field:.3f}",
                    f"{neg_y:.3f}",
                    f"{width:.3f}",
                ),
            )

        lines = [
            (
                f"Peak {r['peak']}: pos={r['pos_x']:.3f}, neg={r['neg_x']:.3f}, "
                f"{r['analysis']}={r['width']:.3f}"
            )
            for r in self.results
        ]

        if self.analyse_btn is not None:
            self.analyse_btn.config(state=tk.NORMAL)
        if self.dhpp_btn is not None:
            self.dhpp_btn.config(state=tk.NORMAL)
        if self.find_btn is not None:
            self.find_btn.config(state=tk.NORMAL)

        # Maintain backwards-compatible notification for the tests
        messagebox.showinfo("Peak analysis", "\n".join(lines))

    # ------------------------------------------------------------------
    def _update_selected_peak(self, value: str) -> None:
        """Update the stored peak position from the slider."""

        try:
            self.selected_peak = float(value)
        except ValueError:
            self.selected_peak = None

    # ------------------------------------------------------------------
    def _fit_lorentzian(self) -> None:
        """Fit a Lorentzian derivative using the full data set.

        The slider only provides an initial guess for the resonance field.  The
        actual fit is performed on the complete spectrum so that the symmetric
        and dispersive components are determined from all available data.
        """

        assert self.ax is not None

        if self.selected_peak is None:
            messagebox.showwarning("Lorentzian Fit", "No peak selected")
            return

        field = self.spectrum.field
        intensity = self.spectrum.intensity

        field_min = float(np.min(field))
        field_max = float(np.max(field))
        window = (field_max - field_min) * 0.05
        start = self.selected_peak - window / 2
        end = self.selected_peak + window / 2

        try:
            pos_idx, neg_idx = find_peak(field, intensity, start, end)
            delta_guess = abs(field[pos_idx] - field[neg_idx]) / 2.0
            a_guess = (intensity[pos_idx] - intensity[neg_idx]) / 2.0
        except ValueError:
            # Fallback guesses if the window does not contain valid peaks
            delta_guess = (field_max - field_min) / 20.0
            a_guess = (float(np.max(intensity)) - float(np.min(intensity))) / 2.0
            if a_guess == 0.0:
                a_guess = 1.0

        p0 = (self.selected_peak, delta_guess, a_guess, 0.0)
        params, stats = fit_lorentzian_derivative(field, intensity, p0=p0)

        h_res, delta, A, B = params
        chi2 = stats["chi2"]
        stderr = stats["stderr"]
        residuals = stats["residuals"]

        def _model(H: np.ndarray, H_res: float, delta: float, A: float, B: float):
            x = H - H_res
            denom = (x**2 + delta**2) ** 2
            sym = -2.0 * delta**2 * x / denom
            disp = delta * (delta**2 - x**2) / denom
            return A * sym + B * disp

        fit = _model(field, h_res, delta, A, B)
        (line,) = self.ax.plot(field, fit, label=f"Lorentzian fit at {self.selected_peak:.3f}")
        self.ax.legend()
        self.ax.figure.canvas.draw_idle()

        # Show residual plot
        res_plot = plot_residuals(field, residuals, h_res, show=self.plot_frame is None)
        if self.plot_frame is not None and isinstance(res_plot, tuple):
            fig_r, _ax_r = res_plot
            canvas_r = FigureCanvasTkAgg(fig_r, master=self.plot_frame)
            canvas_r.draw()
            canvas_r.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.extra_canvases.append(canvas_r)
            if self.root is not None:
                self.root.update_idletasks()

        accept = messagebox.askyesno(
            "Lorentzian Fit",
            (
                f"H_res={h_res:.3f}\n"
                f"Delta={delta:.3f}\nA={A:.3f}\nB={B:.3f}\n"
                f"chi^2={chi2:.3e}\n"
                f"stderr={stderr}\nAccept fit?"
            ),
        )

        if not accept:
            line.remove()
            self.ax.figure.canvas.draw_idle()
            return

        result = {
            "analysis": "Lorentzian",
            "peak": int(self.current_peak),
            "h_res": float(h_res),
            "delta": float(delta),
            "A": float(A),
            "B": float(B),
        }
        self.lorentz_results.append(result)
        if self.lorentz_tree is not None:
            self.lorentz_tree.insert(
                "",
                tk.END,
                values=(
                    "Lorentzian",
                    f"{self.current_peak}",
                    f"{h_res:.3f}",
                    f"{delta:.3f}",
                    f"{A:.3f}",
                    f"{B:.3f}",
                ),
            )

    def fit_lorentzian(self) -> None:
        """Fit the Lorentzian model to the peak chosen via the slider."""

        peak_choice = self._prompt_peak()
        if peak_choice is None:
            return
        self.current_peak = int(peak_choice)
        self._fit_lorentzian()

    # ------------------------------------------------------------------
    def compare_spectra(self) -> None:
        """Compare analysis results between two traces and tabulate differences."""

        indices = self._prompt_traces()
        if indices is None:
            messagebox.showinfo("Compare Spectra", "Two traces are required for comparison")
            return

        first_idx, second_idx = indices
        res1 = self.results_all[first_idx]
        res2 = self.results_all[second_idx]
        lor1 = self.lorentz_all[first_idx]
        lor2 = self.lorentz_all[second_idx]

        def _get(res_list: list[dict[str, float | int | str]], analysis: str, peak: int, key: str) -> float | None:
            for r in res_list:
                if r.get("analysis") == analysis and int(r.get("peak", 0)) == peak:
                    return float(r.get(key))
            return None

        rows: list[tuple[str, float, float, float]] = []
        for peak in (1, 2):
            f1 = _get(res1, "FWHM", peak, "width")
            f2 = _get(res2, "FWHM", peak, "width")
            if f1 is not None and f2 is not None:
                rows.append((f"FWHM P{peak}", f1, f2, f1 - f2))

            d1 = _get(res1, "\u0394H_pp", peak, "width")
            d2 = _get(res2, "\u0394H_pp", peak, "width")
            if d1 is not None and d2 is not None:
                rows.append((f"\u0394H_pp P{peak}", d1, d2, d1 - d2))

            h1 = _get(lor1, "Lorentzian", peak, "h_res")
            h2 = _get(lor2, "Lorentzian", peak, "h_res")
            if h1 is not None and h2 is not None:
                rows.append((f"H_res P{peak}", h1, h2, h1 - h2))

        if not rows:
            messagebox.showinfo("Compare Spectra", "No comparable results found")
            return

        if self.compare_tree is None:
            return

        for item in self.compare_tree.get_children():
            self.compare_tree.delete(item)

        for name, v1, v2, diff in rows:
            self.compare_tree.insert(
                "",
                tk.END,
                values=(name, f"{v1:.3f}", f"{v2:.3f}", f"{diff:.3f}"),
            )

    # ------------------------------------------------------------------
    def baseline_correction(self) -> None:
        """Apply a baseline correction to the currently selected trace."""

        if self.ax is None:
            return

        use_auto = messagebox.askyesno(
            "Baseline Correction",
            "Use automatic baseline fit?\nSelect 'No' for manual placement.",
        )

        field = self.spectrum.field
        intensity = self.spectrum.intensity

        if use_auto:
            corrected, _baseline = baseline_correct(field, intensity)
        else:
            fig = self.ax.figure
            plt.figure(fig.number)
            cursor = Cursor(self.ax, useblit=True, color="red", linewidth=1)
            try:
                pts = plt.ginput(n=-1, timeout=-1)
            except Exception:
                pts = []
            finally:
                try:
                    cursor.disconnect_events()
                except Exception:
                    pass
            if len(pts) < 2:
                messagebox.showwarning(
                    "Baseline Correction",
                    "At least two points are required for manual baseline correction.",
                )
                return
            corrected, _baseline = baseline_correct(field, intensity, points=pts)

        self.spectrum.intensity = corrected
        line = self.trace_lines[self.current]
        line.set_ydata(corrected)
        self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    def integrate_trace(self) -> None:
        """Integrate the selected derivative trace and plot the absorption spectrum."""

        # Operate on the currently selected spectrum
        self.spectrum = self.spectra[self.current]
        absorption = cumulative_trapezoid(
            self.spectrum.intensity, self.spectrum.field, initial=0
        )
        absorption -= float(np.mean(absorption))

        if self.ax is None:
            return

        # Remove an existing absorption line for this trace
        existing = self.abs_lines[self.current]
        if existing is not None:
            existing.remove()

        (line,) = self.ax.plot(
            self.spectrum.field,
            absorption,
            label=f"{self.labels[self.current]} (absorption)",
        )
        self.abs_lines[self.current] = line

        self.ax.legend()
        self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    def save_results(self, path: Path) -> None:
        """Save analysed peak data to a CSV file.

        Parameters
        ----------
        path:
            Destination file path. Existing files will be overwritten.
        """

        import pandas as pd

        pd.DataFrame(self.results).to_csv(Path(path), index=False)

    def _refresh_tables(self) -> None:
        """Refresh the analysis tables for the currently active trace."""

        if self.peak_tree is not None:
            for item in self.peak_tree.get_children():
                self.peak_tree.delete(item)
            label = self.labels[self.current]
            for i, (p, n) in enumerate(self.auto_peaks):
                self.peak_tree.insert(
                    "",
                    tk.END,
                    values=(
                        label,
                        f"{i + 1}",
                        f"{self.spectrum.field[p]:.3f}",
                        f"{self.spectrum.field[n]:.3f}",
                    ),
                )

        if self.tree is not None:
            for item in self.tree.get_children():
                self.tree.delete(item)
            for r in self.results:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        r["analysis"],
                        f"{r['peak']}",
                        f"{r['pos_x']:.3f}",
                        f"{r['pos_y']:.3f}",
                        f"{r['neg_x']:.3f}",
                        f"{r['neg_y']:.3f}",
                        f"{r['width']:.3f}",
                    ),
                )

        if self.lorentz_tree is not None:
            for item in self.lorentz_tree.get_children():
                self.lorentz_tree.delete(item)
            for r in self.lorentz_results:
                self.lorentz_tree.insert(
                    "",
                    tk.END,
                    values=(
                        r["analysis"],
                        f"{r['peak']}",
                        f"{r['h_res']:.3f}",
                        f"{r['delta']:.3f}",
                        f"{r['A']:.3f}",
                        f"{r['B']:.3f}",
                    ),
                )

    def _format_metadata(self, meta: dict[str, object] | None) -> str:
        """Return a human readable string for the acquisition metadata."""

        if not meta:
            return ""

        lines: list[str] = []
        if (freq := meta.get("Frequency")) is not None:
            lines.append(f"Frequency: {freq}")
        if (mod := meta.get("Modulation")) is not None:
            lines.append(f"Modulation: {mod}")
        if (mod_f := meta.get("ModulationFreq")) is not None:
            lines.append(f"Mod. Freq.: {mod_f}")
        if (b_from := meta.get("Bfrom")) is not None and (b_to := meta.get("Bto")) is not None:
            lines.append(f"B Sweep: {b_from}-{b_to}")
        if (mw := meta.get("MicrowavePower")) is not None:
            lines.append(f"MW Power: {mw}")
        if (st := meta.get("SweepTime")) is not None:
            lines.append(f"Sweep Time: {st}")
        if (temp := meta.get("Temperature")) is not None:
            lines.append(f"Temperature: {temp}")
        return "\n".join(lines)

    def _update_metadata_display(self) -> None:
        """Update the metadata label for the currently selected spectrum."""

        text = self._format_metadata(self.spectrum.metadata)
        self.metadata_text = text
        if self.meta_label is not None:
            self.meta_label.config(text=text)

    def _on_trace_change(self, _event: object | None = None) -> None:
        """Update state when the user selects a different trace."""

        if self.trace_var is None:
            return

        label = self.trace_var.get()
        if label not in self.labels:
            return

        self.current = self.labels.index(label)
        self.spectrum = self.spectra[self.current]
        self.results = self.results_all[self.current]
        self.lorentz_results = self.lorentz_all[self.current]
        self.ranges = self.ranges_all[self.current]
        self.auto_peaks = self.auto_peaks_all[self.current]

        field_min = float(np.min(self.spectrum.field))
        field_max = float(np.max(self.spectrum.field))
        if self.peak_slider is not None:
            self.peak_slider.config(from_=field_min, to=field_max)
            mid = (field_min + field_max) / 2
            self.peak_slider.set(mid)
            self.selected_peak = mid

        self._refresh_tables()
        self._update_metadata_display()

    # ------------------------------------------------------------------
    def _toggle_trace(self, index: int, show: bool) -> None:
        """Show or hide the trace at the given index."""

        if not (0 <= index < len(self.trace_lines)):
            return

        self.trace_lines[index].set_visible(show)
        if index < len(self.abs_lines) and self.abs_lines[index] is not None:
            self.abs_lines[index].set_visible(show)

        self.update_legend()
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    def update_legend(self) -> None:
        """Redraw the legend to reflect current line styles."""

        if self.ax is None or len(self.trace_lines) <= 1:
            return

        handles: list[Line2D] = []
        labels: list[str] = []
        for line, label in zip(self.trace_lines, self.labels):
            line.set_label(label)
            if line.get_visible():
                handles.append(line)
                labels.append(label)

        if handles:
            self.ax.legend(handles, labels)
        else:
            leg = self.ax.get_legend()
            if leg is not None:
                leg.remove()

        self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _show_readme(self) -> None:
        """Display the project README in a message box."""
        try:
            readme_path = Path(__file__).resolve().parent.parent / "README.md"
            text = readme_path.read_text(encoding="utf-8")
        except Exception as exc:
            text = f"Unable to load README: {exc}"
        messagebox.showinfo("README", text)

    # ------------------------------------------------------------------
    def _show_workflow(self) -> None:
        """Show a brief description of the typical workflow."""
        workflow = (
            "1. Load one or more CSV files containing ESR spectra.\n"
            "2. Use the controls to select peaks and perform analyses.\n"
            "3. Review the results in the tables on the right.\n"
            "4. Optionally fit Lorentzian lines or compare spectra."
        )
        messagebox.showinfo("Workflow", workflow)

    # ------------------------------------------------------------------
    def _show_functions(self) -> None:
        """List available analysis functions with short descriptions."""
        lines: list[str] = []
        for name, (desc, formula) in FUNCTION_DETAILS.items():
            lines.append(name)
            lines.append(f"    {desc}")
            if formula is not None:
                if isinstance(formula, sp.Basic):
                    pretty_lines = sp.pretty(formula, use_unicode=True).splitlines()
                else:
                    pretty_lines = str(formula).splitlines()
                for fl in pretty_lines:
                    lines.append(f"    {fl}")
            lines.append("")
        messagebox.showinfo("Functions", "\n".join(lines).rstrip())

    # ------------------------------------------------------------------
    def _create_menu(self) -> None:
        """Create the menu bar with a Help menu."""
        if self.root is None:
            return
        try:
            menubar = tk.Menu(self.root)
            help_menu = tk.Menu(menubar, tearoff=0)
            help_menu.add_command(label="Readme", command=self._show_readme)
            help_menu.add_command(label="Workflow", command=self._show_workflow)
            help_menu.add_command(label="Functions", command=self._show_functions)
            menubar.add_cascade(label="Help", menu=help_menu)
            self.root.config(menu=menubar)
        except Exception:
            # In headless environments or tests the Tk primitives may not be
            # fully initialised.  Failing silently keeps the rest of the GUI
            # functional while still allowing the menu to appear when running
            # interactively.
            pass

    # ------------------------------------------------------------------
    def show(self) -> None:
        """Start the Tkinter main loop and display the analysis GUI."""

        self.root = tk.Tk()
        self.root.title("ESR Spectrum")
        self._create_menu()

        ButtonCls: type[tk.Button] | type[ttk.Button] = ttk.Button
        button_kwargs: dict[str, object] = {"style": "Compact.TButton"}
        try:
            style = ttk.Style(self.root)
            style.configure("Compact.TButton", font=("TkDefaultFont", 9, "bold"), padding=(5, 2))
            style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
        except Exception:
            ButtonCls = tk.Button
            button_kwargs = {}

        try:
            plot_container = tk.Frame(self.root)
            plot_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            plot_canvas = tk.Canvas(plot_container, highlightthickness=0)
            plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            plot_scroll = tk.Scrollbar(plot_container, orient=tk.VERTICAL, command=plot_canvas.yview)
            plot_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            plot_canvas.configure(yscrollcommand=plot_scroll.set)

            plot_frame = tk.Frame(plot_canvas)
            plot_id = plot_canvas.create_window((0, 0), window=plot_frame, anchor="nw")

            def _on_plot_configure(_event: tk.Event) -> None:
                plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

            plot_frame.bind("<Configure>", _on_plot_configure)

            def _on_plot_canvas_configure(event: tk.Event) -> None:
                plot_canvas.itemconfigure(plot_id, width=event.width)

            plot_canvas.bind("<Configure>", _on_plot_canvas_configure)

            def _on_plot_mousewheel(event: tk.Event) -> None:
                plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            plot_canvas.bind(
                "<Enter>", lambda _e: plot_canvas.bind_all("<MouseWheel>", _on_plot_mousewheel)
            )
            plot_canvas.bind(
                "<Leave>", lambda _e: plot_canvas.unbind_all("<MouseWheel>")
            )
        except Exception:
            plot_frame = tk.Frame(self.root)
            plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.plot_frame = plot_frame

        try:
            panel_container = tk.Frame(self.root)
            panel_container.pack(side=tk.RIGHT, fill=tk.Y)
            panel_canvas = tk.Canvas(panel_container, highlightthickness=0)
            panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = tk.Scrollbar(panel_container, orient=tk.VERTICAL, command=panel_canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            panel_canvas.configure(yscrollcommand=scrollbar.set)

            panel = tk.Frame(panel_canvas)
            panel_id = panel_canvas.create_window((0, 0), window=panel, anchor="nw")

            def _on_panel_configure(_event: tk.Event) -> None:
                panel_canvas.configure(scrollregion=panel_canvas.bbox("all"))

            panel.bind("<Configure>", _on_panel_configure)

            def _on_canvas_configure(event: tk.Event) -> None:
                panel_canvas.itemconfigure(panel_id, width=event.width)

            panel_canvas.bind("<Configure>", _on_canvas_configure)

            def _on_mousewheel(event: tk.Event) -> None:
                panel_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            panel_canvas.bind(
                "<Enter>", lambda _e: panel_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            )
            panel_canvas.bind(
                "<Leave>", lambda _e: panel_canvas.unbind_all("<MouseWheel>")
            )
        except Exception:
            panel = tk.Frame(self.root)
            panel.pack(side=tk.RIGHT, fill=tk.Y)

        # ------------------------------------------------------------------
        # Metadata panel
        meta_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        meta_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(meta_frame, text="Metadata", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=5, pady=(5, 0)
        )
        self.meta_label = tk.Label(meta_frame, justify=tk.LEFT)
        self.meta_label.pack(anchor="w", padx=5, pady=(0, 5))
        self._update_metadata_display()

        # ------------------------------------------------------------------
        # Plot area with toolbar on top
        fig, self.ax = plt.subplots()
        self.trace_lines = []
        for spec in self.spectra:
            line, = self.ax.plot(spec.field, spec.intensity)
            self.trace_lines.append(line)
        self.ax.set_xlabel("Magnetic Field")
        self.ax.set_ylabel("Intensity")
        self.update_legend()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        toolbar = NavigationToolbarNoSubplots(
            canvas,
            plot_frame,
            get_active_index=lambda: self.current,
            update_legend=self.update_legend,
            pack_toolbar=False,
        )
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ------------------------------------------------------------------
        # Controls
        control_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(control_frame, text="Controls", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=5, pady=(5, 0)
        )

        if len(self.spectra) > 1:
            tk.Label(control_frame, text="Graph to be edited/analyzed").pack(
                padx=5, pady=(0, 5)
            )
            self.trace_var = tk.StringVar(value=self.labels[0])
            self.trace_combo = ttk.Combobox(
                control_frame,
                textvariable=self.trace_var,
                values=self.labels,
                state="readonly",
            )
            self.trace_combo.bind("<<ComboboxSelected>>", self._on_trace_change)
            self.trace_combo.pack(fill=tk.X, padx=5, pady=(0, 5))

            toggle_frame = tk.Frame(control_frame)
            toggle_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            tk.Label(toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, label in enumerate(self.labels):
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(
                    toggle_frame,
                    text=label,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self.trace_vars.append(var)

        # Button rows for compact layout
        button_row1 = tk.Frame(control_frame)
        button_row1.pack(padx=5, pady=2, anchor="w")

        self.analyse_btn = ButtonCls(
            button_row1,
            text="Analyse FWHM",
            command=self.start_analysis,
            **button_kwargs,
        )
        self.analyse_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.dhpp_btn = ButtonCls(
            button_row1,
            text="Analyse \u0394H_pp",
            command=self.start_peak_to_peak,
            **button_kwargs,
        )
        self.dhpp_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.find_btn = ButtonCls(
            button_row1,
            text="Find Peaks",
            command=self.peak_finder,
            **button_kwargs,
        )
        self.find_btn.pack(side=tk.LEFT, padx=2, pady=2)

        field_min = float(np.min(self.spectrum.field))
        field_max = float(np.max(self.spectrum.field))
        self.peak_slider = tk.Scale(
            control_frame,
            from_=field_min,
            to=field_max,
            orient=tk.HORIZONTAL,
            label="Peak position",
            command=self._update_selected_peak,
        )
        mid = (field_min + field_max) / 2
        self.peak_slider.set(mid)
        self.selected_peak = mid
        self.peak_slider.pack(fill=tk.X, padx=5, pady=2)

        button_row2 = tk.Frame(control_frame)
        button_row2.pack(padx=5, pady=(2, 5), anchor="w")

        self.fit_btn = ButtonCls(
            button_row2,
            text="Fit Lorentzian",
            command=self.fit_lorentzian,
            **button_kwargs,
        )
        self.fit_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.integrate_btn = ButtonCls(
            button_row2,
            text="Integrate Trace",
            command=self.integrate_trace,
            **button_kwargs,
        )
        self.integrate_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.baseline_btn = ButtonCls(
            button_row2,
            text="Baseline Correct",
            command=self.baseline_correction,
            **button_kwargs,
        )
        self.baseline_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.compare_btn = ButtonCls(
            button_row2,
            text="Compare Spectra",
            command=self.compare_spectra,
            **button_kwargs,
        )
        self.compare_btn.pack(side=tk.LEFT, padx=2, pady=2)

        # ------------------------------------------------------------------
        # Peak position table
        peak_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        peak_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        tk.Label(
            peak_frame, text="Peak position", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", padx=5, pady=(5, 0))
        peak_columns = ("trace", "peak", "pos", "neg")
        self.peak_tree = ttk.Treeview(
            peak_frame, columns=peak_columns, show="headings", height=5
        )
        peak_headings = {
            "trace": "Trace",
            "peak": "Peak",
            "pos": "Pos X",
            "neg": "Neg X",
        }
        for col, text in peak_headings.items():
            self.peak_tree.heading(col, text=text)
            self.peak_tree.column(col, anchor=tk.CENTER)
        self.peak_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # ------------------------------------------------------------------
        # Results tables
        result_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        tk.Label(
            result_frame, text="Analysis Results", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", padx=5, pady=(5, 0))
        columns = ("analysis", "peak", "pos_x", "pos_y", "neg_x", "neg_y", "width")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=5)
        headings = {
            "analysis": "Analysis",
            "peak": "Peak",
            "pos_x": "Pos X",
            "pos_y": "Pos Y",
            "neg_x": "Neg X",
            "neg_y": "Neg Y",
            "width": "Value",
        }
        for col, text in headings.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        lorentz_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        lorentz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        tk.Label(
            lorentz_frame, text="Lorentzian Fits", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", padx=5, pady=(5, 0))
        lorentz_columns = ("analysis", "peak", "h_res", "delta", "A", "B")
        self.lorentz_tree = ttk.Treeview(
            lorentz_frame, columns=lorentz_columns, show="headings", height=5
        )
        lorentz_headings = {
            "analysis": "Analysis",
            "peak": "Peak",
            "h_res": "H_res",
            "delta": "Delta",
            "A": "A",
            "B": "B",
        }
        for col, text in lorentz_headings.items():
            self.lorentz_tree.heading(col, text=text)
            self.lorentz_tree.column(col, anchor=tk.CENTER)
        self.lorentz_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        compare_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        compare_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            compare_frame, text="Comparison", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", padx=5, pady=(5, 0))
        compare_cols = ("param", "first", "second", "diff")
        self.compare_tree = ttk.Treeview(
            compare_frame, columns=compare_cols, show="headings", height=6
        )
        compare_headings = {
            "param": "Parameter",
            "first": "Trace 1",
            "second": "Trace 2",
            "diff": "Diff",
        }
        for col, text in compare_headings.items():
            self.compare_tree.heading(col, text=text)
            self.compare_tree.column(col, anchor=tk.CENTER)
        self.compare_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Ensure the tables reflect any results already calculated before the GUI
        self._refresh_tables()
        self.root.mainloop()


def main() -> None:
    """Launch a file selection dialog and start the analyser."""

    root = tk.Tk()
    root.withdraw()  # Hide the root window for the file dialog

    file_paths = filedialog.askopenfilenames(
        title="Select ESR CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )

    if hasattr(root, "destroy"):
        root.destroy()

    if not file_paths:
        return

    try:
        spectra = []
        labels: list[str] = []
        for fp in file_paths:
            p = Path(fp)
            spectra.append(ESRLoader.load_csv(p))
            labels.append(p.name)
        app = SpanPeakSelector(spectra, labels=labels)
        app.show()
    except Exception as exc:  # pragma: no cover - GUI error handling
        messagebox.showerror("Error", str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()

