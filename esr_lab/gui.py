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
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from typing import Callable

from .analysis import (
    calc_fwhm,
    find_peak,
    fit_lorentzian_derivative,
    calc_peak_to_peak,
)
from .io import ESRLoader


class NavigationToolbarNoSubplots(NavigationToolbar2Tk):
    """Tk toolbar without the subplot configuration tool.

    The default Matplotlib toolbar includes a *Configure Subplots* button which
    launches a dialog with sliders for adjusting subplot parameters.  The
    application does not rely on this functionality and its presence spawns an
    unnecessary window.  A small subclass of ``NavigationToolbar2Tk`` removes the
    corresponding tool so that users are left with the standard navigation
    controls (home, pan, zoom, save) only.
    """

    # Filter out the "Subplots" entry from the base class' tool items and add a
    # custom "Edit" button which exposes a minimal set of Matplotlib's plotting
    # options.  This allows users to tweak axis limits, labels and line styling
    # directly from the embedded toolbar.
    toolitems = [item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots"]

    # Insert the edit tool just before the standard "Save" action.  The
    # ``qt4_editor_options`` icon is shipped with Matplotlib and provides a
    # sensible default even for the Tk backend.
    _names = [item[0] for item in toolitems if item]
    if "Save" in _names:
        toolitems.insert(
            _names.index("Save"),
            ("Edit", "Edit axis, curve and image parameters", "qt4_editor_options", "edit_parameters"),
        )

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
        lw_init = ax.lines[0].get_linewidth() if ax.lines else 1.0
        lw_ent = add_entry(7, "Line width", f"{lw_init}")

        # Scale selection
        tk.Label(dialog, text="X scale").grid(row=8, column=0, sticky="e")
        xscale_var = tk.StringVar(value=ax.get_xscale())
        tk.OptionMenu(dialog, xscale_var, "linear", "log").grid(row=8, column=1, sticky="w")

        tk.Label(dialog, text="Y scale").grid(row=9, column=0, sticky="e")
        yscale_var = tk.StringVar(value=ax.get_yscale())
        tk.OptionMenu(dialog, yscale_var, "linear", "log").grid(row=9, column=1, sticky="w")

        xticks_ent = add_entry(10, "X ticks", ", ".join(map(str, ax.get_xticks())))
        yticks_ent = add_entry(11, "Y ticks", ", ".join(map(str, ax.get_yticks())))

        def apply() -> None:
            try:
                ax.set_xlim(float(xmin_ent.get()), float(xmax_ent.get()))
            except ValueError:
                pass
            try:
                ax.set_ylim(float(ymin_ent.get()), float(ymax_ent.get()))
            except ValueError:
                pass

            ax.set_title(title_ent.get())
            ax.set_xlabel(xlabel_ent.get())
            ax.set_ylabel(ylabel_ent.get())

            try:
                lw = float(lw_ent.get())
                for line in ax.lines:
                    line.set_linewidth(lw)
            except ValueError:
                pass

            ax.set_xscale(xscale_var.get())
            ax.set_yscale(yscale_var.get())

            try:
                ticks = [float(v) for v in xticks_ent.get().split(",") if v.strip()]
                ax.set_xticks(ticks)
            except ValueError:
                pass
            try:
                ticks = [float(v) for v in yticks_ent.get().split(",") if v.strip()]
                ax.set_yticks(ticks)
            except ValueError:
                pass

            self.canvas.draw_idle()
            dialog.destroy()

        tk.Button(dialog, text="Apply", command=apply).grid(row=12, column=0, columnspan=2, pady=5)



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

    def __init__(self, spectrum) -> None:
        self.spectrum = spectrum
        self.ranges: list[tuple[float, float]] = []

        # GUI related attributes are initialised lazily in ``show`` so that the
        # class can be instantiated in environments without a display (e.g. the
        # test suite).
        self.root: tk.Tk | None = None
        self.ax = None
        self.tree: ttk.Treeview | None = None
        self.lorentz_tree: ttk.Treeview | None = None
        self.analyse_btn: tk.Button | None = None
        self.dhpp_btn: tk.Button | None = None
        self.fit_btn: tk.Button | None = None
        self.peak_slider: tk.Scale | None = None
        self.selected_peak: float | None = None
        self.selector: SpanSelector | None = None
        self.analysis_func: Callable[[np.ndarray, np.ndarray, int, int], float] = calc_fwhm
        self.analysis_label: str = "FWHM"
        # Store analysed peak data for optional export
        self.results: list[dict[str, float | str]] = []
        self.lorentz_results: list[dict[str, float]] = []

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

        self.ranges.clear()
        self.analysis_func = analysis_func
        self.analysis_label = label
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

    def start_peak_to_peak(self) -> None:
        """Start interactive \u0394H_pp analysis using span selection."""

        self.start_analysis(calc_peak_to_peak, "\u0394H_pp")

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
                    f"{pos_field:.3f}",
                    f"{pos_y:.3f}",
                    f"{neg_field:.3f}",
                    f"{neg_y:.3f}",
                    f"{width:.3f}",
                ),
            )

        lines = [
            f"Absorption: pos={r['pos_x']:.3f}, neg={r['neg_x']:.3f}, {r['analysis']}={r['width']:.3f}"
            for r in self.results
        ]

        if self.analyse_btn is not None:
            self.analyse_btn.config(state=tk.NORMAL)
        if self.dhpp_btn is not None:
            self.dhpp_btn.config(state=tk.NORMAL)

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
        params = fit_lorentzian_derivative(field, intensity, p0=p0)

        h_res, delta, A, B = params

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

        accept = messagebox.askyesno(
            "Lorentzian Fit",
            (
                f"H_res={h_res:.3f}\n"
                f"Delta={delta:.3f}\nA={A:.3f}\nB={B:.3f}\nAccept fit?"
            ),
        )

        if not accept:
            line.remove()
            self.ax.figure.canvas.draw_idle()
            return

        result = {
            "peak": float(self.selected_peak),
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
                    f"{self.selected_peak:.3f}",
                    f"{h_res:.3f}",
                    f"{delta:.3f}",
                    f"{A:.3f}",
                    f"{B:.3f}",
                ),
            )

    def fit_lorentzian(self) -> None:
        """Fit the Lorentzian model to the peak chosen via the slider."""

        self._fit_lorentzian()

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

    # ------------------------------------------------------------------
    def show(self) -> None:
        """Start the Tkinter main loop and display the analysis GUI."""

        self.root = tk.Tk()
        self.root.title("ESR Spectrum")

        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        panel = tk.Frame(self.root)
        panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Show acquisition metadata at the top of the analysis panel
        if self.spectrum.metadata:
            meta = self.spectrum.metadata
            lines: list[str] = []
            if (freq := meta.get("Frequency")) is not None:
                lines.append(f"Frequency: {freq}")
            if (mod := meta.get("Modulation")) is not None:
                lines.append(f"Modulation: {mod}")
            if (mod_f := meta.get("ModulationFreq")) is not None:
                lines.append(f"Mod. Freq.: {mod_f}")
            if (b_from := meta.get("Bfrom")) is not None and (
                b_to := meta.get("Bto")
            ) is not None:
                lines.append(f"B Sweep: {b_from}-{b_to}")
            if (mw := meta.get("MicrowavePower")) is not None:
                lines.append(f"MW Power: {mw}")
            if (st := meta.get("SweepTime")) is not None:
                lines.append(f"Sweep Time: {st}")
            if (temp := meta.get("Temperature")) is not None:
                lines.append(f"Temperature: {temp}")
            if lines:
                meta_label = tk.Label(panel, text="\n".join(lines), justify=tk.LEFT)
                meta_label.pack(padx=5, pady=5)

        fig, self.ax = plt.subplots()
        self.ax.plot(self.spectrum.field, self.spectrum.intensity)
        self.ax.set_xlabel("Magnetic Field")
        self.ax.set_ylabel("Intensity")
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        toolbar = NavigationToolbarNoSubplots(canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.analyse_btn = tk.Button(
            panel, text="Analyse FWHM", command=self.start_analysis
        )
        self.analyse_btn.pack(padx=5, pady=5)

        self.dhpp_btn = tk.Button(
            panel, text="Analyse \u0394H_pp", command=self.start_peak_to_peak
        )
        self.dhpp_btn.pack(padx=5, pady=5)

        field_min = float(np.min(self.spectrum.field))
        field_max = float(np.max(self.spectrum.field))

        self.peak_slider = tk.Scale(
            panel,
            from_=field_min,
            to=field_max,
            orient=tk.HORIZONTAL,
            label="Peak position",
            command=self._update_selected_peak,
        )
        mid = (field_min + field_max) / 2
        self.peak_slider.set(mid)
        self.selected_peak = mid
        self.peak_slider.pack(fill=tk.X, padx=5, pady=5)

        self.fit_btn = tk.Button(
            panel, text="Fit Lorentzian", command=self.fit_lorentzian
        )
        self.fit_btn.pack(padx=5, pady=5)

        columns = ("analysis", "pos_x", "pos_y", "neg_x", "neg_y", "width")
        self.tree = ttk.Treeview(panel, columns=columns, show="headings", height=5)
        headings = {
            "analysis": "Analysis",
            "pos_x": "Pos X",
            "pos_y": "Pos Y",
            "neg_x": "Neg X",
            "neg_y": "Neg Y",
            # The value column holds either FWHM or ΔH_pp results depending on
            # the active analysis.  Keeping a generic heading allows appending
            # results from multiple analyses without clearing the table.
            "width": "Value",
        }
        for col, text in headings.items():
            self.tree.heading(col, text=text)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        lorentz_columns = ("peak", "h_res", "delta", "A", "B")
        self.lorentz_tree = ttk.Treeview(
            panel, columns=lorentz_columns, show="headings", height=5
        )
        lorentz_headings = {
            "peak": "Peak",
            "h_res": "H_res",
            "delta": "Delta",
            "A": "A",
            "B": "B",
        }
        for col, text in lorentz_headings.items():
            self.lorentz_tree.heading(col, text=text)
        self.lorentz_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.root.mainloop()


def main() -> None:
    """Launch a file selection dialog and start the analyser."""

    root = tk.Tk()
    root.withdraw()  # Hide the root window for the file dialog

    file_path = filedialog.askopenfilename(
        title="Select ESR CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )

    if hasattr(root, "destroy"):
        root.destroy()

    if not file_path:
        return

    try:
        spectrum = ESRLoader.load_csv(Path(file_path))
        app = SpanPeakSelector(spectrum)
        app.show()
    except Exception as exc:  # pragma: no cover - GUI error handling
        messagebox.showerror("Error", str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()

