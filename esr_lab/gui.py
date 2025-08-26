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

from .analysis import calc_fwhm, find_peak, fit_lorentzian_derivative
from .io import ESRLoader


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
        self.analyse_btn: tk.Button | None = None
        self.fit_btn: tk.Button | None = None
        self.peak_slider: tk.Scale | None = None
        self.selected_peak: float | None = None
        self.selector: SpanSelector | None = None

    # ------------------------------------------------------------------
    def start_analysis(self) -> None:
        """Enable span selection and prepare for analysis."""

        if self.tree is not None:
            for row in self.tree.get_children():
                self.tree.delete(row)
        self.ranges.clear()
        if self.selector is not None:
            self.selector.disconnect_events()
        assert self.ax is not None
        self.selector = SpanSelector(
            self.ax, self.onselect, "horizontal", useblit=True
        )
        if self.analyse_btn is not None:
            self.analyse_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def onselect(self, xmin: float, xmax: float) -> None:
        """Handle span selections and display peak data."""

        start, end = sorted((xmin, xmax))
        self.ranges.append((start, end))
        if len(self.ranges) < 2:
            return

        if self.selector is not None:
            self.selector.disconnect_events()

        lines = []
        for start, end in self.ranges:
            pos_idx, neg_idx = find_peak(
                self.spectrum.field, self.spectrum.intensity, start, end
            )
            fwhm = calc_fwhm(
                self.spectrum.field, self.spectrum.intensity, pos_idx, neg_idx
            )
            pos_field = self.spectrum.field[pos_idx]
            pos_y = self.spectrum.intensity[pos_idx]
            neg_field = self.spectrum.field[neg_idx]
            neg_y = self.spectrum.intensity[neg_idx]

            if self.tree is not None:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"{pos_field:.3f}",
                        f"{pos_y:.3f}",
                        f"{neg_field:.3f}",
                        f"{neg_y:.3f}",
                        f"{fwhm:.3f}",
                    ),
                )

            lines.append(
                f"Absorption: pos={pos_field:.3f}, neg={neg_field:.3f}, FWHM={fwhm:.3f}"
            )

        if self.analyse_btn is not None:
            self.analyse_btn.config(state=tk.NORMAL)

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
        """Fit a Lorentzian line near the selected peak and optionally keep it."""

        assert self.ax is not None

        if self.selected_peak is None:
            messagebox.showwarning("Lorentzian Fit", "No peak selected")
            return

        field_min = float(np.min(self.spectrum.field))
        field_max = float(np.max(self.spectrum.field))
        window = (field_max - field_min) * 0.05
        start = self.selected_peak - window / 2
        end = self.selected_peak + window / 2

        mask = (self.spectrum.field >= start) & (self.spectrum.field <= end)
        if not np.any(mask):
            mask = slice(None)

        field = self.spectrum.field[mask]
        intensity = self.spectrum.intensity[mask]
        params = fit_lorentzian_derivative(field, intensity)

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

    def fit_lorentzian(self) -> None:
        """Fit the Lorentzian model to the peak chosen via the slider."""

        self._fit_lorentzian()

    # ------------------------------------------------------------------
    def show(self) -> None:
        """Start the Tkinter main loop and display the analysis GUI."""

        self.root = tk.Tk()
        self.root.title("ESR Spectrum")

        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        panel = tk.Frame(self.root)
        panel.pack(side=tk.RIGHT, fill=tk.Y)

        fig, self.ax = plt.subplots()
        self.ax.plot(self.spectrum.field, self.spectrum.intensity)
        self.ax.set_xlabel("Magnetic Field")
        self.ax.set_ylabel("Intensity")
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.analyse_btn = tk.Button(
            panel, text="Analyse FWHM", command=self.start_analysis
        )
        self.analyse_btn.pack(padx=5, pady=5)

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

        columns = ("pos_x", "pos_y", "neg_x", "neg_y", "fwhm")
        self.tree = ttk.Treeview(panel, columns=columns, show="headings", height=5)
        headings = {
            "pos_x": "Pos X",
            "pos_y": "Pos Y",
            "neg_x": "Neg X",
            "neg_y": "Neg Y",
            "fwhm": "FWHM",
        }
        for col, text in headings.items():
            self.tree.heading(col, text=text)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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

