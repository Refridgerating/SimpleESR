"""Simple GUI utilities for visualizing and analysing ESR spectra.

This module now provides an interactive plot accompanied by an analysis panel.
Users can press a button to activate two span selections that determine the
positive and negative extrema of an absorption line.  The x and y values of the
identified peaks together with the calculated full width at half maximum (FWHM)
are shown in a small table on the right hand side of the window.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

from .analysis import calc_fwhm, find_peak
from .io import ESRLoader


def has_display() -> bool:
    """Return ``True`` if a graphical display is available."""

    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY"))


class FormatSubplotDialog(tk.Toplevel):
    """Dialog with *Format* and *Subplots* tabs."""

    def __init__(self, canvas: FigureCanvasTkAgg, initial: str = "Format") -> None:
        super().__init__()
        self.canvas = canvas
        self.title("Figure options")
        nb = ttk.Notebook(self)
        self.notebook = nb
        fmt_frame = ttk.Frame(nb)
        sub_frame = ttk.Frame(nb)
        nb.add(fmt_frame, text="Format")
        nb.add(sub_frame, text="Subplots")
        nb.pack(fill=tk.BOTH, expand=True)

        ax = canvas.figure.axes[0]
        self.title_var = tk.StringVar(value=ax.get_title())
        self.xlabel_var = tk.StringVar(value=ax.get_xlabel())
        self.ylabel_var = tk.StringVar(value=ax.get_ylabel())

        ttk.Label(fmt_frame, text="Title").grid(row=0, column=0, sticky="w")
        ttk.Entry(fmt_frame, textvariable=self.title_var).grid(
            row=0, column=1, padx=5, pady=2, sticky="ew"
        )
        ttk.Label(fmt_frame, text="X Label").grid(row=1, column=0, sticky="w")
        ttk.Entry(fmt_frame, textvariable=self.xlabel_var).grid(
            row=1, column=1, padx=5, pady=2, sticky="ew"
        )
        ttk.Label(fmt_frame, text="Y Label").grid(row=2, column=0, sticky="w")
        ttk.Entry(fmt_frame, textvariable=self.ylabel_var).grid(
            row=2, column=1, padx=5, pady=2, sticky="ew"
        )
        fmt_frame.columnconfigure(1, weight=1)
        ttk.Button(fmt_frame, text="Apply", command=self.apply_format).grid(
            row=3, column=0, columnspan=2, pady=5
        )

        self._subplot_vars: dict[str, tk.DoubleVar] = {}
        params = canvas.figure.subplotpars
        names = ["left", "bottom", "right", "top", "wspace", "hspace"]
        self._subplot_init = {name: getattr(params, name) for name in names}
        for i, name in enumerate(names):
            var = tk.DoubleVar(value=self._subplot_init[name])
            self._subplot_vars[name] = var
            ttk.Label(sub_frame, text=name.title()).grid(row=i, column=0, sticky="w")
            ttk.Scale(
                sub_frame,
                from_=0.0,
                to=1.0,
                variable=var,
                command=self._apply_subplots,
            ).grid(row=i, column=1, padx=5, pady=2, sticky="ew")
        sub_frame.columnconfigure(1, weight=1)
        ttk.Button(sub_frame, text="Reset", command=self._reset_subplots).grid(
            row=len(names), column=0, columnspan=2, pady=5
        )

        nb.select(0 if initial == "Format" else 1)

    def apply_format(self) -> None:
        ax = self.canvas.figure.axes[0]
        ax.set_title(self.title_var.get())
        ax.set_xlabel(self.xlabel_var.get())
        ax.set_ylabel(self.ylabel_var.get())
        self.canvas.draw_idle()

    def _apply_subplots(self, _=None) -> None:
        params = {name: var.get() for name, var in self._subplot_vars.items()}
        self.canvas.figure.subplots_adjust(**params)
        self.canvas.draw_idle()

    def _reset_subplots(self) -> None:
        for name, val in self._subplot_init.items():
            self._subplot_vars[name].set(val)
        self._apply_subplots()


class FormattingToolbar(NavigationToolbar2Tk):
    """Navigation toolbar with a combined format/subplot dialog."""

    toolitems = (
        NavigationToolbar2Tk.toolitems[:-2]
        + (
            (
                "Format",
                "Edit axis labels and title",
                "qt4_editor_options",
                "edit_format",
            ),
        )
        + NavigationToolbar2Tk.toolitems[-2:]
    )

    def _open_dialog(self, tab: str) -> None:
        if hasattr(self, "_dialog") and self._dialog.winfo_exists():
            self._dialog.notebook.select(0 if tab == "Format" else 1)
            self._dialog.lift()
        else:
            self._dialog = FormatSubplotDialog(self.canvas, tab)

    def edit_format(self) -> None:
        self._open_dialog("Format")

    def configure_subplots(self) -> None:  # type: ignore[override]
        self._open_dialog("Subplots")


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
    def show(self) -> None:
        """Start the Tkinter main loop and display the analysis GUI."""
        if not has_display():
            raise RuntimeError(
                "No display found. Set the DISPLAY environment variable or run this "
                "program within a virtual display (e.g., using xvfb-run)."
            )

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
        toolbar = FormattingToolbar(canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.analyse_btn = tk.Button(panel, text="Analyse FWHM", command=self.start_analysis)
        self.analyse_btn.pack(padx=5, pady=5)

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
    if not has_display():
        raise RuntimeError(
            "No display found. Set the DISPLAY environment variable or run this "
            "program within a virtual display (e.g., using xvfb-run)."
        )

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

