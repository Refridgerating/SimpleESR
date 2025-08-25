"""Simple GUI utilities for visualizing and analysing ESR spectra.

This module now provides an interactive plot where users can select two
magnetic-field ranges using a :class:`matplotlib.widgets.SpanSelector`. For
each selected window the peak position and full width at half maximum (FWHM)
are calculated and displayed.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

from .analysis import calc_fwhm, find_peak
from .io import ESRLoader


class SpanPeakSelector:
    """Interactive peak analysis using span selections.

    Parameters
    ----------
    spectrum:
        The spectrum to analyse.
    """

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.ranges: list[tuple[float, float]] = []
        self.fig, self.ax = plt.subplots()
        self.ax.plot(spectrum.field, spectrum.intensity)
        self.ax.set_xlabel("Magnetic Field")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("ESR Spectrum")
        self.selector = SpanSelector(self.ax, self.onselect, "horizontal", useblit=True)

    def onselect(self, xmin: float, xmax: float) -> None:
        """Handle span selections and perform peak analysis.

        The start and end of the selection are stored. After two ranges have
        been selected, the most prominent peak and its FWHM are calculated for
        each range and the results shown in a message box.
        """

        start, end = sorted((xmin, xmax))
        self.ranges.append((start, end))
        if len(self.ranges) < 2:
            return

        self.selector.disconnect_events()
        lines = []
        for i, (start, end) in enumerate(self.ranges, start=1):
            pos_idx, neg_idx = find_peak(
                self.spectrum.field, self.spectrum.intensity, start, end
            )
            fwhm = calc_fwhm(
                self.spectrum.field, self.spectrum.intensity, pos_idx, neg_idx
            )
            pos_field = self.spectrum.field[pos_idx]
            neg_field = self.spectrum.field[neg_idx]
            lines.append(
                f"Absorption {i}: pos={pos_field:.3f}, neg={neg_field:.3f}, FWHM={fwhm:.3f}"
            )
        messagebox.showinfo("Peak analysis", "\n".join(lines))

    def show(self) -> None:
        """Display the plot."""

        plt.show()


def main() -> None:
    """Launch a file selection dialog and start the interactive analyser."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select ESR CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )

    if not file_path:
        return

    try:
        spectrum = ESRLoader.load_csv(Path(file_path))
        selector = SpanPeakSelector(spectrum)
        selector.show()
    except Exception as exc:  # pragma: no cover - GUI error handling
        messagebox.showerror("Error", str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
