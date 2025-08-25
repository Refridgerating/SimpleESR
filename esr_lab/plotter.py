"""Plotting utilities for ESR spectra."""

import matplotlib.pyplot as plt
from typing import Optional, Sequence

from .spectrum import ESRSpectrum


class ESRPlotter:
    """Plotter for ESR spectra."""

    def plot(
        self,
        spectrum: ESRSpectrum,
        peaks: Optional[Sequence[float]] = None,
        widths: Optional[Sequence[float]] = None,
    ) -> None:
        """Plot the provided spectrum.

        Parameters
        ----------
        spectrum:
            The spectrum to plot.
        peaks:
            Optional iterable of peak positions to highlight.
        widths:
            Optional iterable of full width at half maximum values corresponding
            to ``peaks``. When provided, shaded regions are drawn to visualise
            the width at half maximum for each peak.
        """

        fig, ax = plt.subplots()
        ax.plot(spectrum.field, spectrum.intensity)
        ax.set_xlabel("Magnetic Field")
        ax.set_ylabel("Intensity")
        ax.set_title("ESR Spectrum")

        if peaks is not None:
            # If widths are provided, iterate over both.  Otherwise simply draw
            # a vertical line to mark the peak position.
            widths_iter = widths if widths is not None else []
            for i, peak in enumerate(peaks):
                ax.axvline(peak, color="red", linestyle="--")
                if i < len(widths_iter):
                    half_width = widths_iter[i] / 2.0
                    left = peak - half_width
                    right = peak + half_width
                    ax.axvspan(left, right, color="orange", alpha=0.3)

        plt.show()
