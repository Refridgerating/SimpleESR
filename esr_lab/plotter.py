"""Plotting utilities for ESR spectra."""

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt

from .spectrum import ESRSpectrum


class ESRPlotter:
    """Plotter for ESR spectra."""

    def __init__(self) -> None:
        self._fig, self._ax = plt.subplots()

    def plot(self, spectrum: ESRSpectrum, *, show: bool = True, save: str | None = None) -> None:
        """Plot the provided spectrum.

        Parameters
        ----------
        spectrum:
            The spectrum to plot.
        show:
            Whether to display the plot in an interactive window.
        save:
            Optional path to save the figure to instead of displaying it.
        """

        self._ax.plot(spectrum.field, spectrum.intensity)
        self._ax.set_xlabel("Magnetic Field")
        self._ax.set_ylabel("Intensity")
        self._ax.set_title("ESR Spectrum")

        if save:
            self._fig.savefig(save)
        if show:
            plt.show()
