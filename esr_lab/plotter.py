"""Plotting utilities for ESR spectra."""

import matplotlib.pyplot as plt

from .spectrum import ESRSpectrum


class ESRPlotter:
    """Plotter for ESR spectra."""

    def plot(self, spectrum: ESRSpectrum) -> None:
        """Plot the provided spectrum.

        Parameters
        ----------
        spectrum:
            The spectrum to plot.
        """

        fig, ax = plt.subplots()
        ax.plot(spectrum.field, spectrum.intensity)
        ax.set_xlabel("Magnetic Field")
        ax.set_ylabel("Intensity")
        ax.set_title("ESR Spectrum")
        plt.show()
