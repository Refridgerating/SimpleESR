"""Plotting utilities for ESR spectra."""

import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

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


def configure_subplot(
    ax: plt.Axes,
    *,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    marker: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    font_size: Optional[float] = None,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
    major_grid: Optional[bool] = None,
    minor_grid: Optional[bool] = None,
) -> None:
    """Configure common display properties of a Matplotlib subplot.

    The function offers a thin convenience wrapper around Matplotlib's ``Axes``
    methods so that unit tests – and eventually users – can easily tweak the
    appearance of a plot without touching the Matplotlib API directly.
    Parameters are optional; only the supplied ones are applied.

    Parameters
    ----------
    ax:
        The :class:`~matplotlib.axes.Axes` instance to configure.
    color, linewidth, marker:
        Styling options applied to all lines currently present in ``ax``.
    x_label, y_label, title:
        Axis and figure titles.
    font_size:
        Font size for axis labels, tick labels and the title.
    x_ticks, y_ticks:
        Explicit tick locations for the respective axes.
    major_grid, minor_grid:
        Toggle visibility of major and minor grid lines.
    """

    # Line styling -------------------------------------------------------
    if color is not None or linewidth is not None or marker is not None:
        for line in ax.lines:
            if color is not None:
                line.set_color(color)
            if linewidth is not None:
                line.set_linewidth(linewidth)
            if marker is not None:
                line.set_marker(marker)

    # Axis labels and title ---------------------------------------------
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size)

    if font_size is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(font_size)

    # Tick marks ---------------------------------------------------------
    if x_ticks is not None:
        ax.set_xticks(list(x_ticks))
    if y_ticks is not None:
        ax.set_yticks(list(y_ticks))

    # Grid lines ---------------------------------------------------------
    if major_grid is not None:
        ax.grid(major_grid, which="major")
    if minor_grid is not None:
        if minor_grid:
            ax.minorticks_on()
        ax.grid(minor_grid, which="minor")


def plot_residuals(field: Sequence[float], residuals: Sequence[float]) -> None:
    """Plot residuals of a fit against the magnetic field.

    A simple helper used to visualise the difference between measured data and
    a fitted model.  The function draws a scatter plot of the residuals and a
    horizontal line at zero for reference.

    Parameters
    ----------
    field:
        Magnetic-field values of the data points.
    residuals:
        Residuals ``observed - fitted`` corresponding to ``field``.
    """

    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(field, residuals, "o")
    ax.set_xlabel("Magnetic Field")
    ax.set_ylabel("Residuals")
    ax.set_title("Fit Residuals")
    plt.show()
