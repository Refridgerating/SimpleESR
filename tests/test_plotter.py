import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch

from esr_lab.plotter import ESRPlotter, configure_subplot, plot_residuals
from esr_lab.spectrum import ESRSpectrum


def test_plot_calls_show():
    spectrum = ESRSpectrum(field=np.array([1, 2]), intensity=np.array([3, 4]))
    plotter = ESRPlotter()

    with patch("matplotlib.pyplot.show") as show_mock:
        plotter.plot(spectrum)
        show_mock.assert_called_once()
    plt.close("all")


def test_plot_peaks_and_widths():
    spectrum = ESRSpectrum(field=np.linspace(0, 10, 50), intensity=np.zeros(50))
    plotter = ESRPlotter()
    peaks = [5.0]
    widths = [2.0]

    with patch("matplotlib.pyplot.show"):
        plotter.plot(spectrum, peaks=peaks, widths=widths)

    fig = plt.gcf()
    ax = fig.axes[0]

    # A shaded region should be created for the FWHM
    assert len(ax.patches) == 1
    fwhm_patch = ax.patches[0]
    assert fwhm_patch.get_x() == peaks[0] - widths[0] / 2
    assert fwhm_patch.get_width() == widths[0]
    plt.close(fig)


def test_configure_subplot_allows_customisation():
    spectrum = ESRSpectrum(field=np.array([0, 1, 2]), intensity=np.array([1, 2, 3]))
    plotter = ESRPlotter()

    with patch("matplotlib.pyplot.show"):
        plotter.plot(spectrum)

    fig = plt.gcf()
    ax = fig.axes[0]

    configure_subplot(
        ax,
        color="green",
        linewidth=2,
        marker="o",
        x_label="X axis",
        y_label="Y axis",
        title="Title",
        font_size=14,
        x_ticks=[0, 1, 2],
        y_ticks=[1, 2, 3],
        major_grid=True,
        minor_grid=True,
    )

    line = ax.lines[0]
    assert line.get_color() == "green"
    assert line.get_linewidth() == 2
    assert line.get_marker() == "o"
    assert ax.get_xlabel() == "X axis"
    assert ax.get_ylabel() == "Y axis"
    assert ax.get_title() == "Title"
    assert list(ax.get_xticks()) == [0, 1, 2]
    assert list(ax.get_yticks()) == [1, 2, 3]
    assert ax.xaxis.label.get_fontsize() == 14
    assert ax.yaxis.label.get_fontsize() == 14
    assert ax.title.get_fontsize() == 14
    assert ax.xaxis._major_tick_kw["gridOn"]
    assert ax.xaxis._minor_tick_kw["gridOn"]
    plt.close(fig)


def test_plot_residuals_calls_show():
    field = np.array([0.0, 1.0])
    residuals = np.array([0.1, -0.1])
    with patch("matplotlib.pyplot.show") as show_mock:
        plot_residuals(field, residuals)
        show_mock.assert_called_once()
    plt.close("all")


def test_plot_residuals_returns_figure_without_show():
    field = np.array([0.0, 1.0])
    residuals = np.array([0.1, -0.1])
    with patch("matplotlib.pyplot.show") as show_mock:
        fig, ax = plot_residuals(field, residuals, show=False)
        show_mock.assert_not_called()
        assert ax.get_ylabel() == "Residuals"
    plt.close(fig)
