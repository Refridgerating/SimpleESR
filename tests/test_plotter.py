import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch

from esr_lab.plotter import ESRPlotter
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
