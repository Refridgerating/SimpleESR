import numpy as np
import matplotlib
matplotlib.use("Agg")
from unittest.mock import patch

from esr_lab.plotter import ESRPlotter
from esr_lab.spectrum import ESRSpectrum


def test_plot_calls_show():
    spectrum = ESRSpectrum(field=np.array([1, 2]), intensity=np.array([3, 4]))
    plotter = ESRPlotter()

    with patch("matplotlib.pyplot.show") as show_mock:
        plotter.plot(spectrum)
        show_mock.assert_called_once()
