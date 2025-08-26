import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch

from esr_lab import gui
from esr_lab.spectrum import ESRSpectrum


def test_gui_main(monkeypatch, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    class DummyTk:
        def withdraw(self):
            pass

    monkeypatch.setattr(gui.tk, "Tk", lambda: DummyTk())
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kwargs: str(csv_file))

    with patch("esr_lab.gui.SpanPeakSelector") as selector_mock:
        gui.main()
        selector_mock.assert_called_once()
        selector_mock.return_value.show.assert_called_once()


def test_span_selector_analysis():
    spectrum = ESRSpectrum(field=np.arange(10.0), intensity=np.zeros(10))
    selector = gui.SpanPeakSelector(spectrum)

    with patch("esr_lab.gui.find_peak", return_value=(1, 3)) as fp, \
        patch("esr_lab.gui.calc_fwhm", return_value=0.5) as cf, \
        patch("esr_lab.gui.messagebox.showinfo") as info:
        selector.onselect(1.0, 2.0)
        selector.onselect(5.0, 8.0)

        assert selector.ranges == [(1.0, 2.0), (5.0, 8.0)]
        assert fp.call_count == 2
        assert cf.call_count == 2
        info.assert_called_once()


def test_peak_to_peak_slider_analysis():
    spectrum = ESRSpectrum(field=np.arange(5.0), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    selector.pos_peak = 1.0
    selector.neg_peak = 3.0

    with patch("esr_lab.gui.calc_peak_to_peak", return_value=2.0) as cpp, \
        patch("esr_lab.gui.messagebox.showinfo") as info:
        selector.analyse_peak_to_peak()
        cpp.assert_called_once()
        info.assert_called_once()


def test_lorentzian_fit_overlay():
    spectrum = ESRSpectrum(field=np.linspace(-1, 1, 5), intensity=np.zeros(5))
    selector = gui.SpanPeakSelector(spectrum)
    fig, selector.ax = plt.subplots()
    selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.selected_peak = -0.5
    with patch(
        "esr_lab.gui.fit_lorentzian_derivative", return_value=(0.0, 1.0, 1.0, 0.0)
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=True) as ask:
        selector.fit_lorentzian()
        fit.assert_called_once()
        ask.assert_called_once()
        assert len(selector.ax.lines) == 2
    plt.close(fig)

    fig, selector.ax = plt.subplots()
    selector.ax.plot(spectrum.field, spectrum.intensity)
    selector.selected_peak = 0.5
    with patch(
        "esr_lab.gui.fit_lorentzian_derivative", return_value=(0.0, 1.0, 1.0, 0.0)
    ) as fit, patch("esr_lab.gui.messagebox.askyesno", return_value=False) as ask:
        selector.fit_lorentzian()
        fit.assert_called_once()
        ask.assert_called_once()
        assert len(selector.ax.lines) == 1
    plt.close(fig)

