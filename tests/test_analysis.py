import numpy as np

from esr_lab import find_peak


def test_find_peak_single_peak():
    field = np.linspace(0, 10, 500)
    intensity = np.exp(-(field - 5) ** 2)

    idx = find_peak(field, intensity, 4, 6)

    assert idx == np.argmax(intensity)


def test_find_peak_multiple_peaks():
    field = np.linspace(0, 10, 500)
    intensity = (
        np.exp(-(field - 3) ** 2)  # smaller positive peak
        - 2 * np.exp(-(field - 7) ** 2)  # larger negative peak
    )

    idx = find_peak(field, intensity, 0, 10)

    assert idx == np.argmin(intensity)
