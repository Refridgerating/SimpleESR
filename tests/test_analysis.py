import numpy as np

from esr_lab import calc_fwhm, find_peak


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


def test_calc_fwhm_symmetric_peak():
    field = np.linspace(-5, 5, 10001)
    sigma = 0.5
    intensity = np.exp(-(field ** 2) / (2 * sigma**2))
    peak_idx = np.argmax(intensity)

    width = calc_fwhm(field, intensity, peak_idx)
    expected = 2 * np.sqrt(2 * np.log(2)) * sigma

    assert np.isclose(width, expected, atol=1e-3)


def test_calc_fwhm_asymmetric_peak():
    field = np.linspace(-5, 5, 10001)
    mu = 0.0
    sigma_left = 0.5
    sigma_right = 1.0
    intensity = np.where(
        field < mu,
        np.exp(-((field - mu) ** 2) / (2 * sigma_left**2)),
        np.exp(-((field - mu) ** 2) / (2 * sigma_right**2)),
    )
    peak_idx = np.argmax(intensity)

    width = calc_fwhm(field, intensity, peak_idx)
    expected = np.sqrt(2 * np.log(2)) * (sigma_left + sigma_right)

    assert np.isclose(width, expected, atol=1e-3)
