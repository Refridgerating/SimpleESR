import numpy as np
import pytest

from esr_lab import (
    find_peak,
    calc_fwhm,
    fit_lorentzian_derivative,
    calc_peak_to_peak,
)


def test_find_peak_pair():
    field = np.arange(10.0)
    intensity = np.array([0, 1, 0, -1, 0, 2, 0, -2, 0, 0])

    pos_idx, neg_idx = find_peak(field, intensity, 4, 8)

    assert pos_idx == 5
    assert neg_idx == 7


def test_calc_fwhm_from_peaks():
    field = np.linspace(-5, 5, 10001)
    intensity = field * np.exp(-field**2 / 2)

    pos_idx, neg_idx = find_peak(field, intensity, -2, 2)
    width = calc_fwhm(field, intensity, pos_idx, neg_idx)

    assert np.isclose(width, 2.0, atol=1e-3)


def test_calc_peak_to_peak_from_peaks():
    field = np.linspace(-5, 5, 10001)
    intensity = field * np.exp(-field**2 / 2)

    pos_idx, neg_idx = find_peak(field, intensity, -2, 2)
    width = calc_peak_to_peak(field, intensity, pos_idx, neg_idx)

    assert np.isclose(width, 2.0, atol=1e-3)


def test_fit_lorentzian_derivative_parameters():
    field = np.linspace(-10, 10, 1001)
    H_res, delta, A, B = 1.0, 2.0, 3.0, 1.5

    x = field - H_res
    denom = (x**2 + delta**2) ** 2
    intensity = A * (-2.0 * delta**2 * x / denom) + B * (
        delta * (delta**2 - x**2) / denom
    )

    params = fit_lorentzian_derivative(field, intensity)

    assert np.allclose(params, (H_res, delta, A, B), atol=1e-6)
