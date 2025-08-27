import numpy as np
import pytest

from esr_lab import (
    find_peak,
    calc_fwhm,
    fit_lorentzian_derivative,
    calc_peak_to_peak,
    peak_finder,
    chi_square,
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

    params, stats = fit_lorentzian_derivative(field, intensity)

    assert np.allclose(params, (H_res, delta, A, B), atol=1e-6)
    assert stats["chi2"] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(stats["stderr"], (0.0, 0.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(stats["residuals"], 0.0, atol=1e-12)


def test_chi_square_matches_manual_calculation():
    observed = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.5, 2.5])
    residuals = observed - expected
    manual = np.sum(residuals**2)
    assert chi_square(observed, expected) == pytest.approx(manual)


def test_peak_finder_pairs():
    field = np.arange(20.0)
    intensity = np.array(
        [
            0, 1, 0, -1, 0, 2, 0, -2, 0, 0,
            0, 1.5, 0, -1.5, 0, 0.5, 0, -0.5, 0, 0,
        ]
    )

    pairs = peak_finder(field, intensity)

    assert pairs == [(5, 7), (11, 13)]
