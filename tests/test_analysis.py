import numpy as np
import pytest
from scipy.constants import h, physical_constants

from esr_lab import (
    find_peak,
    calc_fwhm,
    fit_lorentzian_derivative,
    calc_peak_to_peak,
    peak_finder,
    chi_square,
    baseline_correct,
    calc_g,
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
    expected = 2.0 * np.sqrt(3.0)
    assert np.isclose(width, expected, atol=1e-3)


def test_calc_peak_to_peak_from_peaks():
    field = np.linspace(-5, 5, 10001)
    intensity = field * np.exp(-field**2 / 2)

    pos_idx, neg_idx = find_peak(field, intensity, -2, 2)
    width = calc_peak_to_peak(field, intensity, pos_idx, neg_idx)

    assert np.isclose(width, 2.0, atol=1e-3)


def test_fwhm_relates_to_peak_to_peak():
    field = np.linspace(-5, 5, 10001)
    intensity = field * np.exp(-field**2 / 2)

    pos_idx, neg_idx = find_peak(field, intensity, -2, 2)
    fwhm = calc_fwhm(field, intensity, pos_idx, neg_idx)
    dhpp = calc_peak_to_peak(field, intensity, pos_idx, neg_idx)

    assert not np.isclose(fwhm, dhpp)
    assert np.isclose(fwhm, np.sqrt(3.0) * dhpp, atol=1e-3)


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


def test_peak_finder_curvature_on_absorption():
    field = np.linspace(-5, 5, 10001)
    intensity = np.exp(-field**2)
    peaks = peak_finder(field, intensity, expected=2, method="curvature")
    assert len(peaks) == 1
    peak = peaks[0]
    assert np.isclose(field[peak], 0.0, atol=0.1)


def test_peak_finder_auto_switches_to_curvature():
    field = np.linspace(-5, 5, 10001)
    intensity = np.exp(-field**2)
    peaks = peak_finder(field, intensity, expected=2)
    assert len(peaks) == 1
    peak = peaks[0]
    assert np.isclose(field[peak], 0.0, atol=0.1)


def test_baseline_correct_manual_and_auto():
    field = np.linspace(0, 10, 11)
    baseline = 0.5 * field + 1.0
    signal = np.zeros_like(field)
    intensity = baseline + signal

    corrected, fitted = baseline_correct(
        field, intensity, points=[(0.0, baseline[0]), (10.0, baseline[-1])]
    )
    assert np.allclose(corrected, signal)
    assert np.allclose(fitted, baseline)

    corrected_auto, _ = baseline_correct(field, intensity)
    assert np.allclose(corrected_auto, signal)


def test_calc_g_expected_value():
    mu_B = physical_constants["Bohr magneton"][0]
    g_val = calc_g(339.0, 9.5)
    expected = h * 9.5e9 / (mu_B * 0.339)
    assert np.isclose(g_val, expected)
