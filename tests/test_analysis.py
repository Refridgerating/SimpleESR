import json
import numpy as np
import pytest

from esr_lab import (
    find_peak,
    calc_fwhm,
    fit_lorentzian_derivative,
    calc_peak_to_peak,
    peak_finder,
    chi_square,
    baseline_correct,
    get_resonance_field,
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


def test_get_resonance_field_uses_cache(tmp_path, monkeypatch):
    import esr_lab.analysis as analysis

    # Use a temporary cache file to avoid interfering with real data
    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(analysis, "H_RES_CACHE_FILE", cache_file)
    analysis.H_RES_CACHE.clear()
    analysis.load_h_res_cache()

    # Stub ``fit_lorentzian_derivative`` to track invocations
    calls = {"n": 0}

    def fake_fit(field, intensity, p0=None):
        calls["n"] += 1
        return (1.23, 0.0, 0.0, 0.0), {}

    monkeypatch.setattr(analysis, "fit_lorentzian_derivative", fake_fit)

    field = np.linspace(-5, 5, 11)
    intensity = np.zeros_like(field)

    # First call performs the fit and stores the result
    path1 = tmp_path / "sample_001.csv"
    h1 = get_resonance_field(path1, field, intensity)
    assert calls["n"] == 1

    # Second call with same base name uses cached value
    path2 = tmp_path / "sample_002.csv"
    h2 = get_resonance_field(path2, field, intensity)
    assert calls["n"] == 1
    assert h1 == h2 == 1.23

    # Cache persisted to disk
    data = json.loads(cache_file.read_text())
    assert data["sample"] == 1.23
