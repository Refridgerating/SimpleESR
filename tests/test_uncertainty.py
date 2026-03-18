import numpy as np

from esr_lab import (
    calc_g,
    calc_g_error,
    calc_lorentzian_area,
    calc_lorentzian_area_error,
)


def test_calc_g_error_matches_first_order_propagation():
    h_res = 339.0
    freq = 9.5
    h_res_err = 0.8
    freq_err = 0.02

    g_val = calc_g(h_res, freq)
    expected = abs(g_val) * np.sqrt((h_res_err / h_res) ** 2 + (freq_err / freq) ** 2)
    propagated = calc_g_error(h_res, freq, h_res_err, freq_err)

    assert np.isclose(propagated, expected)


def test_calc_lorentzian_area_error_with_covariance():
    delta = 2.0
    amp = 3.0
    delta_err = 0.1
    amp_err = 0.2
    cov = 0.01

    area = calc_lorentzian_area(delta, amp)
    expected_area = np.pi * amp * delta
    assert np.isclose(area, expected_area)

    d_area_d_a = np.pi * delta
    d_area_d_delta = np.pi * amp
    expected_var = (
        (d_area_d_a**2) * (amp_err**2)
        + (d_area_d_delta**2) * (delta_err**2)
        + 2.0 * d_area_d_a * d_area_d_delta * cov
    )
    expected_err = np.sqrt(expected_var)

    propagated = calc_lorentzian_area_error(delta, amp, delta_err, amp_err, cov)
    assert np.isclose(propagated, expected_err)

