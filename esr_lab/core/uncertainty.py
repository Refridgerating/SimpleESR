"""Uncertainty propagation helpers for ESR derived values."""

from __future__ import annotations

import numpy as np
from scipy.constants import h, physical_constants

MU_B = physical_constants["Bohr magneton"][0]


def propagate_g_error(
    h_res: float,
    frequency: float,
    h_res_err: float,
    frequency_err: float = 0.0,
) -> float:
    """Propagate uncertainty for g = h*nu/(mu_B*H_res).

    Parameters
    ----------
    h_res:
        Resonance field in millitesla.
    frequency:
        Microwave frequency in gigahertz.
    h_res_err:
        Standard uncertainty of resonance field in millitesla.
    frequency_err:
        Standard uncertainty of frequency in gigahertz.
    """

    freq_hz = frequency * 1e9
    h_res_t = h_res * 1e-3
    g_val = h * freq_hz / (MU_B * h_res_t)

    rel_h = float(h_res_err / h_res) if h_res != 0 else np.inf
    rel_f = float(frequency_err / frequency) if frequency != 0 else 0.0
    rel = np.sqrt(rel_h**2 + rel_f**2)
    return float(abs(g_val) * rel)


def propagate_lorentzian_area_error(
    delta: float,
    amplitude: float,
    delta_err: float,
    amplitude_err: float,
    covariance: float = 0.0,
) -> float:
    """Propagate uncertainty for area = pi * A * delta.

    Parameters
    ----------
    delta:
        HWHM (delta) of Lorentzian peak.
    amplitude:
        Lorentzian amplitude A.
    delta_err:
        Standard uncertainty for ``delta``.
    amplitude_err:
        Standard uncertainty for ``amplitude``.
    covariance:
        Covariance term ``cov(amplitude, delta)``.
    """

    d_area_d_a = np.pi * delta
    d_area_d_delta = np.pi * amplitude
    variance = (
        (d_area_d_a**2) * (amplitude_err**2)
        + (d_area_d_delta**2) * (delta_err**2)
        + 2.0 * d_area_d_a * d_area_d_delta * covariance
    )
    variance = max(float(variance), 0.0)
    return float(np.sqrt(variance))

