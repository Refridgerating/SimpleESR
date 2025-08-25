"""Analysis utilities for ESR data."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def find_peak(
    field: np.ndarray, intensity: np.ndarray, start: float, end: float
) -> tuple[int, int]:
    """Locate the positive and negative peaks within a field window.

    ESR spectra recorded in derivative mode exhibit a positive and a negative
    extremum for each absorption line.  When the user selects a magnetic-field
    region, both extrema have to be determined so that the distance between
    them can be used as an estimate for the full width at half maximum (FWHM).

    Parameters
    ----------
    field:
        Array of magnetic-field values.
    intensity:
        Array of recorded intensity values corresponding to ``field``.
    start:
        Lower bound of the magnetic-field window.
    end:
        Upper bound of the magnetic-field window.

    Returns
    -------
    tuple[int, int]
        Indices of the most prominent positive and negative peaks within the
        specified range. The order of the tuple is ``(positive_idx, negative_idx)``.

    Raises
    ------
    ValueError
        If no data points fall within the specified range or if either a
        positive or negative peak cannot be located.
    """

    mask = (field >= start) & (field <= end)
    if not np.any(mask):
        raise ValueError("No data points within the specified range.")

    idx_range = np.where(mask)[0]
    sub_intensity = intensity[idx_range]

    pos_peaks, _ = find_peaks(sub_intensity)
    neg_peaks, _ = find_peaks(-sub_intensity)

    if pos_peaks.size == 0 or neg_peaks.size == 0:
        raise ValueError("Both positive and negative peaks are required in the range.")

    pos_rel = int(pos_peaks[np.argmax(sub_intensity[pos_peaks])])
    neg_rel = int(neg_peaks[np.argmin(sub_intensity[neg_peaks])])

    return int(idx_range[pos_rel]), int(idx_range[neg_rel])


def calc_fwhm(
    field: np.ndarray, intensity: np.ndarray, pos_idx: int, neg_idx: int
) -> float:
    """Estimate the full width at half maximum from a peak pair.

    In derivative-mode ESR the distance between the positive and negative
    extrema of an absorption line is proportional to its FWHM.  Once both peak
    indices are known, the width can therefore be approximated simply as the
    absolute difference of their field values.

    Parameters
    ----------
    field:
        Array of magnetic-field values.
    intensity:
        Unused array of intensity values kept for API compatibility.
    pos_idx:
        Index of the positive peak.
    neg_idx:
        Index of the negative peak.

    Returns
    -------
    float
        The absolute distance between the positive and negative peak positions.
    """

    return float(abs(field[pos_idx] - field[neg_idx]))


def fit_lorentzian_derivative(
    field: np.ndarray,
    intensity: np.ndarray,
    p0: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Fit a derivative ESR line to a Lorentzian derivative model.

    The measured ESR signal in derivative mode can be represented as the
    derivative of a superposition of an absorptive (symmetric) and a dispersive
    (antisymmetric) Lorentzian line.  This function fits the recorded data to
    this model using non-linear least squares and returns the characteristic
    parameters of the resonance.

    The model function is::

        dI/dH = A * d/dH L_abs(H) + B * d/dH L_disp(H)

    where ``L_abs`` is the Lorentzian absorption line and ``L_disp`` is the
    dispersive counterpart.  ``A`` and ``B`` denote the amplitudes of the
    symmetric and antisymmetric contributions respectively.

    Parameters
    ----------
    field:
        Array of magnetic-field values.
    intensity:
        Array of derivative signal intensities corresponding to ``field``.
    p0:
        Optional initial guess for the parameters
        ``(H_res, delta, A, B)``. ``delta`` is the half width at half maximum
        (HWHM).  If omitted, a heuristic guess is derived from the data.

    Returns
    -------
    tuple[float, float, float, float]
        Fitted parameters ``(H_res, delta, A, B)``.
    """

    def _model(H: np.ndarray, H_res: float, delta: float, A: float, B: float) -> np.ndarray:
        x = H - H_res
        denom = (x**2 + delta**2) ** 2
        sym = -2.0 * delta**2 * x / denom
        disp = delta * (delta**2 - x**2) / denom
        return A * sym + B * disp

    if p0 is None:
        pos_idx = int(np.argmax(intensity))
        neg_idx = int(np.argmin(intensity))
        h_res_guess = (field[pos_idx] + field[neg_idx]) / 2.0
        delta_guess = abs(field[pos_idx] - field[neg_idx]) / 2.0
        a_guess = (intensity[pos_idx] - intensity[neg_idx]) / 2.0
        b_guess = 0.0
        p0 = (h_res_guess, delta_guess, a_guess, b_guess)

    from scipy.optimize import curve_fit

    popt, _ = curve_fit(_model, field, intensity, p0=p0)
    h_res, delta, A, B = popt
    return float(h_res), float(delta), float(A), float(B)
