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


def calc_peak_to_peak(
    field: np.ndarray, intensity: np.ndarray, pos_idx: int, neg_idx: int
) -> float:
    r"""Compute the peak-to-peak separation :math:`\Delta H_{pp}`.

    The peak-to-peak width of a derivative ESR line is defined as the distance
    in magnetic-field units between the positive and the negative extrema.  It
    is commonly denoted :math:`\Delta H_{pp}` and is a convenient measure for
    the line width without converting to the absorption representation.

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
        The absolute field distance between the two peaks.
    """

    return float(abs(field[pos_idx] - field[neg_idx]))


def peak_finder(
    field: np.ndarray, intensity: np.ndarray, expected: int = 4
) -> list[tuple[int, int]]:
    """Automatically locate peak pairs in the provided data.

    Parameters
    ----------
    field:
        Array of magnetic-field values.
    intensity:
        Array of recorded intensity values corresponding to ``field``.
    expected:
        Total number of extrema to locate.  This should be an even number as
        each absorption line exhibits a positive and a negative extremum.  The
        default of ``4`` therefore corresponds to two absorption lines.

    Returns
    -------
    list[tuple[int, int]]
        A list of ``(positive_idx, negative_idx)`` tuples ordered by increasing
        field.  Each tuple represents one absorption line with the positive and
        negative peaks chosen as the closest pair in magnetic field.

    Raises
    ------
    ValueError
        If an insufficient number of extrema is found or if ``expected`` is not
        a positive, even integer.
    """

    if expected < 2 or expected % 2 != 0:
        raise ValueError("Expected number of peaks must be an even integer >= 2")

    pos_peaks, _ = find_peaks(intensity)
    neg_peaks, _ = find_peaks(-intensity)

    n_pairs = expected // 2
    if len(pos_peaks) < n_pairs or len(neg_peaks) < n_pairs:
        raise ValueError("Not enough peaks found in the data")

    # Select the most prominent extrema first.
    pos_order = np.argsort(intensity[pos_peaks])[-n_pairs:]
    neg_order = np.argsort(intensity[neg_peaks])[:n_pairs]
    pos_idx = pos_peaks[pos_order]
    neg_idx = neg_peaks[neg_order]

    # Determine pairs based on proximity in the magnetic field so that each
    # positive/negative combination corresponds to a single absorption peak.
    pos_fields = field[pos_idx]
    neg_fields = field[neg_idx]
    distances: list[tuple[float, int, int]] = []
    for p_idx, p in enumerate(pos_idx):
        for n_idx, n in enumerate(neg_idx):
            dist = abs(pos_fields[p_idx] - neg_fields[n_idx])
            distances.append((float(dist), int(p), int(n)))

    distances.sort(key=lambda x: x[0])
    used_pos: set[int] = set()
    used_neg: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _dist, p, n in distances:
        if p in used_pos or n in used_neg:
            continue
        pairs.append((p, n))
        used_pos.add(p)
        used_neg.add(n)
        if len(pairs) == n_pairs:
            break

    if len(pairs) < n_pairs:
        raise ValueError("Not enough distinct peak pairs found")

    pairs.sort(key=lambda pr: min(field[pr[0]], field[pr[1]]))
    return [(int(p), int(n)) for p, n in pairs]


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
