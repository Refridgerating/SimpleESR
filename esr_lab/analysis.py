"""Analysis utilities for ESR data."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def find_peak(field: np.ndarray, intensity: np.ndarray, start: float, end: float) -> int:
    """Find the most prominent peak within a magnetic field window.

    The function filters the provided ``field`` and ``intensity`` arrays to the
    region between ``start`` and ``end``. Local maxima and minima are then
    detected and the index of the peak with the largest absolute intensity is
    returned. The index refers to the position in the original arrays.

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
    int
        Index of the most prominent peak within the specified range.

    Raises
    ------
    ValueError
        If no data points fall within the specified range.
    """
    mask = (field >= start) & (field <= end)
    if not np.any(mask):
        raise ValueError("No data points within the specified range.")

    idx_range = np.where(mask)[0]
    sub_intensity = intensity[idx_range]

    peaks_max, _ = find_peaks(sub_intensity)
    peaks_min, _ = find_peaks(-sub_intensity)
    peaks = np.concatenate((peaks_max, peaks_min))

    if peaks.size == 0:
        rel_index = int(np.argmax(np.abs(sub_intensity)))
    else:
        rel_index = int(peaks[np.argmax(np.abs(sub_intensity[peaks]))])

    return int(idx_range[rel_index])


def calc_fwhm(field: np.ndarray, intensity: np.ndarray, peak_idx: int) -> float:
    """Calculate the full width at half maximum (FWHM) of a peak.

    The function determines the half-maximum level relative to the intensity of
    the peak at ``peak_idx``. Linear interpolation of the neighbouring points is
    used to locate where the curve crosses this level on both sides of the peak.

    Parameters
    ----------
    field:
        Array of magnetic-field values.
    intensity:
        Array of intensity values corresponding to ``field``.
    peak_idx:
        Index of the peak for which the FWHM should be calculated.

    Returns
    -------
    float
        The full width at half maximum in the same units as ``field``.

    Raises
    ------
    ValueError
        If the half-maximum level cannot be located on either side of the
        peak.
    """

    peak_intensity = intensity[peak_idx]
    half_max = peak_intensity / 2.0

    # search to the left of the peak for the half-maximum crossing
    left = None
    for i in range(peak_idx, 0, -1):
        y0 = intensity[i - 1] - half_max
        y1 = intensity[i] - half_max
        if y0 == 0:
            left = field[i - 1]
            break
        if y0 * y1 < 0:
            x0, x1 = field[i - 1], field[i]
            left = x0 + (half_max - intensity[i - 1]) * (x1 - x0) / (intensity[i] - intensity[i - 1])
            break

    # search to the right of the peak for the half-maximum crossing
    right = None
    for i in range(peak_idx, len(field) - 1):
        y0 = intensity[i] - half_max
        y1 = intensity[i + 1] - half_max
        if y1 == 0:
            right = field[i + 1]
            break
        if y0 * y1 < 0:
            x0, x1 = field[i], field[i + 1]
            right = x0 + (half_max - intensity[i]) * (x1 - x0) / (intensity[i + 1] - intensity[i])
            break

    if left is None or right is None:
        raise ValueError("Half-maximum not found on both sides of the peak.")

    return float(right - left)
