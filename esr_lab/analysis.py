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
