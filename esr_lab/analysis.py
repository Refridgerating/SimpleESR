"""Analysis utilities for ESR data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import sympy as sp

# ---------------------------------------------------------------------------
# Cache for previously determined resonance fields
# ---------------------------------------------------------------------------
# Stores resonance fields keyed by the base name of the analysed file.  The
# cache is persisted alongside this module so values survive across sessions.
H_RES_CACHE: dict[str, float] = {}
H_RES_CACHE_FILE = Path(__file__).with_name("h_res_cache.json")


def load_h_res_cache() -> None:
    """Populate :data:`H_RES_CACHE` from the on-disk JSON cache if present."""

    if H_RES_CACHE_FILE.exists():
        try:
            data = json.loads(H_RES_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                H_RES_CACHE.update({str(k): float(v) for k, v in data.items()})
        except (OSError, json.JSONDecodeError):
            pass


def save_h_res_cache() -> None:
    """Serialise :data:`H_RES_CACHE` to :data:`H_RES_CACHE_FILE`."""

    try:
        H_RES_CACHE_FILE.write_text(
            json.dumps(H_RES_CACHE), encoding="utf-8"
        )
    except OSError:
        pass


# Load existing cache on import so repeated analyses can reuse stored values.
load_h_res_cache()


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
    r"""Estimate the full width at half maximum from a peak pair.

    In derivative-mode ESR the distance between the positive and negative
    extrema of an absorption line is not the FWHM itself.  For a Lorentzian
    line shape the conversion factor between the peak-to-peak distance
    :math:`\Delta H_{pp}` and the true FWHM is :math:`\sqrt{3}`.  The width is
    therefore obtained by multiplying the separation of the extrema by this
    factor.  The ``intensity`` array is currently unused but retained for API
    compatibility.

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
        The estimated FWHM assuming a Lorentzian line shape.
    """

    delta_h_pp = abs(field[pos_idx] - field[neg_idx])
    return float(np.sqrt(3.0) * delta_h_pp)


def calc_peak_to_peak(
    field: np.ndarray, intensity: np.ndarray, pos_idx: int, neg_idx: int
) -> float:
    r"""Compute the peak-to-peak separation :math:`Delta H_{pp}`.

    The peak-to-peak width of a derivative ESR line is defined as the distance
    in magnetic-field units between the positive and the negative extrema.  It
    is commonly denoted :math:`Delta H_{pp}` and is a convenient measure for
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
    field: np.ndarray,
    intensity: np.ndarray,
    expected: int = 4,
    width: float = 15.0,
    method: str = "auto",
) -> list[tuple[int, int]] | list[int]:
    """Automatically locate peak pairs in the provided data.

    Two approaches are supported:

    ``method="auto"`` (default)
        Automatically chooses between the ``"zero"`` and ``"curvature"``
        strategies based on the presence of zero crossings in ``intensity``.
        If no zero crossings are found, the curvature approach is used.

    ``method="zero"``
        Operates on derivative-mode data by locating zero crossings and
        searching for extrema in a window of ``±width`` around each crossing.

    ``method="curvature"``
        Intended for absorption spectra.  It searches the trace for local
        maxima only.  The ``width`` parameter is ignored in this mode.

    Parameters
    ----------
    field:
        Array of magnetic-field values.
    intensity:
        Array of recorded intensity values corresponding to ``field``.
    expected:
        Total number of extrema to locate.  This should be an even number.  For
        derivative data the value counts individual maxima and minima, whereas
        for absorption data the number of positive peaks returned is
        ``expected // 2``.  The default of ``4`` therefore corresponds to two
        peak pairs in derivative mode or two positive peaks in absorption
        mode.
    width:
        Half width in magnetic-field units used around each zero crossing to
        determine the local maxima and minima.  The default of ``15.0`` mT
        mirrors the manual analysis range used in the GUI.  Ignored when
        ``"curvature"``.
    method:
        ``"zero"`` to base the detection on zero crossings of the provided
        trace (appropriate for derivative spectra) or ``"curvature"`` to analyse
        the curvature of an absorption trace.

    Returns
    -------
    list[tuple[int, int]] | list[int]
        For derivative data (``method="zero"``) a list of ``(positive_idx,
        negative_idx)`` tuples ordered by increasing field.  Each tuple
        represents one absorption line.  For absorption data
        (``method="curvature"``) a list of indices of positive peaks ordered by
        increasing field is returned.

    Raises
    ------
    ValueError
        If an insufficient number of extrema is found or if ``expected`` is not
        a positive, even integer.
    """

    if expected < 2 or expected % 2 != 0:
        raise ValueError("Expected number of peaks must be an even integer >= 2")

    method = method.lower()
    if method not in {"zero", "curvature", "auto"}:
        raise ValueError("method must be 'zero', 'curvature', or 'auto'")

    zero_crossings: np.ndarray | None = None
    if method in {"zero", "auto"}:
        # Identify zero crossings where the signal changes from positive to
        # negative.  Zeros are ignored by using ``nan`` and compressing the array
        # to regions of constant sign.
        sign = np.sign(intensity)
        sign = np.where(sign == 0, np.nan, sign)
        nonzero_idx = np.where(~np.isnan(sign))[0]
        if nonzero_idx.size == 0:
            raise ValueError("No non-zero data points found")
        nonzero_sign = sign[nonzero_idx]
        changes = np.where((nonzero_sign[:-1] > 0) & (nonzero_sign[1:] < 0))[0]
        zero_crossings = (nonzero_idx[changes] + nonzero_idx[changes + 1]) // 2
        if zero_crossings.size == 0:
            method = "curvature"
        elif method == "auto":
            method = "zero"

    if method == "zero" and zero_crossings is not None:
        pairs: list[tuple[int, int]] = []
        for i, zc in enumerate(zero_crossings):
            center = field[zc]
            left_bound = center - width
            right_bound = center + width
            if i > 0:
                left_bound = max(left_bound, field[zero_crossings[i - 1]])
            if i < len(zero_crossings) - 1:
                right_bound = min(right_bound, field[zero_crossings[i + 1]])

            left_mask = (field >= left_bound) & (field <= center)
            right_mask = (field >= center) & (field <= right_bound)
            left_idx = np.where(left_mask)[0]
            right_idx = np.where(right_mask)[0]
            if left_idx.size == 0 or right_idx.size == 0:
                continue

            pos_idx = left_idx[np.argmax(intensity[left_idx])]
            neg_idx = right_idx[np.argmin(intensity[right_idx])]
            pairs.append((int(pos_idx), int(neg_idx)))

        n_pairs = expected // 2
        if len(pairs) < n_pairs:
            raise ValueError("Not enough peaks found in the data")

        # Rank pairs by their peak-to-peak amplitude and select the most
        # prominent
        pairs = sorted(
            pairs,
            key=lambda p: abs(intensity[p[0]] - intensity[p[1]]),
            reverse=True,
        )[:n_pairs]

        pairs.sort(key=lambda p: field[p[0]])
        return pairs

    # Curvature-based peak finder for absorption traces
    n_peaks = expected // 2
    pos_peaks, _ = find_peaks(intensity)
    if pos_peaks.size < n_peaks:
        raise ValueError("Not enough peaks found in the data")

    # Select the most prominent peaks by their height
    top = np.argsort(intensity[pos_peaks])[::-1][:n_peaks]
    peaks = pos_peaks[top]
    peaks.sort()
    return [int(p) for p in peaks]


def baseline_correct(
    field: np.ndarray,
    intensity: np.ndarray,
    points: list[tuple[float, float]] | None = None,
    degree: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a simple polynomial baseline correction.

    Parameters
    ----------
    field:
        Array of magnetic-field values.
    intensity:
        Recorded intensity values corresponding to ``field``.
    points:
        Optional list of ``(x, y)`` coordinates defining the baseline.  When
        provided a polynomial is fitted through these points.  Otherwise a
        polynomial of ``degree`` is fitted to the entire trace.
    degree:
        Degree of the polynomial used for automatic baseline fitting.  The
        default of ``1`` corresponds to a linear baseline.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The baseline corrected intensity and the fitted baseline evaluated over
        ``field``.
    """

    if points:
        xp = np.array([p[0] for p in points], dtype=float)
        yp = np.array([p[1] for p in points], dtype=float)
        deg = min(degree, len(xp) - 1)
        coeffs = np.polyfit(xp, yp, deg)
    else:
        coeffs = np.polyfit(field, intensity, degree)

    baseline = np.polyval(coeffs, field)
    corrected = intensity - baseline
    return corrected.astype(float), baseline.astype(float)


def chi_square(
    observed: np.ndarray,
    expected: np.ndarray,
    dof: int | None = None,
) -> float:
    """Calculate the (reduced) chi-square statistic for two data sets.

    Parameters
    ----------
    observed:
        Array of measured values.
    expected:
        Array of expected values from a model.
    dof:
        Degrees of freedom. If provided, the returned value is the reduced
        chi-square where ``chi2 / dof`` is computed.  When omitted the raw
        chi-square sum of squared residuals is returned.

    Returns
    -------
    float
        The chi-square or reduced chi-square statistic.
    """

    residuals = observed - expected
    chi2 = float(np.sum(residuals**2))
    if dof is not None and dof > 0:
        chi2 /= dof
    return chi2


def fit_lorentzian_derivative(
    field: np.ndarray,
    intensity: np.ndarray,
    p0: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[float, float, float, float], dict[str, object]]:
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
    dict[str, object]
        A dictionary with diagnostic information.  Keys are ``"chi2"`` for the
        reduced chi-square statistic, ``"stderr"`` containing the standard error
        of the fitted parameters as a tuple and ``"residuals"`` holding the
        residual array.
    """
    ''''
    def _model(H: np.ndarray, H_0: float, gamma_0: float, H_1: float, gamma_1: float) -> np.ndarray:
        x = H - H_0
        y = H - H_1
        return -2*x/(gamma_0**2*(1+(x/gamma_0)**2)**2) + -2*y/(gamma_1**2*(1+(y/gamma_1)**2)**2)
    '''

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

    popt, pcov = curve_fit(_model, field, intensity, p0=p0)
    h_res, delta, A, B = popt

    fitted = _model(field, h_res, delta, A, B)
    residuals = intensity - fitted
    chi2 = chi_square(intensity, fitted, dof=len(field) - len(popt))
    stderr = np.sqrt(np.diag(pcov))
    stats = {
        "chi2": float(chi2),
        "stderr": tuple(float(s) for s in stderr),
        "residuals": residuals.astype(float),
    }

    return (float(h_res), float(delta), float(A), float(B)), stats


def get_resonance_field(
    path: str | Path,
    field: np.ndarray,
    intensity: np.ndarray,
    p0: tuple[float, float, float, float] | None = None,
) -> float:
    """Return the resonance field for ``path`` using a cached value when available.

    Parameters
    ----------
    path:
        Path to the analysed trace.  The base name of the file (portion before
        the first underscore) is used as the cache key.
    field, intensity:
        Arrays containing the spectral data used for the fit when a cached value
        is unavailable.
    p0:
        Optional initial guess forwarded to :func:`fit_lorentzian_derivative`
        when a fit is performed.
    """

    base = Path(path).stem.split("_")[0]
    if base in H_RES_CACHE:
        return H_RES_CACHE[base]

    params, _stats = fit_lorentzian_derivative(field, intensity, p0=p0)
    h_res = params[0]
    H_RES_CACHE[base] = h_res
    save_h_res_cache()
    return h_res


# ---------------------------------------------------------------------------
# Metadata for help menu
# ---------------------------------------------------------------------------
H_plus, H_minus = sp.symbols("H_+ H_-")
FWHM, ΔH_pp, H = sp.symbols("FWHM ΔH_pp H")
A, B = sp.symbols("A B")
I = sp.Function("I")
L_abs = sp.Function("L_abs")
L_disp = sp.Function("L_disp")

# Symbols for displaying derivative expressions as textbook-style fractions
dI_H, dL_abs_H, dL_disp_H, dH = sp.symbols(
    "dI(H) dL_abs(H) dL_disp(H) dH"
)

FUNCTION_DETAILS: dict[str, tuple[str, sp.Expr | None]] = {
    "find_peak": (
        "Locate the positive and negative peaks within a field window",
        None,
    ),
    "calc_fwhm": (
        "Estimate the full width at half maximum (FWHM) assuming a Lorentzian line",
        sp.Eq(FWHM, sp.sqrt(3) * sp.Abs(H_plus - H_minus)),
    ),
    "calc_peak_to_peak": (
        "Compute the peak-to-peak separation ΔH_pp",
        sp.Eq(ΔH_pp, sp.Abs(H_plus - H_minus)),
    ),
    "peak_finder": (
        "Automatically locate peak pairs in the provided data",
        None,
    ),
    "baseline_correct": (
        "Perform a polynomial baseline correction",
        None,
    ),
    "fit_lorentzian_derivative": (
        "Fit a derivative ESR line to a Lorentzian derivative model",
        sp.Eq(
            dI_H / dH,
            A * dL_abs_H / dH + B * dL_disp_H / dH,
        ),
    ),
}
