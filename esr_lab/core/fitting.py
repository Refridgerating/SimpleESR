"""Lorentzian model definitions and fitting helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .models import FitDiagnostics, FitResult


def _reduced_residual_energy(
    observed: np.ndarray,
    expected: np.ndarray,
    dof: int | None = None,
    sigma: float | None = None,
) -> float:
    """Return reduced chi2-like residual score.

    The score uses a robust scale estimate from residual median absolute
    deviation (MAD) so it is less sensitive to raw signal magnitude:

    chi2 = (1 / dof) * sum((residual_i / sigma)^2),
    sigma ~= 1.4826 * MAD(residuals).

    If ``sigma`` is numerically zero or non-finite, fallback to
    ``sum(residual^2) / dof``.
    """

    residuals = observed - expected
    rss = float(np.sum(residuals**2))
    baseline = float(rss / dof) if (dof is not None and dof > 0) else rss
    obs_scale = float(np.sum(np.abs(observed)))
    tol = np.finfo(float).eps * max(obs_scale, 1.0)
    if rss <= tol:
        return baseline

    if sigma is not None and np.isfinite(sigma) and sigma > np.finfo(float).eps:
        scaled = residuals / float(sigma)
        rss = float(np.sum(scaled**2))
        if dof is None or dof <= 0:
            return rss
        return float(rss / dof)

    if dof is None or dof <= 0:
        return rss

    centered = residuals - np.median(residuals)
    mad = float(np.median(np.abs(centered)))
    sigma_mad = float(1.4826 * mad)
    if not np.isfinite(sigma_mad) or sigma_mad <= np.finfo(float).eps:
        return baseline
    return float(np.sum((residuals / sigma_mad) ** 2) / dof)


def lorentzian_derivative_model(
    H: np.ndarray,
    H_res: float,
    delta: float,
    A: float,
    B: float,
) -> np.ndarray:
    """Derivative-mode ESR model (absorptive + dispersive terms)."""

    x = H - H_res
    denom = (x**2 + delta**2) ** 2
    sym = -2.0 * delta**2 * x / denom
    disp = delta * (delta**2 - x**2) / denom
    return A * sym + B * disp


def lorentzian_absorption_model(
    H: np.ndarray,
    H_res: float,
    delta: float,
    A: float,
    C: float,
) -> np.ndarray:
    """Absorption-mode Lorentzian model."""

    x = H - H_res
    return A * delta**2 / (x**2 + delta**2) + C


def _sanitize_covariance(pcov: np.ndarray) -> np.ndarray:
    """Ensure covariance values are finite for downstream consumers."""

    if not np.all(np.isfinite(pcov)):
        return np.full_like(pcov, np.nan, dtype=float)
    return pcov.astype(float)


def _build_fit_result(
    field: np.ndarray,
    intensity: np.ndarray,
    model: callable,
    popt: np.ndarray,
    pcov: np.ndarray,
    *,
    sigma: float | None = None,
) -> FitResult:
    fitted = model(field, *popt)
    residuals = (intensity - fitted).astype(float)
    covariance = _sanitize_covariance(pcov)
    diag = np.diag(covariance)
    diag = np.where(np.isfinite(diag), diag, np.nan)
    diag = np.where(diag >= 0.0, diag, np.nan)
    stderr = tuple(float(v) for v in np.sqrt(diag))

    diagnostics = FitDiagnostics(
        chi2=_reduced_residual_energy(
            intensity,
            fitted,
            dof=len(field) - len(popt),
            sigma=sigma,
        ),
        stderr=stderr,  # type: ignore[arg-type]
        residuals=residuals,
        covariance=covariance,
    )
    params = tuple(float(v) for v in popt)
    return FitResult(
        params=(
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
        ),
        diagnostics=diagnostics,
    )


def _derivative_bounds(field: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    f_min = float(np.min(field))
    f_max = float(np.max(field))
    span = max(f_max - f_min, 1e-6)
    delta_min = span * 1e-6
    delta_max = span * 2.0
    amp_range = float(np.max(intensity) - np.min(intensity))
    amp_limit = max(amp_range, 1.0) * 2.0
    disp_limit = amp_limit
    lower = np.array([f_min, delta_min, -amp_limit, -disp_limit], dtype=float)
    upper = np.array([f_max, delta_max, amp_limit, disp_limit], dtype=float)
    return lower, upper


def _absorption_bounds(field: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    f_min = float(np.min(field))
    f_max = float(np.max(field))
    span = max(f_max - f_min, 1e-6)
    delta_min = span * 1e-6
    delta_max = span * 2.0
    amp_range = float(np.max(intensity) - np.min(intensity))
    amp_limit = max(amp_range, 1.0) * 2.0
    offset_min = float(np.min(intensity)) - amp_limit
    offset_max = float(np.max(intensity)) + amp_limit
    lower = np.array([f_min, delta_min, -amp_limit, offset_min], dtype=float)
    upper = np.array([f_max, delta_max, amp_limit, offset_max], dtype=float)
    return lower, upper


def fit_lorentzian_derivative(
    field: np.ndarray,
    intensity: np.ndarray,
    p0: tuple[float, float, float, float] | None = None,
    sigma: float | None = None,
) -> FitResult:
    """Fit derivative-mode ESR data to a Lorentzian derivative model."""

    if p0 is None:
        pos_idx = int(np.argmax(intensity))
        neg_idx = int(np.argmin(intensity))
        h_res_guess = (field[pos_idx] + field[neg_idx]) / 2.0
        delta_guess = abs(field[pos_idx] - field[neg_idx]) / 2.0
        a_guess = (intensity[pos_idx] - intensity[neg_idx]) / 2.0
        b_guess = 0.0
        p0 = (h_res_guess, delta_guess, a_guess, b_guess)

    lower, upper = _derivative_bounds(field, intensity)
    p0_arr = np.asarray(p0, dtype=float)
    eps = 1e-9
    p0_arr = np.clip(p0_arr, lower + eps, upper - eps)
    popt, pcov = curve_fit(
        lorentzian_derivative_model,
        field,
        intensity,
        p0=p0_arr,
        bounds=(lower, upper),
        maxfev=10000,
    )
    return _build_fit_result(
        field,
        intensity,
        lorentzian_derivative_model,
        np.asarray(popt, dtype=float),
        np.asarray(pcov, dtype=float),
        sigma=sigma,
    )


def fit_lorentzian_absorption(
    field: np.ndarray,
    intensity: np.ndarray,
    p0: tuple[float, float, float, float] | None = None,
    sigma: float | None = None,
) -> FitResult:
    """Fit absorption-mode ESR data to a Lorentzian model."""

    if p0 is None:
        peak_idx = int(np.argmax(intensity))
        h_res_guess = float(field[peak_idx])
        a_guess = float(intensity[peak_idx] - np.min(intensity))
        c_guess = float(np.min(intensity))
        if len(field) > 1:
            delta_guess = float((field[-1] - field[0]) / len(field))
        else:
            delta_guess = 1.0
        p0 = (h_res_guess, delta_guess, a_guess, c_guess)

    lower, upper = _absorption_bounds(field, intensity)
    p0_arr = np.asarray(p0, dtype=float)
    eps = 1e-9
    p0_arr = np.clip(p0_arr, lower + eps, upper - eps)
    popt, pcov = curve_fit(
        lorentzian_absorption_model,
        field,
        intensity,
        p0=p0_arr,
        bounds=(lower, upper),
        maxfev=10000,
    )
    return _build_fit_result(
        field,
        intensity,
        lorentzian_absorption_model,
        np.asarray(popt, dtype=float),
        np.asarray(pcov, dtype=float),
        sigma=sigma,
    )
