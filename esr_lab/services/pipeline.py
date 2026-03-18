"""Headless analysis pipeline designed for batch processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from ..analysis import (
    calc_fwhm,
    calc_g,
    calc_lorentzian_area,
    calc_peak_to_peak,
    peak_finder,
)
from ..core import (
    FitResult,
    fit_lorentzian_absorption,
    fit_lorentzian_derivative,
    propagate_g_error,
    propagate_lorentzian_area_error,
)
from ..spectrum import ESRSpectrum


@dataclass(frozen=True)
class PeakWidthResult:
    peak: int
    pos_idx: int
    neg_idx: int
    peak_to_peak: float
    fwhm: float


@dataclass(frozen=True)
class FitSummary:
    peak: int
    kind: str
    h_res: float
    delta: float
    A: float
    B: float
    chi2: float
    stderr: tuple[float, float, float, float]
    g: float | None = None
    g_err: float | None = None
    g_err_pct: float | None = None
    area: float | None = None
    area_err: float | None = None
    area_err_pct: float | None = None


def _absorption_initial_guess(
    field: np.ndarray,
    intensity: np.ndarray,
    peak_idx: int,
) -> tuple[float, float, float, float]:
    """Derive a stable initial guess for absorption-mode fitting."""

    h_res_guess = float(field[peak_idx])
    peak_val = float(intensity[peak_idx])
    half_val = peak_val / 2.0

    left = int(peak_idx)
    while left > 0 and intensity[left] > half_val:
        left -= 1
    right = int(peak_idx)
    while right < len(intensity) - 1 and intensity[right] > half_val:
        right += 1

    if left == peak_idx or right == peak_idx:
        delta_guess = abs(float(field[1] - field[0])) * 5.0 if len(field) > 1 else 1.0
    else:
        delta_guess = abs(float(field[right] - field[left])) / 2.0

    a_guess = peak_val - float(np.min(intensity))
    c_guess = float(np.min(intensity))
    return (h_res_guess, delta_guess, a_guess, c_guess)


def _derivative_initial_guess(
    field: np.ndarray,
    intensity: np.ndarray,
    pos_idx: int,
    neg_idx: int,
) -> tuple[float, float, float, float]:
    h_res_guess = float((field[pos_idx] + field[neg_idx]) / 2.0)
    delta_guess = float(abs(field[pos_idx] - field[neg_idx]) / 2.0)
    a_guess = float((intensity[pos_idx] - intensity[neg_idx]) / 2.0)
    b_guess = 0.0
    return (h_res_guess, delta_guess, a_guess, b_guess)


def _relative_error_percent(value: float | None, error: float | None) -> float | None:
    if value is None or error is None:
        return None
    if not np.isfinite(value) or not np.isfinite(error) or value == 0.0:
        return None
    return float(100.0 * abs(error) / abs(value))


def _seed_key(seed: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(round(float(v), 8) for v in seed)


def _add_seed(
    seeds: list[tuple[float, float, float, float]],
    seen: set[tuple[float, float, float, float]],
    seed: tuple[float, float, float, float],
) -> None:
    h, delta, amp, extra = seed
    delta = float(max(delta, 1e-9))
    normalized = (float(h), delta, float(amp), float(extra))
    key = _seed_key(normalized)
    if key in seen:
        return
    seen.add(key)
    seeds.append(normalized)


def _derivative_seed_variants(p0: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    h, delta, amp, disp = p0
    seeds: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    span = max(abs(delta), 1.0)
    _add_seed(seeds, seen, (h, delta, amp, disp))
    for frac in (-0.05, 0.05):
        _add_seed(seeds, seen, (h + frac * span, delta, amp, disp))
        _add_seed(seeds, seen, (h, delta * (1.0 + frac), amp, disp))
        _add_seed(seeds, seen, (h, delta, amp * (1.0 + frac), disp))
        _add_seed(seeds, seen, (h + frac * span, delta * (1.0 + frac), amp * (1.0 + frac), disp))
    return seeds


def _absorption_seed_variants(p0: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    h, delta, amp, offset = p0
    seeds: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    span = max(abs(delta), 1.0)
    _add_seed(seeds, seen, (h, delta, amp, offset))
    for frac in (-0.05, 0.05):
        _add_seed(seeds, seen, (h + frac * span, delta, amp, offset))
        _add_seed(seeds, seen, (h, delta * (1.0 + frac), amp, offset))
        _add_seed(seeds, seen, (h, delta, amp * (1.0 + frac), offset))
        _add_seed(seeds, seen, (h + frac * span, delta * (1.0 + frac), amp * (1.0 + frac), offset))
    return seeds


def _best_fit_from_seeds(
    fitter: Callable[..., FitResult],
    field: np.ndarray,
    intensity: np.ndarray,
    seeds: list[tuple[float, float, float, float]],
    sigma: float | None,
) -> FitResult:
    best: FitResult | None = None
    last_error: Exception | None = None
    for seed in seeds:
        try:
            # Support both the legacy signature and the sigma-aware version
            try:
                fit = fitter(field, intensity, p0=seed, sigma=sigma)  # type: ignore[arg-type]
            except TypeError:
                fit = fitter(field, intensity, p0=seed)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - convergence failures are rare
            last_error = exc
            continue
        if best is None or fit.diagnostics.chi2 < best.diagnostics.chi2:
            best = fit
    if best is not None:
        return best
    if last_error is not None:
        raise last_error
    raise RuntimeError("All Lorentzian fit attempts failed")


def _best_derivative_fit(
    field: np.ndarray,
    intensity: np.ndarray,
    pos_idx: int,
    neg_idx: int,
    sigma: float | None,
) -> FitResult:
    base = _derivative_initial_guess(field, intensity, pos_idx, neg_idx)
    seeds = _derivative_seed_variants(base)
    return _best_fit_from_seeds(fit_lorentzian_derivative, field, intensity, seeds, sigma)


def _best_absorption_fit(
    field: np.ndarray,
    intensity: np.ndarray,
    peak_idx: int,
    sigma: float | None,
) -> FitResult:
    base = _absorption_initial_guess(field, intensity, peak_idx)
    seeds = _absorption_seed_variants(base)
    return _best_fit_from_seeds(fit_lorentzian_absorption, field, intensity, seeds, sigma)


def _noise_sigma_from_metadata(metadata: dict[str, Any] | None) -> float | None:
    if not isinstance(metadata, dict):
        return None
    candidates: list[float] = []
    for key, value in metadata.items():
        norm = str(key).strip().lower()
        if "noise" not in norm:
            continue
        if not any(token in norm for token in ("std", "sigma", "rms", "dev")):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0:
            candidates.append(float(val))
    if not candidates:
        return None
    return float(min(candidates))


def analyze_spectrum(
    spectrum: ESRSpectrum,
    *,
    expected: int = 4,
    method: str = "auto",
    frequency_ghz: float | None = None,
    frequency_err_ghz: float = 0.0,
) -> dict[str, Any]:
    """Analyze one spectrum without any GUI side effects."""

    field = np.asarray(spectrum.field, dtype=float)
    intensity = np.asarray(spectrum.intensity, dtype=float)
    noise_sigma = _noise_sigma_from_metadata(spectrum.metadata)
    peaks = peak_finder(field, intensity, expected=expected, method=method)

    widths: list[PeakWidthResult] = []
    fits: list[FitSummary] = []

    if peaks and isinstance(peaks[0], tuple):
        pair_peaks = peaks  # derivative mode
        derivative_candidates: list[dict[str, Any]] = []
        for pos_idx, neg_idx in pair_peaks:
            peak_to_peak = calc_peak_to_peak(field, intensity, pos_idx, neg_idx)
            fwhm = calc_fwhm(field, intensity, pos_idx, neg_idx)
            width_info = {
                "pos_idx": int(pos_idx),
                "neg_idx": int(neg_idx),
                "peak_to_peak": float(peak_to_peak),
                "fwhm": float(fwhm),
            }
            fit = _best_derivative_fit(field, intensity, pos_idx, neg_idx, noise_sigma)
            h_res, delta, A, B = fit.params
            stderr = tuple(float(v) for v in fit.diagnostics.stderr)
            g_val = None
            g_err = None
            g_err_pct = None
            if frequency_ghz is not None:
                g_val = calc_g(h_res, frequency_ghz)
                g_err = propagate_g_error(
                    h_res=h_res,
                    frequency=frequency_ghz,
                    h_res_err=float(stderr[0]),
                    frequency_err=frequency_err_ghz,
                )
                g_err_pct = _relative_error_percent(g_val, g_err)
            fit_info = {
                "kind": "derivative",
                "h_res": float(h_res),
                "delta": float(delta),
                "A": float(A),
                "B": float(B),
                "chi2": float(fit.diagnostics.chi2),
                "stderr": stderr,
                "g": g_val,
                "g_err": g_err,
                "g_err_pct": g_err_pct,
            }
            derivative_candidates.append(
                {
                    "chi2": fit_info["chi2"],
                    "pair": (int(pos_idx), int(neg_idx)),
                    "width": width_info,
                    "fit": fit_info,
                }
            )
        derivative_candidates.sort(key=lambda item: item["chi2"])
        if derivative_candidates:
            peaks = []
        for new_peak, entry in enumerate(derivative_candidates, start=1):
            peaks.append(entry["pair"])
            width_info = entry["width"]
            widths.append(
                PeakWidthResult(
                    peak=new_peak,
                    pos_idx=width_info["pos_idx"],
                    neg_idx=width_info["neg_idx"],
                    peak_to_peak=width_info["peak_to_peak"],
                    fwhm=width_info["fwhm"],
                )
            )
            fit_info = entry["fit"]
            fits.append(
                FitSummary(
                    peak=new_peak,
                    kind=fit_info["kind"],
                    h_res=fit_info["h_res"],
                    delta=fit_info["delta"],
                    A=fit_info["A"],
                    B=fit_info["B"],
                    chi2=fit_info["chi2"],
                    stderr=fit_info["stderr"],
                    g=fit_info["g"],
                    g_err=fit_info["g_err"],
                    g_err_pct=fit_info["g_err_pct"],
                )
            )
    else:
        abs_peaks = peaks  # absorption mode
        absorption_candidates: list[dict[str, Any]] = []
        for peak_idx in abs_peaks:
            fit = _best_absorption_fit(field, intensity, int(peak_idx), noise_sigma)
            h_res, delta, A, C = fit.params
            stderr = tuple(float(v) for v in fit.diagnostics.stderr)
            g_val = None
            g_err = None
            g_err_pct = None
            if frequency_ghz is not None:
                g_val = calc_g(h_res, frequency_ghz)
                g_err = propagate_g_error(
                    h_res=h_res,
                    frequency=frequency_ghz,
                    h_res_err=float(stderr[0]),
                    frequency_err=frequency_err_ghz,
                )
                g_err_pct = _relative_error_percent(g_val, g_err)
            area = calc_lorentzian_area(delta, A)
            cov = float(fit.diagnostics.covariance[2, 1]) if fit.diagnostics.covariance.shape == (4, 4) else 0.0
            area_err = propagate_lorentzian_area_error(
                delta=float(delta),
                amplitude=float(A),
                delta_err=float(stderr[1]),
                amplitude_err=float(stderr[2]),
                covariance=cov,
            )
            area_err_pct = _relative_error_percent(area, area_err)
            absorption_candidates.append(
                {
                    "chi2": float(fit.diagnostics.chi2),
                    "peak_idx": int(peak_idx),
                    "fit": {
                        "kind": "absorption",
                        "h_res": float(h_res),
                        "delta": float(delta),
                        "A": float(A),
                        "B": float(C),
                        "chi2": float(fit.diagnostics.chi2),
                        "stderr": stderr,
                        "g": g_val,
                        "g_err": g_err,
                        "g_err_pct": g_err_pct,
                        "area": float(area),
                        "area_err": float(area_err),
                        "area_err_pct": area_err_pct,
                    },
                }
            )
        absorption_candidates.sort(key=lambda item: item["chi2"])
        if absorption_candidates:
            peaks = []
        for new_peak, entry in enumerate(absorption_candidates, start=1):
            peaks.append(entry["peak_idx"])
            fit_info = entry["fit"]
            fits.append(
                FitSummary(
                    peak=new_peak,
                    kind=fit_info["kind"],
                    h_res=fit_info["h_res"],
                    delta=fit_info["delta"],
                    A=fit_info["A"],
                    B=fit_info["B"],
                    chi2=fit_info["chi2"],
                    stderr=fit_info["stderr"],
                    g=fit_info["g"],
                    g_err=fit_info["g_err"],
                    g_err_pct=fit_info["g_err_pct"],
                    area=fit_info["area"],
                    area_err=fit_info["area_err"],
                    area_err_pct=fit_info["area_err_pct"],
                )
            )

    return {
        "metadata": dict(spectrum.metadata or {}),
        "peaks": peaks,
        "widths": [asdict(item) for item in widths],
        "fits": [asdict(item) for item in fits],
    }


def analyze_batch(
    spectra: list[ESRSpectrum],
    *,
    expected: int = 4,
    method: str = "auto",
    frequency_key: str = "Frequency",
    frequency_err_ghz: float = 0.0,
) -> list[dict[str, Any]]:
    """Analyze multiple spectra with a consistent headless pipeline."""

    results: list[dict[str, Any]] = []
    for spectrum in spectra:
        freq = None
        if spectrum.metadata and frequency_key in spectrum.metadata:
            try:
                freq = float(spectrum.metadata[frequency_key])
            except Exception:
                freq = None
        results.append(
            analyze_spectrum(
                spectrum,
                expected=expected,
                method=method,
                frequency_ghz=freq,
                frequency_err_ghz=frequency_err_ghz,
            )
        )
    return results
