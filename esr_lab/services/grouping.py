"""Replicate grouping and aggregation utilities for batch analysis."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np

_REPLICATE_TOKEN = re.compile(r"(?i)(^|[-_])r(?P<rep>\d{1,2})(?=$|[-_])")


def parse_replicate_label(label: str) -> tuple[str, int | None]:
    """Return (group_key, replicate_number) from a trace label.

    Examples
    --------
    ``CoCal-60C-IP-R3-2JAN2024`` -> (``CoCal-60C-IP-2JAN2024``, 3)
    """

    text = str(label).strip()
    match = _REPLICATE_TOKEN.search(text)
    if match is None:
        return text, None

    rep = int(match.group("rep"))
    group_key = (text[: match.start()] + text[match.end() :]).strip(" _-")
    group_key = re.sub(r"[-_]{2,}", "-", group_key)
    return group_key, rep


def _stderr_h_res(fit_row: dict[str, Any]) -> float | None:
    stderr = fit_row.get("stderr")
    if isinstance(stderr, (tuple, list)) and stderr:
        try:
            value = float(stderr[0])
        except Exception:
            return None
        if np.isfinite(value) and value > 0.0:
            return value
    return None


def _weighted_mean(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """Return weighted mean and standard error."""

    weights = 1.0 / np.square(errors)
    wsum = float(np.sum(weights))
    if wsum <= 0.0 or not np.isfinite(wsum):
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        return mean, sem
    mean = float(np.sum(weights * values) / wsum)
    sem = float(np.sqrt(1.0 / wsum))
    return mean, sem


def _robust_inlier_mask(values: np.ndarray, outlier_z: float) -> np.ndarray:
    """Return True mask of values inside robust z-score threshold."""

    if len(values) < 3:
        return np.ones(len(values), dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    if sigma <= np.finfo(float).eps or not np.isfinite(sigma):
        return np.ones(len(values), dtype=bool)
    z = np.abs((values - median) / sigma)
    return z <= float(outlier_z)


def summarize_replicate_fits(
    labels: list[str],
    payloads: list[dict[str, Any]],
    *,
    min_replicates: int = 2,
    max_chi2: float | None = 25.0,
    outlier_z: float = 3.5,
) -> list[dict[str, Any]]:
    """Aggregate per-trace fit rows into replicate group summaries.

    Grouping is inferred from labels by stripping ``R#`` tokens.
    """

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)

    for idx, payload in enumerate(payloads):
        if idx >= len(labels):
            break
        label = str(labels[idx])
        group_key, rep = parse_replicate_label(label)
        fits = payload.get("fits", [])
        if not isinstance(fits, list):
            continue
        for fit_row in fits:
            if not isinstance(fit_row, dict):
                continue
            try:
                peak = int(fit_row.get("peak", 0))
            except Exception:
                continue
            kind = str(fit_row.get("kind", "unknown"))
            grouped[(group_key, peak, kind)].append(
                {
                    "label": label,
                    "replicate": rep,
                    "fit": fit_row,
                }
            )

    summaries: list[dict[str, Any]] = []
    for (group_key, peak, kind), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        kept: list[dict[str, Any]] = []
        rejected: list[str] = []
        for row in rows:
            fit = row["fit"]
            try:
                h_res = float(fit.get("h_res", np.nan))
                chi2 = float(fit.get("chi2", np.nan))
            except Exception:
                rejected.append(f'{row["label"]}: invalid numeric fit')
                continue
            if not np.isfinite(h_res):
                rejected.append(f'{row["label"]}: non-finite h_res')
                continue
            if max_chi2 is not None and np.isfinite(chi2) and chi2 > max_chi2:
                rejected.append(f'{row["label"]}: chi2>{max_chi2}')
                continue
            kept.append(row)

        if len(kept) < min_replicates:
            continue

        h_vals = np.asarray([float(r["fit"]["h_res"]) for r in kept], dtype=float)
        inlier_mask = _robust_inlier_mask(h_vals, outlier_z=outlier_z)
        if not bool(np.all(inlier_mask)):
            for i, ok in enumerate(inlier_mask):
                if not ok:
                    rejected.append(f'{kept[i]["label"]}: h_res outlier')
        filtered = [kept[i] for i, ok in enumerate(inlier_mask) if ok]
        if len(filtered) < min_replicates:
            continue

        h = np.asarray([float(r["fit"]["h_res"]) for r in filtered], dtype=float)
        delta = np.asarray([float(r["fit"].get("delta", np.nan)) for r in filtered], dtype=float)
        chi2_vals = np.asarray([float(r["fit"].get("chi2", np.nan)) for r in filtered], dtype=float)

        h_mean = float(np.mean(h))
        h_std = float(np.std(h, ddof=1)) if len(h) > 1 else 0.0
        h_sem = float(h_std / np.sqrt(len(h))) if len(h) > 1 else 0.0

        stderr_vals = np.asarray(
            [v for v in (_stderr_h_res(r["fit"]) for r in filtered) if v is not None],
            dtype=float,
        )
        if len(stderr_vals) == len(h) and len(h) > 0:
            h_wmean, h_wsem = _weighted_mean(h, stderr_vals)
            mean_fit_var = float(np.mean(np.square(stderr_vals)))
            h_total_err = float(np.sqrt(h_sem**2 + mean_fit_var / len(h)))
        else:
            h_wmean = h_mean
            h_wsem = h_sem
            h_total_err = h_sem

        g_vals = np.asarray([float(r["fit"]["g"]) for r in filtered if r["fit"].get("g") is not None], dtype=float)
        g_mean = float(np.mean(g_vals)) if len(g_vals) else None
        g_std = float(np.std(g_vals, ddof=1)) if len(g_vals) > 1 else (0.0 if len(g_vals) == 1 else None)

        area_vals = np.asarray(
            [float(r["fit"]["area"]) for r in filtered if r["fit"].get("area") is not None],
            dtype=float,
        )
        area_mean = float(np.mean(area_vals)) if len(area_vals) else None
        area_std = (
            float(np.std(area_vals, ddof=1))
            if len(area_vals) > 1
            else (0.0 if len(area_vals) == 1 else None)
        )

        delta_valid = delta[np.isfinite(delta)]
        delta_mean = float(np.mean(delta_valid)) if len(delta_valid) else None
        delta_std = float(np.std(delta_valid, ddof=1)) if len(delta_valid) > 1 else (0.0 if len(delta_valid) == 1 else None)
        chi2_valid = chi2_vals[np.isfinite(chi2_vals)]
        chi2_mean = float(np.mean(chi2_valid)) if len(chi2_valid) else None

        summaries.append(
            {
                "group": group_key,
                "peak": peak,
                "kind": kind,
                "n_total": len(rows),
                "n_used": len(filtered),
                "h_res_mean": h_mean,
                "h_res_std": h_std,
                "h_res_sem": h_sem,
                "h_res_wmean": h_wmean,
                "h_res_wsem": h_wsem,
                "h_res_err_total": h_total_err,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "g_mean": g_mean,
                "g_std": g_std,
                "area_mean": area_mean,
                "area_std": area_std,
                "chi2_mean": chi2_mean,
                "included": "|".join(str(r["label"]) for r in filtered),
                "rejected": "|".join(rejected),
            }
        )

    return summaries

