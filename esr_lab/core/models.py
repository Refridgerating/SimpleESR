"""Typed data models for core ESR computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitDiagnostics:
    """Diagnostics produced by a non-linear least-squares fit."""

    chi2: float
    stderr: tuple[float, float, float, float]
    residuals: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class FitResult:
    """Result container for 4-parameter Lorentzian fits."""

    params: tuple[float, float, float, float]
    diagnostics: FitDiagnostics

