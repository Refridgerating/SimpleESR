"""Core numerical routines for ESR analysis."""

from .fitting import (
    fit_lorentzian_absorption,
    fit_lorentzian_derivative,
    lorentzian_absorption_model,
    lorentzian_derivative_model,
)
from .models import FitDiagnostics, FitResult
from .uncertainty import (
    propagate_g_error,
    propagate_lorentzian_area_error,
)

__all__ = [
    "FitDiagnostics",
    "FitResult",
    "lorentzian_derivative_model",
    "lorentzian_absorption_model",
    "fit_lorentzian_derivative",
    "fit_lorentzian_absorption",
    "propagate_g_error",
    "propagate_lorentzian_area_error",
]

