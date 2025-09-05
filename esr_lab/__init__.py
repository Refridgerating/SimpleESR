"""ESR Lab package for analyzing ESR data."""

from .spectrum import ESRSpectrum
from .io import ESRLoader
from .plotter import ESRPlotter
from .plotter import plot_residuals
from .analysis import (
    find_peak,
    calc_fwhm,
    fit_lorentzian_derivative,
    fit_lorentzian_absorption,
    calc_peak_to_peak,
    peak_finder,
    chi_square,
    baseline_correct,
    get_resonance_field,
    calc_g,
)

__all__ = [
    "ESRSpectrum",
    "ESRLoader",
    "ESRPlotter",
    "plot_residuals",
    "find_peak",
    "calc_fwhm",
    "fit_lorentzian_derivative",
    "fit_lorentzian_absorption",
    "calc_peak_to_peak",
    "peak_finder",
    "chi_square",
    "baseline_correct",
    "get_resonance_field",
    "calc_g",
]
