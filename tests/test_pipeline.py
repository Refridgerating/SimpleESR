import numpy as np

from esr_lab.services import analyze_batch, analyze_spectrum
from esr_lab.spectrum import ESRSpectrum


def _derivative_trace(field: np.ndarray, h_res: float, delta: float, A: float, B: float) -> np.ndarray:
    x = field - h_res
    denom = (x**2 + delta**2) ** 2
    sym = -2.0 * delta**2 * x / denom
    disp = delta * (delta**2 - x**2) / denom
    return A * sym + B * disp


def test_analyze_spectrum_returns_headless_batch_payload():
    field = np.linspace(-8.0, 8.0, 2001)
    intensity = _derivative_trace(field, h_res=0.25, delta=1.5, A=2.0, B=0.3)
    spectrum = ESRSpectrum(field=field, intensity=intensity, metadata={"Frequency": 9.5})

    result = analyze_spectrum(
        spectrum,
        expected=2,
        method="auto",
        frequency_ghz=9.5,
        frequency_err_ghz=0.01,
    )

    assert "widths" in result
    assert "fits" in result
    assert len(result["widths"]) == 1
    assert len(result["fits"]) == 1
    fit = result["fits"][0]
    assert fit["kind"] == "derivative"
    assert np.isfinite(fit["h_res"])
    assert np.isfinite(fit["chi2"])
    assert fit["g"] is not None
    assert fit["g_err"] is not None
    assert fit["g_err_pct"] is not None
    assert fit["area_err_pct"] is None


def test_analyze_batch_uses_metadata_frequency_when_available():
    field = np.linspace(-8.0, 8.0, 1001)
    intensity = _derivative_trace(field, h_res=0.0, delta=1.0, A=1.0, B=0.0)
    with_freq = ESRSpectrum(field=field, intensity=intensity, metadata={"Frequency": 9.5})
    no_freq = ESRSpectrum(field=field, intensity=intensity, metadata={})

    results = analyze_batch([with_freq, no_freq], expected=2, method="auto")
    assert len(results) == 2
    assert results[0]["fits"][0]["g"] is not None
    assert results[1]["fits"][0]["g"] is None
