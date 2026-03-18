"""Qt worker objects for asynchronous tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from ..io import ESRLoader
from ..services import analyze_spectrum
from ..spectrum import ESRSpectrum


def pipeline_frequency_from_metadata(spectrum: ESRSpectrum) -> float | None:
    """Best-effort frequency extraction from spectrum metadata."""

    if not isinstance(spectrum.metadata, dict):
        return None
    if "Frequency" not in spectrum.metadata:
        return None
    try:
        return float(spectrum.metadata["Frequency"])
    except Exception:
        return None


class PipelineWorker(QObject):
    """Background worker for running pipeline analysis."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(
        self,
        *,
        spectra: list[ESRSpectrum],
        indices: list[int],
        expected_peaks: int,
        peak_method: str,
    ) -> None:
        super().__init__()
        self._spectra = spectra
        self._indices = indices
        sanitized = max(2, int(expected_peaks))
        if sanitized % 2 != 0:
            sanitized += 1
        self._expected_peaks = sanitized
        method = str(peak_method).strip().lower()
        self._peak_method = method if method in {"auto", "zero", "curvature"} else "auto"

    @Slot()
    def run(self) -> None:
        results: list[tuple[int, dict[str, Any]]] = []
        total = max(len(self._indices), 1)
        for n, idx in enumerate(self._indices, start=1):
            label = f"Trace {idx + 1}"
            self.progress.emit(n, total, label)
            try:
                spectrum = self._spectra[idx]
                payload = analyze_spectrum(
                    spectrum,
                    expected=self._expected_peaks,
                    method=self._peak_method,
                    frequency_ghz=pipeline_frequency_from_metadata(spectrum),
                )
                results.append((idx, payload))
            except Exception as exc:
                self.failed.emit(f"{label}: {exc}")
                return
        self.finished.emit(results)


class FileLoadWorker(QObject):
    """Background worker for loading many CSV files."""

    finished = Signal(object, object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, *, paths: list[str]) -> None:
        super().__init__()
        self._paths = paths

    @Slot()
    def run(self) -> None:
        loaded: list[tuple[str, ESRSpectrum]] = []
        errors: list[str] = []

        total = max(len(self._paths), 1)
        for n, raw_path in enumerate(self._paths, start=1):
            path = Path(raw_path)
            self.progress.emit(n, total, path.name)
            try:
                spec = ESRLoader.load_csv(path)
                loaded.append((path.name, spec))
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        self.finished.emit(loaded, errors)
