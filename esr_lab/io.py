"""Data loading utilities for ESR spectra."""

from pathlib import Path
from typing import Union
from io import StringIO

import pandas as pd

from .spectrum import ESRSpectrum


class ESRLoader:
    """Loader for ESR spectra files."""

    @staticmethod
    def load_csv(path: Union[str, Path]) -> ESRSpectrum:
        """Load a spectrum from a CSV file.

        The ESR control software exports data as CSV files with a large
        metadata header followed by a ``Meas`` section containing the actual
        spectrum.  The measurement block starts with a line
        ``BField [mT];MW_Absorption []`` and uses semicolons as delimiters.
        For backwards compatibility plain CSV files without such a header are
        also supported.

        Parameters
        ----------
        path:
            Path to the CSV file containing at least two columns: field and
            intensity.
        """

        path = Path(path)

        # Read all lines first so we can inspect the header and strip the UTF-8
        # byte order mark if present.
        with path.open("r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        # Find the beginning of the measurement block.
        data_start = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("BField"):
                data_start = idx
                break

        if data_start is not None:
            # Parse only the measurement block using a fixed semicolon
            # delimiter.  This matches the export format of our spectrometer.
            data = "".join(lines[data_start:])
            df = pd.read_csv(StringIO(data), sep=";", engine="python")
        else:
            # Fall back to automatic delimiter detection for generic CSV files
            # that contain only data without a metadata header.
            df = pd.read_csv(StringIO("".join(lines)), sep=None, engine="python")

        # Ensure we have at least two columns for field and intensity.
        if df.shape[1] < 2:
            raise ValueError(
                "CSV file must contain at least two columns for field and intensity",
            )

        # Use only the first two columns and give them standard names.
        df = df.iloc[:, :2].copy()
        df.columns = ["BField [mT]", "MW_Absorption []"]

        return ESRSpectrum.from_dataframe(df)

