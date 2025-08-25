"""Data loading utilities for ESR spectra."""

from pathlib import Path
from typing import Union

import pandas as pd

from .spectrum import ESRSpectrum


class ESRLoader:
    """Loader for ESR spectra files."""

    @staticmethod
    def load_csv(path: Union[str, Path]) -> ESRSpectrum:
        """Load a spectrum from a CSV file.

        The loader automatically detects whether values are separated by
        commas or semicolons.  Only the first two columns are used and they are
        renamed to ``"BField [mT]"`` for the magnetic field and
        ``"MW_Absorption []"`` for the intensity.

        Parameters
        ----------
        path:
            Path to the CSV file containing at least two columns: field and
            intensity.
        """

        path = Path(path)

        # ``sep=None`` together with the python engine lets pandas sniff the
        # delimiter, allowing us to support both comma and semicolon separated
        # data.  This is important because some spectrometers export data using
        # semicolons which previously resulted in a single-column DataFrame and
        # downstream ``IndexError`` when accessing the intensity column.
        df = pd.read_csv(path, sep=None, engine="python")

        # Ensure we have at least two columns for field and intensity.
        if df.shape[1] < 2:
            raise ValueError(
                "CSV file must contain at least two columns for field and intensity"
            )

        # Use only the first two columns and give them standard names.
        df = df.iloc[:, :2].copy()
        df.columns = ["BField [mT]", "MW_Absorption []"]

        return ESRSpectrum.from_dataframe(df)
