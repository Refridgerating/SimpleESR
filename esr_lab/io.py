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

        Parameters
        ----------
        path:
            Path to the CSV file containing at least two columns: field and
            intensity.
        """

        path = Path(path)
        df = pd.read_csv(path)
        return ESRSpectrum.from_dataframe(df)
