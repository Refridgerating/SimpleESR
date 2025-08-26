"""Data structures for ESR spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ESRSpectrum:
    """Represents a single ESR spectrum.

    Parameters
    ----------
    field:
        Magnetic field values.
    intensity:
        Recorded intensity corresponding to the field values.
    metadata:
        Optional dictionary with acquisition metadata.
    """

    field: np.ndarray
    intensity: np.ndarray
    metadata: Optional[Dict[str, Any]] = field(default=None)

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> "ESRSpectrum":
        """Create a spectrum from a pandas DataFrame.

        The DataFrame is expected to have at least two columns where the
        first column contains the magnetic field values and the second column
        the corresponding intensity.

        Parameters
        ----------
        df:
            Data frame with field values in the first column and intensity in
            the second column.
        metadata:
            Optional metadata dictionary to attach to the resulting spectrum.
        """

        field_vals = df.iloc[:, 0].to_numpy()
        intensity_vals = df.iloc[:, 1].to_numpy()
        return cls(field=field_vals, intensity=intensity_vals, metadata=metadata)
