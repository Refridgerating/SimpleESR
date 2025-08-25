import numpy as np
import pandas as pd

from esr_lab.spectrum import ESRSpectrum


def test_from_dataframe():
    df = pd.DataFrame({
        "field": [1.0, 2.0, 3.0],
        "intensity": [0.1, 0.2, 0.3],
        "extra": [7, 8, 9],
    })

    spectrum = ESRSpectrum.from_dataframe(df)

    assert np.array_equal(spectrum.field, df.iloc[:, 0].to_numpy())
    assert np.array_equal(spectrum.intensity, df.iloc[:, 1].to_numpy())
