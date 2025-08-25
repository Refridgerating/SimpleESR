"""Command line application for visualizing ESR spectra."""

import argparse
from pathlib import Path

from .io import ESRLoader
from .plotter import ESRPlotter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ESR spectra from CSV files"
    )
    parser.add_argument("file", type=Path, help="Path to a CSV ESR file")
    args = parser.parse_args()

    spectrum = ESRLoader.load_csv(args.file)
    plotter = ESRPlotter()
    plotter.plot(spectrum)


if __name__ == "__main__":  # pragma: no cover
    main()
