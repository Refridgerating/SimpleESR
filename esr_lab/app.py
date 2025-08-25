"""Command line application for visualizing ESR spectra."""

import argparse
from pathlib import Path

from .io import ESRLoader
from .plotter import ESRPlotter


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ESR spectra from CSV files")
    parser.add_argument("file", type=Path, help="Path to a CSV ESR file")
    parser.add_argument(
        "--save",
        type=Path,
        help="Path to save the generated plot instead of displaying it",
    )
    args = parser.parse_args()

    spectrum = ESRLoader.load_csv(args.file)
    plotter = ESRPlotter()
    plotter.plot(spectrum, show=args.save is None, save=str(args.save) if args.save else None)


if __name__ == "__main__":  # pragma: no cover
    main()
