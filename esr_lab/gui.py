"""Simple GUI for selecting ESR data files and visualizing spectra."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from .io import ESRLoader
from .plotter import ESRPlotter


def main() -> None:
    """Launch a file selection dialog and plot the chosen ESR spectrum."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select ESR CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )

    if not file_path:
        return

    try:
        spectrum = ESRLoader.load_csv(Path(file_path))
        plotter = ESRPlotter()
        plotter.plot(spectrum)
    except Exception as exc:  # pragma: no cover - GUI error handling
        messagebox.showerror("Error", str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
