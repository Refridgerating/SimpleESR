# SimpleESR

A simple program for analyzing and visualizing **Electron Spin Resonance (ESR)** data.  
This project is at an early stage — we’re starting from the basics and will expand features over time.

## Features (so far)
- Load ESR data files (e.g., Bruker CSV export).
- Display raw spectra.
- Basic plotting and inspection tools.
- Simple GUI for selecting a data file to plot.
- Interactive span selection to measure peak position and FWHM.

## Requirements
- Python 3.10+
- Common scientific libraries:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

Run from the command line:

```bash
python -m esr_lab.app data/example_esr.csv
```

Or launch the GUI to choose a file interactively:

```bash
python -m esr_lab.gui
```

After the spectrum is shown, drag the mouse over a region that contains the
positive and negative peaks of an absorption line. The program determines both
extrema and reports their positions and the full width at half maximum (FWHM)
given by the distance between them. Repeat the selection for additional
absorptions.

## Project Goals

Keep it simple at first.

Add features step by step (fitting, peak detection, g-factor extraction, etc.).

Maintain clean and easy-to-use code.

Contributing

Right now this project is in its early development stage. Contributions and ideas are welcome — just keep it simple and incremental.
