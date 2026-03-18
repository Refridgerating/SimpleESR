# GUI2 Architecture And Next Steps

## Goal
Build a modern, fast, locally deployable ESR desktop app that supports:
- Live plotting of large traces.
- Responsive analysis/fitting with async workers.
- Clear table-driven outputs for lab decisions.
- Offline `.exe` deployment for local lab machines.

## Stack
- UI shell: `PySide6`
- Plot engine: `pyqtgraph`
- Compute: existing `esr_lab.services.pipeline` + `esr_lab.core`
- Data IO: existing `esr_lab.io`
- Packaging: `pyside6-deploy` (later phase)

## Current Scaffold
- Entrypoint: `esr_lab/gui2/app.py`
- Main window: `esr_lab/gui2/main_window.py`
- Plot widget: `esr_lab/gui2/plot_widget.py`
- Async worker: `esr_lab/gui2/workers.py`
- View model/state: `esr_lab/gui2/viewmodels.py`, `esr_lab/gui2/types.py`
- Table models: `esr_lab/gui2/table_models.py`
- Theme: `esr_lab/gui2/theme.py`

## Async Worker Plan
Run these off the UI thread:
1. Pipeline analysis (`Analyze Active`, `Analyze All`)
2. Data loading for large multi-file imports
3. Export jobs (CSV/image/report)
4. Optional streaming acquisition reader

Keep these on UI thread:
1. Widget updates
2. Table model resets
3. User interaction and control state changes

## Migration Phases
1. Parallel-run scaffold
- Keep `esr_lab/gui.py` as stable fallback.
- Build equivalent actions in `gui2` via `services.pipeline`.

2. Feature parity
- Add peak markers and fit overlays.
- Add trace selector and visibility toggles.
- Add baseline and integration actions.

3. Lab UX polish
- Presets (`Live`, `Batch`, `Compare`).
- Better annotation styles and axis formatting.
- Persist settings with `QSettings`.

4. Packaging
- Add deploy config and build script.
- Produce signed lab-distributable `.exe`.

## Immediate Next Tasks
1. Add trace list panel with active-trace switching in `MainWindow`.
2. Add file-load worker for non-blocking multi-file import.
3. Render fit overlays from pipeline payload.
4. Add export action using existing CSV schema plus uncertainty columns.
5. Add GUI2 smoke tests (headless + view model contract tests).

## Recent UX Additions
- "Analyze All" now opens an analysis wizard before launching the pipeline. Users can set the
  expected number of extrema, choose the peak-finding strategy, and decide which critical
  outputs (Delta H\_pp, FWHM) should be displayed. The wizard writes selections back to
  `QSettings` so batch runs share consistent defaults.
- The analysis panel exposes "Critical points" toggles that hide/show Delta H\_pp or FWHM rows
  and simultaneously add or remove the corresponding markers on the live plot. This makes peak
  verification explicit after each analysis pass.
