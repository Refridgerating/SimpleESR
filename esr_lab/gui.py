"""Simple GUI utilities for visualizing and analysing ESR spectra.

This module now provides an interactive plot accompanied by an analysis panel.
Users can press a button to activate two span selections that determine the
positive and negative extrema of an absorption line.  The x and y values of the
identified peaks together with the calculated full width at half maximum (FWHM)
are shown in a small table on the right hand side of the window.
"""

from __future__ import annotations

from pathlib import Path
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser, simpledialog

# Attempt to import ``ttkbootstrap`` for modern themed widgets.  The optional
# dependency provides nicer looking controls with rounded corners.  Importing it
# here is safe even when the library is not installed; we simply fall back to
# the standard ttk widgets in that case.
try:  # pragma: no cover - optional dependency
    import ttkbootstrap  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    ttkbootstrap = None
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from typing import Callable
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
import sympy as sp
import copy

from .analysis import (
    calc_fwhm,
    find_peak,
    fit_lorentzian_derivative,
    fit_lorentzian_absorption,
    calc_peak_to_peak,
    peak_finder as auto_peak_finder,
    baseline_correct,
    calc_g,
    calc_lorentzian_area,
    FUNCTION_DETAILS,
)
from .io import ESRLoader
from .plotter import plot_residuals
from .spectrum import ESRSpectrum


def _filter_ticks(ticks: list[float], lower: float, upper: float) -> list[float]:
    """Return only tick locations within the provided axis limits."""

    return [t for t in ticks if lower <= t <= upper]


class NavigationToolbarNoSubplots(NavigationToolbar2Tk):
    """Tk toolbar without the subplot configuration tool.

    The default Matplotlib toolbar includes a *Configure Subplots* button which
    launches a dialog with sliders for adjusting subplot parameters.  The
    application does not rely on this functionality and its presence spawns an
    unnecessary window.  A small subclass of ``NavigationToolbar2Tk`` removes the
    corresponding tool so that users are left with the standard navigation
    controls (home, pan, zoom, save) only.  The toolbar additionally knows about
    the currently selected spectrum through a ``get_active_index`` callback so
    that line editing operates on the user‑chosen graph.
    """

    # Filter out the "Subplots" entry from the base class' tool items and add
    # custom tools for editing axes and updating the legend.  This allows users
    # to tweak axis limits and refresh legend entries directly from the embedded
    # toolbar.
    toolitems = [item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots"]

    # Insert the edit tool and the legend updater just before the standard
    # "Save" action.  The ``qt4_editor_options`` icon is shipped with Matplotlib
    # and provides a sensible default even for the Tk backend.
    _names = [item[0] for item in toolitems if item]
    if "Save" in _names:
        idx = _names.index("Save")
        toolitems.insert(
            idx,
            ("Edit", "Edit axis, curve and image parameters", "qt4_editor_options", "edit_parameters"),
        )
        toolitems.insert(
            idx + 1,
            ("Legend", "Update legend to reflect line styles", "qt4_editor_options", "update_legend"),
        )

    def __init__(
        self,
        canvas,
        window,
        get_active_index: Callable[[], int] | None = None,
        update_legend: Callable[[], None] | None = None,
        set_label: Callable[[int, str], None] | None = None,
        get_trace_line: Callable[[int], Line2D | None] | None = None,
        get_canvas_size: Callable[[], tuple[int, int]] | None = None,
        set_canvas_size: Callable[[int, int], None] | None = None,
        get_theme_palette: Callable[[], dict[str, str]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(canvas, window, **kwargs)
        self.get_active_index = get_active_index or (lambda: 0)
        self.update_legend_callback = update_legend
        self.set_label_callback = set_label
        self.get_trace_line = get_trace_line
        self._get_canvas_size_cb = get_canvas_size
        self._set_canvas_size_cb = set_canvas_size
        self._get_theme_palette = get_theme_palette or (lambda: {})
        # Keep a reference to the edit dialog so it isn't GC'd and can be reused
        self._edit_dialog: tk.Toplevel | None = None

    def update_theme(self, palette: dict[str, str]) -> None:
        button_bg = palette.get('button_bg', '#4a90e2')
        button_fg = palette.get('button_fg', '#ffffff')
        button_active = palette.get('button_active_bg', button_bg)
        toolbar_bg = palette.get('toolbar_bg', button_bg)
        toolbar_fg = palette.get('text', '#1e1e1e')
        try:
            self.configure(background=toolbar_bg)
        except Exception:
            pass
        try:
            style = ttk.Style(self)
            style.configure('Toolbar.TButton', background=button_bg, foreground=button_fg)
            style.map('Toolbar.TButton', background=[('pressed', button_active), ('active', button_active), ('!disabled', button_bg)], foreground=[('disabled', '#9a9a9a'), ('!disabled', button_fg)])
            style.configure('Toolbar.TCheckbutton', background=toolbar_bg, foreground=button_fg)
            style.map('Toolbar.TCheckbutton', background=[('selected', button_active), ('active', button_active), ('!disabled', toolbar_bg)], foreground=[('disabled', '#9a9a9a'), ('!disabled', button_fg)])
        except Exception:
            style = None

        def _style(widget: tk.Widget) -> None:
            try:
                if isinstance(widget, tk.Button):
                    widget.configure(
                        bg=button_bg,
                        fg=button_fg,
                        activebackground=button_active,
                        activeforeground=button_fg,
                        highlightbackground=palette.get('accent', button_bg),
                    )
                elif isinstance(widget, tk.Checkbutton):
                    widget.configure(
                        bg=toolbar_bg,
                        fg=toolbar_fg,
                        selectcolor=toolbar_bg,
                        activebackground=toolbar_bg,
                        activeforeground=toolbar_fg,
                        highlightbackground=palette.get('accent', button_bg),
                    )
                elif isinstance(widget, ttk.Button):
                    try:
                        widget.configure(style='Toolbar.TButton')
                    except Exception:
                        pass
                elif isinstance(widget, ttk.Checkbutton):
                    try:
                        widget.configure(style='Toolbar.TCheckbutton')
                    except Exception:
                        pass
                elif isinstance(widget, tk.Label):
                    widget.configure(bg=toolbar_bg, fg=toolbar_fg)
                elif isinstance(widget, (tk.Frame, tk.Canvas)):
                    widget.configure(bg=toolbar_bg)
                else:
                    widget.configure(bg=toolbar_bg)
            except Exception:
                pass
            for sub in getattr(widget, 'winfo_children', lambda: [])():
                _style(sub)

        _style(self)


    def update_legend(self) -> None:
        if self.update_legend_callback:
            self.update_legend_callback()

    def _get_selected_line(self, ax):
        """Return the Line2D corresponding to the active trace.

        Prefer an explicit callback from the embedding app. Fallback to
        selecting among lines tagged as trace via ``gid=='trace'``. As a last
        resort, use the positional index in ``ax.lines``.
        """

        idx = self.get_active_index()
        # Prefer explicit callback if provided
        if callable(self.get_trace_line):
            try:
                line = self.get_trace_line(idx)
                if isinstance(line, Line2D):
                    return line
            except Exception:
                pass

        # Fallback: consider only lines marked as traces
        trace_lines = [ln for ln in getattr(ax, "lines", []) if getattr(ln, "get_gid", lambda: None)() == "trace"]
        if trace_lines and 0 <= idx < len(trace_lines):
            return trace_lines[idx]

        # Last resort: positional access into ax.lines
        if ax.lines and 0 <= idx < len(ax.lines):
            return ax.lines[idx]
        return ax.lines[0] if getattr(ax, "lines", None) else None

    def edit_parameters(self) -> None:
        """Open a small Tk dialog to edit basic plot parameters.

        The dialog exposes a subset of Matplotlib's axis and line configuration
        options.  Users can modify axis limits, labels, titles, tick locations,
        scales and line width.  Invalid inputs are ignored in order to keep the
        interface straightforward and robust.
        """

        figure = self.canvas.figure
        axes = figure.axes
        if not axes:
            messagebox.showwarning("Edit Plot", "No axes to configure")
            return

        ax = axes[0]
        dpi = float(getattr(figure, 'dpi', 72.0))
        palette = {}
        if callable(getattr(self, '_get_theme_palette', None)):
            try:
                palette = self._get_theme_palette() or {}
            except Exception:
                palette = {}
        defaults = {
            'panel_bg': '#f4f4f4',
            'text': '#1e1e1e',
            'entry_bg': '#ffffff',
            'button_bg': '#4a90e2',
            'button_fg': '#ffffff',
            'button_active_bg': '#357ab7',
        }
        palette = {**defaults, **palette}

        # Reuse the existing dialog if it is still open
        try:
            if self._edit_dialog is not None and int(self._edit_dialog.winfo_exists()):
                try:
                    self._edit_dialog.lift()
                    self._edit_dialog.focus_set()
                except Exception:
                    pass
                return
        except Exception:
            self._edit_dialog = None

        # Create a new dialog; parent it to the top-level window hosting the canvas
        try:
            master = self.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            master = self
        dialog = tk.Toplevel(master)
        dialog.title("Edit Plot")
        try:
            dialog.configure(bg=palette['panel_bg'])
        except Exception:
            pass

        # Track dialog so it survives while open
        self._edit_dialog = dialog

        def _on_close() -> None:
            try:
                dialog.destroy()
            finally:
                self._edit_dialog = None

            
        try:
            dialog.protocol("WM_DELETE_WINDOW", _on_close)
        except Exception:
            pass

        # Helper to create a labelled entry
        def add_entry(row: int, label: str, initial: str) -> tk.Entry:
            lbl = tk.Label(dialog, text=label)
            lbl.grid(row=row, column=0, sticky="e")
            try:
                lbl.configure(bg=palette['panel_bg'], fg=palette['text'])
            except Exception:
                pass
            entry = tk.Entry(dialog)
            entry.insert(0, initial)
            try:
                entry.configure(bg=palette['entry_bg'], fg=palette['text'], insertbackground=palette['text'])
            except Exception:
                pass
            entry.grid(row=row, column=1, padx=5, pady=2)
            return entry

        def _theme_dialog_widget(widget: tk.Widget) -> None:
            for child in getattr(widget, 'winfo_children', lambda: [])():
                if getattr(child, '_theme_exempt', False):
                    continue
                try:
                    if isinstance(child, tk.Frame):
                        child.configure(bg=palette['panel_bg'])
                    elif isinstance(child, tk.Label):
                        child.configure(bg=palette['panel_bg'], fg=palette['text'])
                    elif isinstance(child, tk.Entry):
                        child.configure(bg=palette['entry_bg'], fg=palette['text'], insertbackground=palette['text'])
                    elif isinstance(child, tk.Button):
                        child.configure(bg=palette['button_bg'], fg=palette['button_fg'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
                    elif isinstance(child, tk.Checkbutton):
                        child.configure(bg=palette['panel_bg'], fg=palette['text'], selectcolor=palette['panel_bg'], activebackground=palette['panel_bg'], activeforeground=palette['text'])
                    elif isinstance(child, tk.Radiobutton):
                        child.configure(bg=palette['panel_bg'], fg=palette['text'], selectcolor=palette['panel_bg'], activebackground=palette['panel_bg'], activeforeground=palette['text'])
                    elif isinstance(child, tk.OptionMenu):
                        child.configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
                        try:
                            child['menu'].configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
                        except Exception:
                            pass
                except Exception:
                    pass
                _theme_dialog_widget(child)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        title_ent = add_entry(0, "Title", ax.get_title())
        xlabel_ent = add_entry(1, "X label", ax.get_xlabel())
        ylabel_ent = add_entry(2, "Y label", ax.get_ylabel())
        xmin_ent = add_entry(3, "X min", f"{xmin}")
        xmax_ent = add_entry(4, "X max", f"{xmax}")
        ymin_ent = add_entry(5, "Y min", f"{ymin}")
        ymax_ent = add_entry(6, "Y max", f"{ymax}")

        # Gather editable trace lines and allow user to choose which to edit
        trace_lines = [
            ln for ln in getattr(ax, "lines", []) if getattr(ln, "get_gid", lambda: None)() == "trace"
        ]
        display_names: list[str] = []
        for i, ln in enumerate(trace_lines, start=1):
            try:
                nm = ln.get_label()
            except Exception:
                nm = f"Trace {i}"
            display_names.append(f"{i}: {nm}")

        sel_idx = 0
        try:
            active = max(0, int(self.get_active_index()))
            if trace_lines:
                sel_idx = min(active, len(trace_lines) - 1)
        except Exception:
            sel_idx = 0

        trace_label = tk.Label(dialog, text="Trace")
        trace_label.grid(row=7, column=0, sticky="e")
        try:
            trace_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        trace_var = tk.StringVar(value=(display_names[sel_idx] if display_names else ""))
        try:
            trace_choice = ttk.Combobox(dialog, values=display_names, textvariable=trace_var, state="readonly")
            trace_choice.grid(row=7, column=1, padx=5, pady=2, sticky="w")
            try:
                combo_style = ttk.Style(dialog)
                combo_style.configure('PlotEditor.TCombobox', fieldbackground=palette['entry_bg'], foreground=palette['text'], background=palette['entry_bg'])
                trace_choice.configure(style='PlotEditor.TCombobox')
            except Exception:
                pass
        except Exception:
            trace_choice = tk.OptionMenu(dialog, trace_var, *(display_names or [""]))
            trace_choice.grid(row=7, column=1, padx=5, pady=2, sticky="w")
            try:
                trace_choice.configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
                trace_choice['menu'].configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            except Exception:
                pass

        def _selected_index() -> int:
            name = trace_var.get()
            try:
                # Names are formatted as "<i>: <label>"
                return max(0, min(len(display_names) - 1, int(name.split(":", 1)[0]) - 1))
            except Exception:
                return 0

        line = trace_lines[sel_idx] if trace_lines else None
        lw_init = line.get_linewidth() if line is not None else 1.0
        lw_ent = add_entry(8, "Line width", f"{lw_init}")

        # Prepare a Tk-compatible initial color value.
        def _to_hex_safe(c: object) -> str:
            try:
                rgba = mcolors.to_rgba(c)  # accepts many MPL color specs (incl. 'C0')
                return mcolors.to_hex(rgba)
            except Exception:
                return ""

        color_init = _to_hex_safe(line.get_color()) if line is not None else ""
        color_label = tk.Label(dialog, text="Line color")
        color_label.grid(row=9, column=0, sticky="e")
        try:
            color_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        color_frame = tk.Frame(dialog)
        try:
            color_frame.configure(bg=palette['panel_bg'])
        except Exception:
            pass
        color_frame.grid(row=9, column=1, padx=5, pady=2, sticky="w")
        color_ent = tk.Entry(color_frame)
        color_ent.insert(0, color_init)
        color_ent.grid(row=0, column=0)
        color_ent._theme_exempt = True

        # Some color strings are not valid Tk colors; guard preview creation.
        try:
            preview = tk.Canvas(color_frame, width=20, height=20, bg=color_init, highlightthickness=0)
        except Exception:
            preview = tk.Canvas(color_frame, width=20, height=20, highlightthickness=0)
        preview.grid(row=0, column=1, padx=5)
        preview._theme_exempt = True

        def choose_color() -> None:
            """Open a color chooser and update the preview on success."""
            color = None
            try:
                color = colorchooser.askcolor(
                    initialcolor=color_ent.get(), parent=dialog, title="Choose Line Color"
                )[1]
            except tk.TclError:
                try:
                    color = colorchooser.askcolor(parent=dialog)[1]
                except tk.TclError:
                    color = None
            if color:
                color_ent.delete(0, tk.END)
                color_ent.insert(0, color)
                preview.config(bg=color)

        try:
            pick_button = ttk.Button(
                color_frame,
                text="Pick",
                command=choose_color,
                style="Compact.TButton",
            )
            pick_button.grid(row=0, column=2, padx=5)
        except Exception:
            pick_button = tk.Button(color_frame, text="Pick", command=choose_color)
            pick_button.grid(row=0, column=2, padx=5)
            try:
                pick_button.configure(bg=palette['button_bg'], fg=palette['button_fg'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            except Exception:
                pass

        def _update_preview(*_args: object) -> None:
            color_val = color_ent.get().strip()
            try:
                # Accept raw hex or known color names; ignore invalid ones
                preview.config(bg=color_val)
            except tk.TclError:
                try:
                    preview.config(bg=_to_hex_safe(color_val))
                except Exception:
                    pass

        color_ent.bind("<KeyRelease>", _update_preview)

        legend_init = line.get_label() if line is not None else ""
        legend_ent = add_entry(10, "Trace label", legend_init)

        marker_label = tk.Label(dialog, text="Marker")
        marker_label.grid(row=11, column=0, sticky="e")
        try:
            marker_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        marker_init = line.get_marker() if line is not None else "None"
        marker_var = tk.StringVar(value=marker_init)
        markers = ["None", "o", "s", "^", "D", "*", "x", "+"]
        # Ensure the option menu honours the initial marker selection
        marker_widget = tk.OptionMenu(dialog, marker_var, marker_var.get(), *markers)
        marker_widget.grid(row=11, column=1, sticky="w")
        try:
            marker_widget.configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            marker_widget['menu'].configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
        except Exception:
            pass

        # Scale selection
        xscale_label = tk.Label(dialog, text="X scale")
        xscale_label.grid(row=12, column=0, sticky="e")
        try:
            xscale_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        xscale_var = tk.StringVar(value=ax.get_xscale())
        xscale_widget = tk.OptionMenu(dialog, xscale_var, "linear", "log")
        xscale_widget.grid(row=12, column=1, sticky="w")
        try:
            xscale_widget.configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            xscale_widget['menu'].configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
        except Exception:
            pass

        yscale_label = tk.Label(dialog, text="Y scale")
        yscale_label.grid(row=13, column=0, sticky="e")
        try:
            yscale_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        yscale_var = tk.StringVar(value=ax.get_yscale())
        yscale_widget = tk.OptionMenu(dialog, yscale_var, "linear", "log")
        yscale_widget.grid(row=13, column=1, sticky="w")
        try:
            yscale_widget.configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            yscale_widget['menu'].configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
        except Exception:
            pass

        xticks_ent = add_entry(14, "X ticks", ", ".join(map(str, ax.get_xticks())))
        yticks_ent = add_entry(15, "Y ticks", ", ".join(map(str, ax.get_yticks())))

        major_var = tk.BooleanVar(value=ax.xaxis._major_tick_kw.get("gridOn", False))
        major_cb = tk.Checkbutton(dialog, text="Major grid", variable=major_var)
        major_cb.grid(row=16, column=0, columnspan=2, sticky="w")
        minor_var = tk.BooleanVar(value=ax.xaxis._minor_tick_kw.get("gridOn", False))
        minor_cb = tk.Checkbutton(dialog, text="Minor grid", variable=minor_var)
        minor_cb.grid(row=17, column=0, columnspan=2, sticky="w")
        try:
            for cb in (major_cb, minor_cb):
                cb.configure(bg=palette['panel_bg'], fg=palette['text'], selectcolor=palette['panel_bg'], activebackground=palette['panel_bg'], activeforeground=palette['text'])
        except Exception:
            pass

        unit_map = {"px": 1.0, "in": dpi, "cm": dpi / 2.54, "mm": dpi / 25.4, "pt": dpi / 72.0}
        size_units = ("px", "in", "cm", "mm", "pt")
        if callable(getattr(self, "_get_canvas_size_cb", None)):
            try:
                width_px, height_px = self._get_canvas_size_cb()  # type: ignore[call-arg]
            except Exception:
                width_px, height_px = figure.get_figwidth() * dpi, figure.get_figheight() * dpi
        else:
            widget = self.canvas.get_tk_widget()
            width_px = widget.winfo_width()
            height_px = widget.winfo_height()
            if width_px <= 1 or height_px <= 1:
                width_px = figure.get_figwidth() * dpi
                height_px = figure.get_figheight() * dpi
        size_px = [max(1, int(round(width_px))), max(1, int(round(height_px)))]
        size_unit_var = tk.StringVar(value="px")
        width_var = tk.StringVar()
        height_var = tk.StringVar()

        def _refresh_size_entries(*_args: object) -> None:
            unit = size_unit_var.get() or "px"
            factor = unit_map.get(unit, 1.0) or 1.0
            width_val = size_px[0] / factor
            height_val = size_px[1] / factor
            if unit == "px":
                width_var.set(f"{int(round(width_val))}")
                height_var.set(f"{int(round(height_val))}")
            else:
                width_var.set(f"{width_val:.2f}")
                height_var.set(f"{height_val:.2f}")

        width_label = tk.Label(dialog, text="Figure width")
        width_label.grid(row=18, column=0, sticky="e")
        try:
            width_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        width_ent = tk.Entry(dialog, textvariable=width_var)
        try:
            width_ent.configure(bg=palette['entry_bg'], fg=palette['text'], insertbackground=palette['text'])
        except Exception:
            pass
        width_ent.grid(row=18, column=1, padx=5, pady=2, sticky="w")
        height_label = tk.Label(dialog, text="Figure height")
        height_label.grid(row=19, column=0, sticky="e")
        try:
            height_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        height_ent = tk.Entry(dialog, textvariable=height_var)
        try:
            height_ent.configure(bg=palette['entry_bg'], fg=palette['text'], insertbackground=palette['text'])
        except Exception:
            pass
        height_ent.grid(row=19, column=1, padx=5, pady=2, sticky="w")
        units_label = tk.Label(dialog, text="Units")
        units_label.grid(row=20, column=0, sticky="e")
        try:
            units_label.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        try:
            size_unit_widget = ttk.Combobox(
                dialog, values=size_units, textvariable=size_unit_var, state="readonly"
            )
            size_unit_widget.grid(row=20, column=1, padx=5, pady=2, sticky="w")
            try:
                combo_style = ttk.Style(dialog)
                combo_style.configure('PlotEditor.TCombobox', fieldbackground=palette['entry_bg'], foreground=palette['text'], background=palette['entry_bg'])
                size_unit_widget.configure(style='PlotEditor.TCombobox')
            except Exception:
                pass
        except Exception:
            size_unit_widget = tk.OptionMenu(dialog, size_unit_var, size_unit_var.get(), *size_units)
            size_unit_widget.grid(row=20, column=1, padx=5, pady=2, sticky="w")
            try:
                size_unit_widget.configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
                size_unit_widget['menu'].configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            except Exception:
                pass

        try:
            size_unit_var.trace_add("write", _refresh_size_entries)  # type: ignore[attr-defined]
        except Exception:
            try:
                size_unit_var.trace("w", _refresh_size_entries)  # type: ignore[misc]
            except Exception:
                pass
        _refresh_size_entries()

        def _refresh_line_fields(*_args: object) -> None:
            idx = _selected_index()
            ln = trace_lines[idx] if 0 <= idx < len(trace_lines) else None
            if ln is None:
                return
            try:
                lw_ent.delete(0, tk.END)
                lw_ent.insert(0, f"{ln.get_linewidth()}")
            except Exception:
                pass
            try:
                cval = _to_hex_safe(ln.get_color())
                color_ent.delete(0, tk.END)
                color_ent.insert(0, cval)
                preview.config(bg=cval)
            except Exception:
                pass
            try:
                legend_ent.delete(0, tk.END)
                legend_ent.insert(0, ln.get_label())
            except Exception:
                pass
            try:
                current_marker = ln.get_marker() or "None"
                if current_marker == "":
                    current_marker = "None"
                marker_var.set(current_marker)
            except Exception:
                pass

        # Update line-specific fields when the selected trace changes
        try:
            trace_var.trace_add("write", _refresh_line_fields)  # type: ignore[attr-defined]
        except Exception:
            try:
                trace_var.trace("w", _refresh_line_fields)  # type: ignore[misc]
            except Exception:
                pass

        def apply() -> None:
            try:
                xmin_val = float(xmin_ent.get())
                xmax_val = float(xmax_ent.get())
                ax.set_xlim(xmin_val, xmax_val)
                xmin_val, xmax_val = ax.get_xlim()
            except ValueError:
                xmin_val, xmax_val = ax.get_xlim()
            try:
                ymin_val = float(ymin_ent.get())
                ymax_val = float(ymax_ent.get())
                ax.set_ylim(ymin_val, ymax_val)
                ymin_val, ymax_val = ax.get_ylim()
            except ValueError:
                ymin_val, ymax_val = ax.get_ylim()

            ax.set_title(title_ent.get())
            ax.set_xlabel(xlabel_ent.get())
            ax.set_ylabel(ylabel_ent.get())

            resize_applied = False
            try:
                width_input = float(width_ent.get())
                height_input = float(height_ent.get())
                unit = size_unit_var.get() or "px"
                factor = unit_map.get(unit, 1.0) or 1.0
                width_px = max(1, int(round(width_input * factor)))
                height_px = max(1, int(round(height_input * factor)))
                size_px[0], size_px[1] = width_px, height_px
                if callable(getattr(self, '_set_canvas_size_cb', None)):
                    try:
                        self._set_canvas_size_cb(width_px, height_px)  # type: ignore[call-arg]
                    except Exception:
                        fig = self.canvas.figure
                        widget = self.canvas.get_tk_widget()
                        fig.set_size_inches(width_px / dpi, height_px / dpi, forward=True)
                        widget.configure(width=width_px, height=height_px)
                        self.canvas.draw_idle()
                else:
                    fig = self.canvas.figure
                    widget = self.canvas.get_tk_widget()
                    fig.set_size_inches(width_px / dpi, height_px / dpi, forward=True)
                    widget.configure(width=width_px, height=height_px)
                    self.canvas.draw_idle()
                resize_applied = True
            except ValueError:
                pass

            # Apply line-specific edits to the selected trace
            try:
                idx = _selected_index()
            except Exception:
                idx = 0
            line = trace_lines[idx] if 0 <= idx < len(trace_lines) else None
            if line is not None:
                try:
                    lw = float(lw_ent.get())
                    line.set_linewidth(lw)
                except ValueError:
                    pass

                color_val = color_ent.get().strip()
                if color_val:
                    try:
                        rgba = mcolors.to_rgba(color_val)
                        line.set_color(rgba)
                        # Keep marker colors consistent if markers are used
                        mk = line.get_marker()
                        if mk not in (None, "", "None"):
                            line.set_markerfacecolor(rgba)
                            line.set_markeredgecolor(rgba)
                    except Exception:
                        # Fallback to raw string; if MPL accepts it, fine
                        try:
                            line.set_color(color_val)
                        except Exception:
                            pass

                marker_val = marker_var.get()
                line.set_marker("" if marker_val == "None" else marker_val)

                legend_text = legend_ent.get()
                line.set_label(legend_text)
                if self.set_label_callback is not None:
                    try:
                        self.set_label_callback(idx, legend_text)
                    except Exception:
                        pass
                self.update_legend()

            ax.set_xscale(xscale_var.get())
            ax.set_yscale(yscale_var.get())

            try:
                ticks = [float(v) for v in xticks_ent.get().split(",") if v.strip()]
                ticks = _filter_ticks(ticks, xmin_val, xmax_val)
                ax.set_xticks(ticks)
            except ValueError:
                pass
            try:
                ticks = [float(v) for v in yticks_ent.get().split(",") if v.strip()]
                ticks = _filter_ticks(ticks, ymin_val, ymax_val)
                ax.set_yticks(ticks)
            except ValueError:
                pass

            if resize_applied:
                _refresh_size_entries()

            ax.grid(major_var.get(), which="major")
            if minor_var.get():
                ax.minorticks_on()
            else:
                ax.minorticks_off()
            ax.grid(minor_var.get(), which="minor")

            self.canvas.draw_idle()

        try:
            ttk.Button(dialog, text="Apply", command=apply, style="Compact.TButton").grid(
                row=21, column=0, pady=5
            )
            ttk.Button(
                dialog, text="Close", command=_on_close, style="Compact.TButton"
            ).grid(row=21, column=1, pady=5)
        except Exception:
            tk.Button(dialog, text="Apply", command=apply).grid(row=21, column=0, pady=5)
            tk.Button(dialog, text="Close", command=_on_close).grid(
                row=21, column=1, pady=5
            )

        _theme_dialog_widget(dialog)

    def save_figure(self, *args, **kwargs) -> None:  # type: ignore[override]
        choice = self._prompt_save_style()
        if choice is None:
            return
        if choice == "current":
            super().save_figure(*args, **kwargs)
            return
        fig = self.canvas.figure
        state = self._capture_figure_state(fig)
        try:
            self._apply_export_style(fig, facecolor="white", text_color="black")
            super().save_figure(*args, **kwargs)
        finally:
            self._restore_figure_state(state)
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

    def _prompt_save_style(self) -> str | None:
        widget = self.canvas.get_tk_widget()
        master = widget.winfo_toplevel() if hasattr(widget, "winfo_toplevel") else widget
        dialog = tk.Toplevel(master)
        dialog.title("Save Figure")
        dialog.transient(master)
        dialog.resizable(False, False)
        dialog.grab_set()

        palette = {}
        if callable(getattr(self, '_get_theme_palette', None)):
            try:
                palette = self._get_theme_palette() or {}
            except Exception:
                palette = {}
        defaults = {
            'panel_bg': '#f4f4f4',
            'text': '#1e1e1e',
            'button_bg': '#4a90e2',
            'button_fg': '#ffffff',
            'button_active_bg': '#357ab7',
        }
        palette = {**defaults, **palette}
        try:
            dialog.configure(bg=palette['panel_bg'])
        except Exception:
            pass

        var = tk.StringVar(value="current")
        result: dict[str, str | None] = {"choice": None}

        label_prompt = tk.Label(dialog, text="Export figure using:")
        label_prompt.pack(padx=20, pady=(12, 6), anchor="w")
        try:
            label_prompt.configure(bg=palette['panel_bg'], fg=palette['text'])
        except Exception:
            pass
        for label, value in (("Current theme", "current"), ("White background", "white")):
            rb = tk.Radiobutton(dialog, text=label, value=value, variable=var)
            rb.pack(anchor="w", padx=30)
            try:
                rb.configure(bg=palette['panel_bg'], fg=palette['text'], selectcolor=palette['panel_bg'], activebackground=palette['panel_bg'], activeforeground=palette['text'])
            except Exception:
                pass

        btn_frame = tk.Frame(dialog)
        try:
            btn_frame.configure(bg=palette['panel_bg'])
        except Exception:
            pass
        btn_frame.pack(pady=10)

        def _confirm() -> None:
            result["choice"] = var.get()
            dialog.destroy()

        def _cancel() -> None:
            dialog.destroy()

        save_btn = tk.Button(btn_frame, text="Save", width=10, command=_confirm)
        save_btn.pack(side=tk.LEFT, padx=5)
        cancel_btn = tk.Button(btn_frame, text="Cancel", width=10, command=_cancel)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        for btn in (save_btn, cancel_btn):
            try:
                btn.configure(bg=palette['button_bg'], fg=palette['button_fg'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            except Exception:
                pass
        dialog.protocol("WM_DELETE_WINDOW", _cancel)
        dialog.wait_window()
        return result["choice"]

    def _capture_figure_state(self, fig) -> dict[str, object]:
        axes_state: list[dict[str, object]] = []
        for ax in fig.axes:
            tick_color = None
            try:
                lines = ax.xaxis.get_ticklines()
            except Exception:
                lines = []
            if lines:
                tick_color = lines[0].get_color()
            if not tick_color:
                tick_color = ax.xaxis.label.get_color()
            axes_state.append({
                "ax": ax,
                "facecolor": ax.get_facecolor(),
                "spines": {name: spine.get_edgecolor() for name, spine in ax.spines.items()},
                "x_label": ax.xaxis.label.get_color(),
                "y_label": ax.yaxis.label.get_color(),
                "title": ax.title.get_color() if hasattr(ax.title, "get_color") else ax.title.get_color(),
                "xtick_colors": [lbl.get_color() for lbl in ax.get_xticklabels()],
                "ytick_colors": [lbl.get_color() for lbl in ax.get_yticklabels()],
                "legend": self._capture_legend_state(ax.get_legend()),
                "tick_color": tick_color,
            })
        return {"fig": fig, "facecolor": fig.get_facecolor(), "axes": axes_state}

    def _capture_legend_state(self, legend):
        if legend is None:
            return None
        frame = legend.get_frame()
        return {
            "legend": legend,
            "facecolor": frame.get_facecolor(),
            "edgecolor": frame.get_edgecolor(),
            "text_colors": [txt.get_color() for txt in legend.get_texts()],
        }

    def _apply_export_style(self, fig, facecolor: str, text_color: str) -> None:
        try:
            fig.patch.set_facecolor(facecolor)
        except Exception:
            pass
        for ax in fig.axes:
            ax.set_facecolor(facecolor)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            ax.tick_params(colors=text_color)
            for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                tick.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            legend = ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor(facecolor)
                frame.set_edgecolor(text_color)
                for txt in legend.get_texts():
                    txt.set_color(text_color)

    def _restore_figure_state(self, state: dict[str, object]) -> None:
        fig = state.get("fig")
        if fig is None:
            return
        face = state.get("facecolor")
        try:
            fig.patch.set_facecolor(face)
        except Exception:
            pass
        for ax_state in state.get("axes", []):
            ax = ax_state["ax"]
            ax.set_facecolor(ax_state["facecolor"])
            for name, color in ax_state["spines"].items():
                ax.spines[name].set_color(color)
            tick_color = ax_state.get("tick_color")
            if tick_color:
                ax.tick_params(colors=tick_color)
            ax.xaxis.label.set_color(ax_state["x_label"])
            ax.yaxis.label.set_color(ax_state["y_label"])
            ax.title.set_color(ax_state["title"])
            for tick, color in zip(ax.get_xticklabels(), ax_state["xtick_colors"]):
                tick.set_color(color)
            for tick, color in zip(ax.get_yticklabels(), ax_state["ytick_colors"]):
                tick.set_color(color)
            legend_state = ax_state.get("legend")
            if legend_state:
                legend = legend_state.get("legend")
                if legend is not None:
                    frame = legend.get_frame()
                    frame.set_facecolor(legend_state.get("facecolor"))
                    frame.set_edgecolor(legend_state.get("edgecolor"))
                    for txt, color in zip(legend.get_texts(), legend_state.get("text_colors", [])):
                        txt.set_color(color)


class BaselineOptionsDialog(tk.Toplevel):
    """Dialog for selecting baseline correction options."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("Baseline Options")
        self.resizable(False, False)

        self.fit_type = tk.IntVar(value=1)
        self.use_auto = tk.BooleanVar(value=True)

        tk.Label(self, text="Baseline fit type:").pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        tk.Radiobutton(
            self,
            text="Polynomial fit",
            variable=self.fit_type,
            value=1,
        ).pack(anchor="w", padx=20)
        tk.Radiobutton(
            self,
            text="Linear fit",
            variable=self.fit_type,
            value=0,
        ).pack(anchor="w", padx=20, pady=(0, 5))
        tk.Checkbutton(
            self,
            text="Automatic point placement",
            variable=self.use_auto,
        ).pack(anchor="w", padx=10, pady=5)

        btn = tk.Frame(self)
        btn.pack(pady=5)
        tk.Button(btn, text="OK", command=self._on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn, text="Cancel", command=self._on_cancel).pack(
            side=tk.LEFT, padx=5
        )

        self.result: tuple[bool, bool] | None = None
        self.grab_set()

    def _on_ok(self) -> None:
        self.result = (bool(self.fit_type.get()), self.use_auto.get())
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


class BaselinePointEditor:
    """Interactive helper for selecting and adjusting baseline points.

    Points can be added with a mouse click and subsequently dragged along the
    recorded trace. A provisional baseline derived from the current set of
    points is drawn as a dashed line and updated dynamically as markers move.
    """

    def __init__(
        self,
        ax,
        field: np.ndarray,
        intensity: np.ndarray,
        degree: int = 1,
        on_update: Callable[[], None] | None = None,
    ) -> None:
        self.ax = ax
        self.field = field
        self.intensity = intensity
        self.degree = degree
        self.on_update = on_update
        self.points: list[tuple[float, float]] = []
        self.point_artists: list[Line2D] = []
        self.baseline_line: Line2D | None = None
        self.drag_idx: int | None = None

        canvas = self.ax.figure.canvas
        self.cid_click = canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_release = canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = canvas.mpl_connect("motion_notify_event", self.on_motion)

    def _nearest_point(self, x: float) -> tuple[float, float]:
        idx = int(np.argmin((self.field - x) ** 2))
        return float(self.field[idx]), float(self.intensity[idx])

    def on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        for i, artist in enumerate(self.point_artists):
            contains, _ = artist.contains(event)
            if contains:
                self.drag_idx = i
                return

        x, y = self._nearest_point(event.xdata)
        (artist,) = self.ax.plot(x, y, "ro")
        self.points.append((x, y))
        self.point_artists.append(artist)
        self.update_baseline()
        if self.on_update:
            self.on_update()
        self.ax.figure.canvas.draw_idle()

    def on_motion(self, event) -> None:
        if self.drag_idx is None or event.inaxes != self.ax:
            return
        x, y = self._nearest_point(event.xdata)
        self.points[self.drag_idx] = (x, y)
        self.point_artists[self.drag_idx].set_data([x], [y])
        self.update_baseline()
        if self.on_update:
            self.on_update()
        self.ax.figure.canvas.draw_idle()

    def on_release(self, _event) -> None:
        self.drag_idx = None

    def update_baseline(self) -> None:
        if len(self.points) < 2:
            return
        _corr, baseline = baseline_correct(
            self.field, self.intensity, points=self.points, degree=self.degree
        )
        if self.baseline_line is None:
            (self.baseline_line,) = self.ax.plot(
                self.field, baseline, "k--", linewidth=1
            )
        else:
            self.baseline_line.set_data(self.field, baseline)

    def get_points(self) -> list[tuple[float, float]]:
        return self.points

    def disconnect(self) -> None:
        canvas = self.ax.figure.canvas
        canvas.mpl_disconnect(self.cid_click)
        canvas.mpl_disconnect(self.cid_release)
        canvas.mpl_disconnect(self.cid_motion)

    def clear_artists(self) -> None:
        for artist in self.point_artists:
            artist.remove()
        self.point_artists.clear()
        if self.baseline_line is not None:
            self.baseline_line.remove()
            self.baseline_line = None
        self.ax.figure.canvas.draw_idle()



class SpanPeakSelector:
    """Interactive peak analysis with an optional Tk GUI.

    The class keeps a small backwards-compatible API for the unit tests.  When
    :meth:`show` is invoked a full Tkinter application with an analysis panel is
    created.  Peak positions and the resulting FWHM are listed in a table on the
    side.  Without calling :meth:`show` the class can still be used in a headless
    manner where selections are fed directly to :meth:`onselect` and the results
    are reported via a message box, mirroring the behaviour of previous
    versions.
    """

    _THEMES: dict[str, dict[str, str]] = {
        "light": {
            "bg": "#f4f4f4",
            "panel_bg": "#ffffff",
            "text": "#1e1e1e",
            "accent": "#4a90e2",
            "plot_face": "#ffffff",
            "axes_face": "#ffffff",
            "axes_edge": "#1e1e1e",
            "toolbar_bg": "#e0e0e0",
            "entry_bg": "#ffffff",
            "button_bg": "#4a90e2",
            "button_fg": "#ffffff",
            "button_active_bg": "#357ab7",
        },
        "dark": {
            "bg": "#1e1e1e",
            "panel_bg": "#252526",
            "text": "#f0f0f0",
            "accent": "#9c8cd8",
            "plot_face": "#1e1e1e",
            "axes_face": "#1e1e1e",
            "axes_edge": "#f0f0f0",
            "toolbar_bg": "#2d2d30",
            "entry_bg": "#2d2d30",
            "button_bg": "#b39ddb",
            "button_fg": "#1e1e1e",
            "button_active_bg": "#9f8ad6",
        },
    }


    def _get_theme_palette(self) -> dict[str, str]:
        return dict(self._THEMES.get(self._theme, self._THEMES['light']))

    def __init__(self, spectrum=None, labels: list[str] | None = None) -> None:
        """Initialise the selector with zero, one, or more spectra.

        Parameters
        ----------
        spectrum:
            Either a single :class:`~esr_lab.spectrum.ESRSpectrum` instance or a
            list of spectra to overlay.  The previous API accepted only a single
            spectrum which is still supported for backwards compatibility.
        """

        # Normalise ``spectrum`` so the rest of the implementation can
        # treat all inputs uniformly. Allow starting with no data loaded.
        if spectrum is None:
            self.spectra = []
        elif isinstance(spectrum, list):
            self.spectra = spectrum
        else:
            self.spectra = [spectrum]

        if labels is not None and len(labels) == len(self.spectra):
            self.labels = labels
        else:
            self.labels = [f"Trace {i + 1}" for i in range(len(self.spectra))]
        self.current = 0
        self.spectrum = self.spectra[0] if self.spectra else None

        # Keep analysis results for each loaded spectrum separately.  ``results``
        # and ``lorentz_results`` always refer to the currently selected trace so
        # the public API remains unchanged.
        self.results_all: list[list[dict[str, float | str | int]]] = [
            [] for _ in self.spectra
        ]
        self.lorentz_all: list[list[dict[str, float | str | int]]] = [
            [] for _ in self.spectra
        ]
        self.results = self.results_all[self.current] if self.results_all else []
        self.lorentz_results = (
            self.lorentz_all[self.current] if self.lorentz_all else []
        )

        # Individual span selections per trace
        self.ranges_all: list[list[tuple[float, float]]] = [
            [] for _ in self.spectra
        ]
        self.ranges = self.ranges_all[self.current] if self.ranges_all else []

        # Automatically detected peak indices per trace
        self.auto_peaks_all: list[list[tuple[int, int]]] = [
            [] for _ in self.spectra
        ]
        self.auto_peaks = (
            self.auto_peaks_all[self.current] if self.auto_peaks_all else []
        )

        # Automatically detected absorption peak indices per trace
        self.abs_peaks_all: list[list[int]] = [
            [] for _ in self.spectra
        ]
        self.abs_peaks = (
            self.abs_peaks_all[self.current] if self.abs_peaks_all else []
        )

        # Plot lines and visibility state for each spectrum
        self.trace_lines: list[Line2D] = []
        self.trace_vars: list[tk.BooleanVar] = []

        # GUI related attributes are initialised lazily in ``show`` so that the
        # class can be instantiated in environments without a display (e.g. the
        # test suite).
        self.root: tk.Tk | None = None
        self.ax = None
        self.peak_tree: ttk.Treeview | None = None
        self.tree: ttk.Treeview | None = None
        self.lorentz_tree: ttk.Treeview | None = None
        self.analyze_btn: tk.Button | ttk.Button | None = None
        self.analyse_btn: tk.Button | ttk.Button | None = None
        self.dhpp_btn: tk.Button | ttk.Button | None = None
        self.find_btn: tk.Button | ttk.Button | None = None
        self.find_abs_btn: tk.Button | ttk.Button | None = None
        self.fit_btn: tk.Button | ttk.Button | None = None
        self.integrate_btn: tk.Button | ttk.Button | None = None
        self.baseline_btn: tk.Button | ttk.Button | None = None
        self.compare_btn: tk.Button | ttk.Button | None = None
        self.compare_tree: ttk.Treeview | None = None
        self.batch_tree: ttk.Treeview | None = None
        self.trace_combo: ttk.Combobox | None = None
        self.plot_frame: tk.Frame | None = None
        self.control_frame: tk.Frame | None = None
        self.toggle_frame: tk.Frame | None = None
        self.extra_canvases: list[FigureCanvasTkAgg] = []
        self.trace_var: tk.StringVar | None = None
        self.meta_label: tk.Label | None = None
        self.peak_table_label: tk.Label | None = None
        self.results_table_label: tk.Label | None = None
        self.lorentz_table_label: tk.Label | None = None
        self.compare_table_label: tk.Label | None = None
        self._peak_table_title = "Peak position"
        self._analysis_table_title = "Analysis Results"
        self._lorentz_table_title = "Lorentzian Fits"
        self._compare_table_title = "Comparison"
        self.metadata_text: str = ""
        self._theme: str = "light"
        self._dark_mode_var: tk.BooleanVar | None = None
        self._updating_theme = False
        self.toolbar: NavigationToolbarNoSubplots | None = None
        self.plot_container: tk.Frame | None = None
        self.panel_container: tk.Frame | None = None
        self.panel_frame: tk.Frame | None = None
        self.meta_frame: tk.Frame | None = None
        self.results_frame: tk.Frame | None = None
        self.lorentz_frame: tk.Frame | None = None
        self.compare_frame: tk.Frame | None = None
        self.batch_frame: tk.Frame | None = None
        self._button_rows: list[tk.Frame] = []
        self._button_cls: type[tk.Button] | type[ttk.Button] = ttk.Button if hasattr(ttk, 'Button') else tk.Button
        self._button_kwargs: dict[str, object] = {}
        self.figure_canvas: FigureCanvasTkAgg | None = None
        self.figure_container: tk.Frame | None = None
        self.figure_widget: tk.Widget | None = None
        self._canvas_pixel_size: tuple[int, int] | None = None
        self._resize_handles: list[tk.Widget] = []
        self._resize_state: dict[str, int] | None = None
        self._min_canvas_width = 200
        self._min_canvas_height = 150
        self.delete_btn: tk.Button | ttk.Button | None = None
        self.export_btn: tk.Button | ttk.Button | None = None
        self.g_btn: tk.Button | ttk.Button | None = None
        self.area_btn: tk.Button | ttk.Button | None = None
        self.batch_btn: tk.Button | ttk.Button | None = None
        self.undo_btn: tk.Button | ttk.Button | None = None
        self._history: list[dict[str, object]] = []
        # Keep track of which peak (1 or 2) the user is analysing.
        # Default to the first peak so headless usage remains functional
        # without invoking the interactive prompt.
        self.current_peak: int = 1
        self.selector: SpanSelector | None = None
        self.analysis_func: Callable[[np.ndarray, np.ndarray, int, int], float] = calc_fwhm
        self.analysis_label: str = "FWHM"

    # ------------------------------------------------------------------
    def _prompt_peak(self) -> int | None:
        """Ask the user which peak should be analysed.

        Returns
        -------
        int | None
            ``1`` or ``2`` depending on the user's choice.  ``None`` is
            returned if the dialog is cancelled.  In headless environments
            where the dialog cannot be shown, the function falls back to
            the first peak to keep scripted use working.
        """

        try:
            return simpledialog.askinteger(
                "Select Peak", "Analyse peak 1 or 2?", minvalue=1, maxvalue=2
            )
        except Exception:
            # When running without a display (e.g. during tests) Tk may raise
            # errors.  Defaulting to the first peak keeps the API usable.
            return 1

    # ------------------------------------------------------------------
    def _prompt_traces(self) -> tuple[int, int] | None:
        """Ask the user which two traces should be compared.

        Returns
        -------
        tuple[int, int] | None
            Zero-based indices of the selected traces or ``None`` if the dialog
            is cancelled. In headless environments the first two traces are
            chosen by default when available.
        """

        if len(self.spectra) < 2:
            return None

        try:
            class _TraceDialog(simpledialog.Dialog):
                """Dialog with two drop-downs to choose spectra."""

                def __init__(self, parent, labels: list[str]):
                    self.labels = labels
                    self.first_var = tk.StringVar()
                    self.second_var = tk.StringVar()
                    super().__init__(parent, title="Compare Spectra")

                def body(self, master):  # type: ignore[override]
                    ttk.Label(master, text="First trace:").grid(row=0, column=0, padx=5, pady=5)
                    first = ttk.Combobox(
                        master,
                        values=self.labels,
                        textvariable=self.first_var,
                        state="readonly",
                    )
                    first.grid(row=0, column=1, padx=5, pady=5)
                    if self.labels:
                        first.current(0)
                        self.first_var.set(self.labels[0])

                    ttk.Label(master, text="Second trace:").grid(row=1, column=0, padx=5, pady=5)
                    second = ttk.Combobox(
                        master,
                        values=self.labels,
                        textvariable=self.second_var,
                        state="readonly",
                    )
                    second.grid(row=1, column=1, padx=5, pady=5)
                    if len(self.labels) > 1:
                        second.current(1)
                        self.second_var.set(self.labels[1])
                    elif self.labels:
                        second.current(0)
                        self.second_var.set(self.labels[0])
                    return first

                def apply(self) -> None:  # type: ignore[override]
                    self.result = (
                        self.labels.index(self.first_var.get()),
                        self.labels.index(self.second_var.get()),
                    )

            parent = self.root
            temp_root = None
            if parent is None:
                temp_root = tk.Tk()
                temp_root.withdraw()
                parent = temp_root

            dialog = _TraceDialog(parent, self.labels)
            if temp_root is not None:
                temp_root.destroy()
            return dialog.result
        except Exception:
            if len(self.spectra) >= 2:
                return 0, 1
            return None

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        """Store a deep copy of the current state for undo."""

        state = {
            "spectra": [
                ESRSpectrum(
                    field=s.field.copy(),
                    intensity=s.intensity.copy(),
                    metadata=copy.deepcopy(s.metadata),
                )
                for s in self.spectra
            ],
            "labels": self.labels.copy(),
            "results_all": copy.deepcopy(self.results_all),
            "lorentz_all": copy.deepcopy(self.lorentz_all),
            "ranges_all": copy.deepcopy(self.ranges_all),
            "auto_peaks_all": copy.deepcopy(self.auto_peaks_all),
            "abs_peaks_all": copy.deepcopy(self.abs_peaks_all),
            "current": self.current,
            "visibility": [
                bool(getattr(ln, "get_visible", lambda: True)()) for ln in getattr(self, "trace_lines", [])
            ],
        }
        self._history.append(state)
        if self.undo_btn is not None:
            self.undo_btn.config(state=tk.NORMAL)

    def undo(self) -> None:
        """Revert the last operation if possible."""

        if not self._history:
            return
        state = self._history.pop()
        self.spectra = state["spectra"]
        self.labels = state["labels"]
        self.results_all = state["results_all"]
        self.lorentz_all = state["lorentz_all"]
        self.ranges_all = state["ranges_all"]
        self.auto_peaks_all = state["auto_peaks_all"]
        self.abs_peaks_all = state["abs_peaks_all"]
        self.current = state["current"]
        self.spectrum = self.spectra[self.current]
        self.results = self.results_all[self.current]
        self.lorentz_results = self.lorentz_all[self.current]
        self.ranges = self.ranges_all[self.current]
        self.auto_peaks = self.auto_peaks_all[self.current]
        self.abs_peaks = self.abs_peaks_all[self.current]
        if self.ax is not None:
            self.ax.clear()
            self.trace_lines = []
            for sp, lbl in zip(self.spectra, self.labels):
                (line,) = self.ax.plot(sp.field, sp.intensity, label=lbl)
                try:
                    line.set_gid("trace")
                except Exception:
                    pass
                self.trace_lines.append(line)
            # Restore previous visibility per trace if available
            vis = state.get("visibility", [])
            for i, line in enumerate(self.trace_lines):
                try:
                    line.set_visible(bool(vis[i]))
                except Exception:
                    pass
            self.ax.figure.canvas.draw_idle()
        if self.trace_combo is not None and self.trace_var is not None:
            self.trace_combo["values"] = self.labels
            self.trace_var.set(self.labels[self.current])
        if self.toggle_frame is not None:
            # Preserve existing visibility states for remaining traces
            vis_states = [ln.get_visible() for ln in self.trace_lines]
            for child in self.toggle_frame.winfo_children():
                child.destroy()
            tk.Label(self.toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, (lbl, visible) in enumerate(zip(self.labels, vis_states)):
                var = tk.BooleanVar(value=bool(visible))
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=lbl,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)
        if self.delete_btn is not None:
            state_str = tk.NORMAL if len(self.spectra) > 0 else tk.DISABLED
            self.delete_btn.config(state=state_str)
        if self.undo_btn is not None and not self._history:
            self.undo_btn.config(state=tk.DISABLED)
        self._refresh_tables()
        self._update_metadata_display()
        self.update_legend()
        self._rescale()
        self._update_button_states()

    # ------------------------------------------------------------------
    def start_analysis(
        self,
        analysis_func: Callable[[np.ndarray, np.ndarray, int, int], float] = calc_fwhm,
        label: str = "FWHM",
        auto: bool = False,
    ) -> None:
        """Enable span selection and prepare for analysis.

        Previously this method cleared any existing analysis results each time a
        new analysis was started.  This behaviour made it impossible to perform
        multiple analyses in succession without losing earlier data.  The method
        now preserves ``self.results`` and any existing table entries so that
        users can accumulate measurements across different analyses.
        """
        self._save_state()

        if self.tree is not None and "width" in self.tree["columns"]:
            # The tree keeps previously analysed data; only the analysis label
            # column distinguishes between different result types so the width
            # heading can remain unchanged.
            pass

        self.analysis_func = analysis_func
        self.analysis_label = label

        if auto:
            lines: list[str] = []
            for i, (pos_idx, neg_idx) in enumerate(self.auto_peaks, start=1):
                self.current_peak = i
                width = self.analysis_func(
                    self.spectrum.field, self.spectrum.intensity, pos_idx, neg_idx
                )
                pos_field = self.spectrum.field[pos_idx]
                pos_y = self.spectrum.intensity[pos_idx]
                neg_field = self.spectrum.field[neg_idx]
                neg_y = self.spectrum.intensity[neg_idx]
                result = {
                    "analysis": self.analysis_label,
                    "peak": int(self.current_peak),
                    "pos_x": float(pos_field),
                    "pos_y": float(pos_y),
                    "neg_x": float(neg_field),
                    "neg_y": float(neg_y),
                    "width": float(width),
                }
                self.results.append(result)
                if self.tree is not None:
                    self.tree.insert(
                        "",
                        tk.END,
                        values=(
                            self.analysis_label,
                            f"{self.current_peak}",
                            f"{pos_field:.3f}",
                            f"{pos_y:.3f}",
                            f"{neg_field:.3f}",
                            f"{neg_y:.3f}",
                            f"{width:.3f}",
                        ),
                    )
                lines.append(
                    f"Peak {self.current_peak}: pos={pos_field:.3f}, neg={neg_field:.3f}, {self.analysis_label}={width:.3f}"
                )
            # Only show summary when not running in silent auto mode
            if lines and not getattr(self, "_silent_auto", False):
                messagebox.showinfo("Peak analysis", "\n".join(lines))
            return

        peak_choice = self._prompt_peak()
        if peak_choice is None:
            return
        self.current_peak = int(peak_choice)

        self.ranges.clear()
        if self.selector is not None:
            self.selector.disconnect_events()
        assert self.ax is not None
        self.selector = SpanSelector(
            self.ax, self.onselect, "horizontal", useblit=True
        )
        if self.analyse_btn is not None:
            self.analyse_btn.config(state=tk.DISABLED)
        if self.dhpp_btn is not None:
            self.dhpp_btn.config(state=tk.DISABLED)
        if self.find_btn is not None:
            self.find_btn.config(state=tk.DISABLED)
        if getattr(self, "find_abs_btn", None) is not None:
            self.find_abs_btn.config(state=tk.DISABLED)

    def start_peak_to_peak(self, auto: bool = False) -> None:
        """Start interactive \u0394H_pp analysis using span selection."""

        self.start_analysis(calc_peak_to_peak, "\u0394H_pp", auto=auto)

    def peak_finder(self, auto: bool = False) -> None:
        """Automatically detect peak pairs and store them for analysis.

        Temporary markers are drawn on the plot to aid the user in verifying
        the detected peak positions.  The markers are removed once the user
        decides whether to accept the peaks.
        """

        self._save_state()

        # Always operate on the currently selected trace
        self.spectrum = self.spectra[self.current]
        self.auto_peaks = self.auto_peaks_all[self.current]

        if auto:
            num = 4
        else:
            try:
                num = simpledialog.askinteger(
                    "Peak Finder", "How many peaks to expect?", initialvalue=4, minvalue=2
                )
            except Exception:
                num = 4
            if num is None:
                return
        try:
            pairs = auto_peak_finder(
                self.spectrum.field,
                self.spectrum.intensity,
                expected=int(num),
                method="auto",
            )
        except ValueError as exc:
            messagebox.showerror("Peak Finder", str(exc))
            return

        if not auto:
            markers: list[Line2D] = []
            if self.ax is not None:
                for p, n in pairs:
                    (pos_marker,) = self.ax.plot(
                        self.spectrum.field[p],
                        self.spectrum.intensity[p],
                        marker="o",
                        color="red",
                    )
                    (neg_marker,) = self.ax.plot(
                        self.spectrum.field[n],
                        self.spectrum.intensity[n],
                        marker="o",
                        color="blue",
                    )
                    markers.extend([pos_marker, neg_marker])
                self.ax.figure.canvas.draw_idle()

            lines = [
                (
                    f"Peak {i + 1}: pos={self.spectrum.field[p]:.3f}, "
                    f"neg={self.spectrum.field[n]:.3f}"
                )
                for i, (p, n) in enumerate(pairs)
            ]
            accept = messagebox.askyesno(
                "Peak Finder", "\n".join(lines) + "\nAccept peaks?"
            )

            for m in markers:
                m.remove()
            if self.ax is not None:
                self.ax.figure.canvas.draw_idle()

            if not accept:
                return

        self.auto_peaks.clear()
        self.auto_peaks.extend(pairs)
        self._refresh_tables()
        if not auto:
            messagebox.showinfo("Peak Finder", "Peaks stored for analysis")

    def peak_finder_absorption(self, auto: bool = False) -> None:
        """Locate local maxima in absorption spectra and store them."""

        self._save_state()

        self.spectrum = self.spectra[self.current]
        self.abs_peaks = self.abs_peaks_all[self.current]

        if auto:
            num = 2
        else:
            try:
                num = simpledialog.askinteger(
                    "Peak Finder", "How many peaks to expect?", initialvalue=2, minvalue=1
                )
            except Exception:
                num = 2
            if num is None:
                return
        # Locate local maxima in the absorption trace. ``find_peaks`` returns
        # all peak indices which are then ranked by their height to select the
        # most prominent ``num`` peaks.
        peaks, _ = find_peaks(self.spectrum.intensity)
        if peaks.size < int(num):
            messagebox.showerror("Peak Finder", "Not enough peaks found in the data")
            return

        top = np.argsort(self.spectrum.intensity[peaks])[::-1][: int(num)]
        peaks = peaks[top]
        peaks.sort()
        peaks = [int(p) for p in peaks]

        if not auto:
            markers: list[Line2D] = []
            if self.ax is not None:
                for p in peaks:
                    (marker,) = self.ax.plot(
                        self.spectrum.field[p],
                        self.spectrum.intensity[p],
                        marker="o",
                        color="red",
                    )
                    markers.append(marker)
                self.ax.figure.canvas.draw_idle()

            lines = [
                f"Peak {i + 1}: pos={self.spectrum.field[p]:.3f}"
                for i, p in enumerate(peaks)
            ]
            accept = messagebox.askyesno(
                "Peak Finder", "\n".join(lines) + "\nAccept peaks?"
            )

            for m in markers:
                m.remove()
            if self.ax is not None:
                self.ax.figure.canvas.draw_idle()

            if not accept:
                return

        self.abs_peaks.clear()
        self.abs_peaks.extend(peaks)
        self._refresh_tables()
        if not auto:
            messagebox.showinfo("Peak Finder", "Peaks stored for analysis")

    # ------------------------------------------------------------------
    def onselect(self, xmin: float, xmax: float) -> None:
        """Handle span selections and display peak data."""

        start, end = sorted((xmin, xmax))
        self.ranges.append((start, end))

        try:
            pos_idx, neg_idx = find_peak(
                self.spectrum.field, self.spectrum.intensity, start, end
            )
            width = self.analysis_func(
                self.spectrum.field, self.spectrum.intensity, pos_idx, neg_idx
            )
            pos_field = self.spectrum.field[pos_idx]
            pos_y = self.spectrum.intensity[pos_idx]
            neg_field = self.spectrum.field[neg_idx]
            neg_y = self.spectrum.intensity[neg_idx]

            result = {
                "analysis": self.analysis_label,
                "peak": int(self.current_peak),
                "pos_x": float(pos_field),
                "pos_y": float(pos_y),
                "neg_x": float(neg_field),
                "neg_y": float(neg_y),
                "width": float(width),
            }
            self.results.append(result)

            if self.tree is not None:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        self.analysis_label,
                        f"{self.current_peak}",
                        f"{pos_field:.3f}",
                        f"{pos_y:.3f}",
                        f"{neg_field:.3f}",
                        f"{neg_y:.3f}",
                        f"{width:.3f}",
                    ),
                )

            lines = [
                (
                    f"Peak {r['peak']}: pos={r['pos_x']:.3f}, neg={r['neg_x']:.3f}, "
                    f"{r['analysis']}={r['width']:.3f}"
                )
                for r in self.results
            ]

            # Maintain backwards-compatible notification for the tests
            messagebox.showinfo("Peak analysis", "\n".join(lines))
        except ValueError:
            messagebox.showerror(
                "Peak analysis", "Both peaks must be within the selected range"
            )
        finally:
            if self.selector is not None:
                self.selector.disconnect_events()
            if self.analyse_btn is not None:
                self.analyse_btn.config(state=tk.NORMAL)
            if self.dhpp_btn is not None:
                self.dhpp_btn.config(state=tk.NORMAL)
            if self.find_btn is not None:
                self.find_btn.config(state=tk.NORMAL)
            if getattr(self, "find_abs_btn", None) is not None:
                self.find_abs_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    def _fit_lorentzian(self, auto: bool = False) -> None:
        """Fit a Lorentzian derivative using the full data set.

        Initial parameter guesses are derived from the automatically
        detected peak indices.
        """

        self._save_state()

        assert self.ax is not None

        field = self.spectrum.field
        intensity = self.spectrum.intensity

        is_absorption = "absorption" in self.labels[self.current].lower()
        if is_absorption:
            if len(self.abs_peaks) < self.current_peak:
                self.peak_finder_absorption(auto=auto)
                if len(self.abs_peaks) < self.current_peak:
                    if not auto:
                        messagebox.showwarning("Lorentzian Fit", "No peaks available")
                    return
            peak_idx = self.abs_peaks[self.current_peak - 1]
            h_res_guess = field[peak_idx]
            peak_val = intensity[peak_idx]
            half_val = peak_val / 2.0
            left = peak_idx
            while left > 0 and intensity[left] > half_val:
                left -= 1
            right = peak_idx
            while right < len(intensity) - 1 and intensity[right] > half_val:
                right += 1
            if left == peak_idx or right == peak_idx:
                delta_guess = abs(field[1] - field[0]) * 5.0
            else:
                delta_guess = abs(field[right] - field[left]) / 2.0
            a_guess = peak_val - float(np.min(intensity))
            b_guess = float(np.min(intensity))
            p0 = (h_res_guess, delta_guess, a_guess, b_guess)

            def _model(H: np.ndarray, H_res: float, delta: float, A: float, B: float):
                x = H - H_res
                return A * delta**2 / (x**2 + delta**2) + B

            fit_func = fit_lorentzian_absorption
            param_label = "C"
        else:
            if len(self.auto_peaks) < self.current_peak:
                self.peak_finder(auto=auto)
                if len(self.auto_peaks) < self.current_peak:
                    if not auto:
                        messagebox.showwarning("Lorentzian Fit", "No peaks available")
                    return
            pos_idx, neg_idx = self.auto_peaks[self.current_peak - 1]
            h_res_guess = (field[pos_idx] + field[neg_idx]) / 2.0
            delta_guess = abs(field[pos_idx] - field[neg_idx]) / 2.0
            a_guess = (intensity[pos_idx] - intensity[neg_idx]) / 2.0
            b_guess = 0.0
            p0 = (h_res_guess, delta_guess, a_guess, b_guess)

            def _model(H: np.ndarray, H_res: float, delta: float, A: float, B: float):
                x = H - H_res
                denom = (x**2 + delta**2) ** 2
                sym = -2.0 * delta**2 * x / denom
                disp = delta * (delta**2 - x**2) / denom
                return A * sym + B * disp

            fit_func = fit_lorentzian_derivative
            param_label = "B"

        CHI2_THRESHOLD = 1e-6
        palette = self._get_theme_palette()

        # Perform the initial fit and set up the residual plot and fit line
        params, stats = fit_func(field, intensity, p0=p0)
        h_res, delta, A, B = params
        residuals = stats["residuals"]
        fit = _model(field, h_res, delta, A, B)
        (line,) = self.ax.plot(field, fit, label=f"Lorentzian fit at {h_res:.3f}")
        try:
            line.set_gid("fit")
        except Exception:
            pass
        self.ax.figure.canvas.draw_idle()

        res_plot = plot_residuals(field, residuals, h_res, show=self.plot_frame is None)
        if isinstance(res_plot, tuple):
            fig_r, _ax_r = res_plot
            self._apply_mpl_theme(palette, fig_r)
            if self.plot_frame is not None:
                canvas_r = FigureCanvasTkAgg(fig_r, master=self.plot_frame)
                canvas_r.draw()
                widget_r = canvas_r.get_tk_widget()
                try:
                    widget_r.configure(bg=palette['panel_bg'], highlightthickness=0)
                except Exception:
                    pass
                widget_r.pack(fill=tk.BOTH, expand=True)
                self.extra_canvases.append(canvas_r)
                if self.root is not None:
                    self.root.update_idletasks()
        elif isinstance(res_plot, Figure):
            self._apply_mpl_theme(palette, res_plot)

        # Allow the user to iterate the fit if the statistics indicate a poor result
        if not auto:
            while True:
                chi2 = stats["chi2"]
                stderr = stats["stderr"]

                if chi2 <= CHI2_THRESHOLD:
                    accept = messagebox.askyesno(
                        "Lorentzian Fit",
                        (
                            f"H_res={h_res:.3f}\n",
                            f"Delta={delta:.3f}\nA={A:.3f}\n{param_label}={B:.3f}\n",
                            f"chi^2={chi2:.3e}\n",
                            f"stderr={stderr}\nAccept fit?",
                        ),
                    )
                    if not accept:
                        line.remove()
                        self.ax.figure.canvas.draw_idle()
                        return
                    break

                choice = messagebox.askyesnocancel(
                    "Lorentzian Fit",
                    (
                        f"H_res={h_res:.3f}\n",
                        f"Delta={delta:.3f}\nA={A:.3f}\n{param_label}={B:.3f}\n",
                        f"chi^2={chi2:.3e}\n",
                        f"stderr={stderr}\n",
                        "Fit not optimal.\n",
                        "Yes: accept fit\n",
                        "No: iterate once\n",
                        "Cancel: iterate until convergence",
                    ),
                )

                if choice is True:
                    break
                elif choice is False:
                    p0 = params
                    params, stats = fit_func(field, intensity, p0=p0)
                else:
                    prev = params
                    for _ in range(50):
                        p0 = params
                        params, stats = fit_func(field, intensity, p0=p0)
                        if np.allclose(params, prev, atol=1e-12, rtol=0):
                            break
                        prev = params

                h_res, delta, A, B = params
                fit = _model(field, h_res, delta, A, B)
                line.set_ydata(fit)
                residuals = stats["residuals"]
                res_plot = plot_residuals(field, residuals, h_res, show=self.plot_frame is None)
                if isinstance(res_plot, tuple):
                    fig_r, _ax_r = res_plot
                    self._apply_mpl_theme(palette, fig_r)
                    if self.plot_frame is not None:
                        canvas_r = FigureCanvasTkAgg(fig_r, master=self.plot_frame)
                        canvas_r.draw()
                        widget_r = canvas_r.get_tk_widget()
                        try:
                            widget_r.configure(bg=palette['panel_bg'], highlightthickness=0)
                        except Exception:
                            pass
                        widget_r.pack(fill=tk.BOTH, expand=True)
                        self.extra_canvases.append(canvas_r)
                        if self.root is not None:
                            self.root.update_idletasks()
                elif isinstance(res_plot, Figure):
                    self._apply_mpl_theme(palette, res_plot)
                self.ax.figure.canvas.draw_idle()

        result = {
            "analysis": "Lorentzian",
            "peak": int(self.current_peak),
            "h_res": float(h_res),
            "delta": float(delta),
            "A": float(A),
            "B": float(B),
            "kind": "absorption" if is_absorption else "derivative",
        }
        self.lorentz_results.append(result)
        if self.lorentz_tree is not None:
            self.lorentz_tree.insert(
                "",
                tk.END,
                values=(
                    "Lorentzian",
                    f"{self.current_peak}",
                    f"{h_res:.3f}",
                    f"{delta:.3f}",
                    f"{A:.3f}",
                    f"{B:.3f}",
                ),
            )

        label = line.get_label()
        self.trace_lines.append(line)
        self.spectra.append(
            ESRSpectrum(
                field=field.copy(),
                intensity=fit.copy(),
                metadata=self.spectrum.metadata,
            )
        )
        self.labels.append(label)
        self.results_all.append([])
        self.lorentz_all.append([])
        self.ranges_all.append([])
        self.auto_peaks_all.append([])
        self.abs_peaks_all.append([])

        if (
            self.trace_combo is None
            and self.control_frame is not None
            and self.root is not None
        ):
            insert_before = self.delete_btn if self.delete_btn is not None else None
            self.trace_var = tk.StringVar(value=self.labels[0])
            self.trace_combo = ttk.Combobox(
                self.control_frame,
                textvariable=self.trace_var,
                values=self.labels,
                state="readonly",
            )
            combo_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                combo_pack["before"] = insert_before
            self.trace_combo.bind("<<ComboboxSelected>>", self._on_trace_change)
            self.trace_combo.pack(**combo_pack)

            self.toggle_frame = tk.Frame(self.control_frame)
            toggle_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                toggle_pack["before"] = insert_before
            self.toggle_frame.pack(**toggle_pack)
            tk.Label(self.toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, lbl in enumerate(self.labels):
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=lbl,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)
        else:
            if self.trace_combo is not None and self.trace_var is not None:
                self.trace_combo["values"] = self.labels
            if self.delete_btn is not None:
                self.delete_btn.config(
                    state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED
                )
            if self.toggle_frame is not None:
                var = tk.BooleanVar(value=True)
                idx = len(self.trace_lines) - 1
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=label,
                    variable=var,
                    command=lambda i=idx, v=var: self._toggle_trace(i, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)

        if self.delete_btn is not None:
            self.delete_btn.config(state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED)

        self.update_legend()
        self._rescale()

    def fit_lorentzian(self, auto: bool = False) -> None:
        """Fit the Lorentzian model to automatically detected peak(s)."""

        if auto:
            is_absorption = "absorption" in self.labels[self.current].lower()
            num_peaks = len(self.abs_peaks) if is_absorption else len(self.auto_peaks)
            for i in range(1, num_peaks + 1):
                self.current_peak = i
                self._fit_lorentzian(auto=True)
            return

        peak_choice = self._prompt_peak()
        if peak_choice is None:
            return
        self.current_peak = int(peak_choice)
        self._fit_lorentzian()

    # ------------------------------------------------------------------
    def compare_spectra(self) -> None:
        """Compare analysis results between two traces and tabulate differences."""
        self._save_state()

        indices = self._prompt_traces()
        if indices is None:
            messagebox.showinfo("Compare Spectra", "Two traces are required for comparison")
            return

        first_idx, second_idx = indices
        res1 = self.results_all[first_idx]
        res2 = self.results_all[second_idx]
        lor1 = self.lorentz_all[first_idx]
        lor2 = self.lorentz_all[second_idx]

        def _get(res_list: list[dict[str, float | int | str]], analysis: str, peak: int, key: str) -> float | None:
            for r in res_list:
                if r.get("analysis") == analysis and int(r.get("peak", 0)) == peak:
                    return float(r.get(key))
            return None

        rows: list[tuple[str, float, float, float]] = []
        for peak in (1, 2):
            f1 = _get(res1, "FWHM", peak, "width")
            f2 = _get(res2, "FWHM", peak, "width")
            if f1 is not None and f2 is not None:
                rows.append((f"FWHM P{peak}", f1, f2, f1 - f2))

            d1 = _get(res1, "\u0394H_pp", peak, "width")
            d2 = _get(res2, "\u0394H_pp", peak, "width")
            if d1 is not None and d2 is not None:
                rows.append((f"\u0394H_pp P{peak}", d1, d2, d1 - d2))

            h1 = _get(lor1, "Lorentzian", peak, "h_res")
            h2 = _get(lor2, "Lorentzian", peak, "h_res")
            if h1 is not None and h2 is not None:
                rows.append((f"H_res P{peak}", h1, h2, h1 - h2))

        if not rows:
            messagebox.showinfo("Compare Spectra", "No comparable results found")
            return

        if self.compare_tree is None:
            return

        if self.compare_table_label is not None:
            first_label = self.labels[first_idx] if 0 <= first_idx < len(self.labels) else f"Trace {first_idx + 1}"
            second_label = self.labels[second_idx] if 0 <= second_idx < len(self.labels) else f"Trace {second_idx + 1}"
            detail = f"{first_label} vs {second_label}"
            self.compare_table_label.config(
                text=self._format_table_title(self._compare_table_title, detail)
            )

        for item in self.compare_tree.get_children():
            self.compare_tree.delete(item)

        for name, v1, v2, diff in rows:
            self.compare_tree.insert(
                "",
                tk.END,
                values=(name, f"{v1:.3f}", f"{v2:.3f}", f"{diff:.3f}"),
            )

    # ------------------------------------------------------------------
    def calculate_g(self, quiet: bool = False) -> None:
        """Compute the g-factor for fitted peaks using metadata frequency.

        When ``quiet`` is True, suppresses message dialogs.
        """

        if not self.lorentz_results:
            if not quiet:
                messagebox.showinfo("Calculate g", "No Lorentzian fits available")
            return

        freq = None
        if self.spectrum.metadata is not None:
            freq = self.spectrum.metadata.get("Frequency")
        if freq is None:
            if not quiet:
                messagebox.showinfo("Calculate g", "Frequency metadata not available")
            return

        try:
            freq_val = float(freq)
        except Exception:
            if not quiet:
                messagebox.showinfo("Calculate g", "Invalid frequency value")
            return

        lines: list[str] = []
        for r in self.lorentz_results:
            h_res = float(r.get("h_res", 0.0))
            g_val = calc_g(h_res, freq_val)
            r["g"] = g_val
            lines.append(f"Peak {r['peak']}: g={g_val:.3f}")

        self._refresh_tables()
        if not quiet:
            messagebox.showinfo("Calculate g", "\n".join(lines))

    # ------------------------------------------------------------------
    def analyze_spectra(self) -> None:
        """Run peak finding, linewidth, fitting and g-factor analysis."""

        try:
            # Suppress informational popups during automated pipeline
            self._silent_auto = True
            self.peak_finder(auto=True)
            self.start_peak_to_peak(auto=True)
            self.start_analysis(auto=True)
            self.fit_lorentzian(auto=True)
            self.calculate_g(quiet=True)
        except Exception as exc:
            messagebox.showerror("Analyze Spectra", str(exc))
        finally:
            try:
                del self._silent_auto
            except Exception:
                self._silent_auto = False

    # ------------------------------------------------------------------
    def _select_delete_traces(self) -> list[int] | None:
        """Prompt the user to choose one or more traces to delete.

        Returns a list of selected indices or `None` if cancelled.
        """

        if self.root is None:
            # Fallback: delete current only
            return [self.current] if self.spectra else None

        dialog = tk.Toplevel(self.root)
        dialog.title("Delete Traces")
        tk.Label(dialog, text="Select traces to delete:").pack(padx=10, pady=5)

        listbox = tk.Listbox(dialog, selectmode=tk.EXTENDED, exportselection=False)
        for lbl in self.labels:
            listbox.insert(tk.END, lbl)
        listbox.pack(padx=10, pady=(5, 0), fill=tk.BOTH, expand=True)
        listbox.focus_set()
        tk.Label(
            dialog, text="Tip: Shift-click to select a range; Ctrl-click to toggle items."
        ).pack(padx=10, pady=(2, 8), anchor="w")

        # Preselect current trace for convenience
        try:
            listbox.selection_set(self.current)
        except Exception:
            pass

        selected: list[int] = []

        def on_ok() -> None:
            selected.extend(listbox.curselection())
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        def on_select_all() -> None:
            listbox.select_set(0, tk.END)

        button_cls = getattr(self, "_button_cls", ttk.Button if hasattr(ttk, "Button") else tk.Button)
        base_kwargs = dict(getattr(self, "_button_kwargs", {}))

        def make_button(parent: tk.Widget, **kwargs: object) -> tk.Widget:
            opts = base_kwargs.copy()
            opts.update(kwargs)
            return button_cls(parent, **opts)

        btn = tk.Frame(dialog)
        btn.pack(pady=(0, 10))
        make_button(btn, text="Select All", command=on_select_all).pack(side=tk.LEFT, padx=5)
        make_button(btn, text="Delete", command=on_ok).pack(side=tk.LEFT, padx=5)
        make_button(btn, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

        dialog.grab_set()
        dialog.wait_window()

        if not selected:
            return None
        return selected

    # ------------------------------------------------------------------
    def calculate_area(self) -> None:
        """Calculate the area for fitted Lorentzian absorption peaks."""

        if not self.lorentz_results:
            messagebox.showinfo("Area Integral", "No Lorentzian fits available")
            return

        lines: list[str] = []
        for r in self.lorentz_results:
            if r.get("kind") != "absorption":
                continue
            delta = float(r.get("delta", 0.0))
            amp = float(r.get("A", 0.0))
            area = calc_lorentzian_area(delta, amp)
            r["area"] = area
            lines.append(f"Peak {r['peak']}: area={area:.3f}")

        if not lines:
            messagebox.showinfo("Area Integral", "No absorption Lorentzian fits available")
            return

        self._refresh_tables()
        messagebox.showinfo("Area Integral", "\n".join(lines))

    # ------------------------------------------------------------------
    def _select_batch_spectra(self) -> list[int] | None:
        """Prompt the user to choose spectra for batch processing.

        Returns indices of selected spectra or ``None`` if the user cancels. If
        no Tk root is present all spectra are selected automatically.
        """

        if self.root is None:
            return list(range(len(self.spectra)))

        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Selection")
        tk.Label(dialog, text="Select spectra to process:").pack(padx=10, pady=5)

        listbox = tk.Listbox(dialog, selectmode=tk.EXTENDED, exportselection=False)
        for lbl in self.labels:
            listbox.insert(tk.END, lbl)
        listbox.pack(padx=10, pady=(5, 0), fill=tk.BOTH, expand=True)
        listbox.focus_set()
        tk.Label(
            dialog, text="Tip: Shift-click to select a range; Ctrl-click to toggle items."
        ).pack(padx=10, pady=(2, 8), anchor="w")

        selected: list[int] = []

        def on_ok() -> None:
            selected.extend(listbox.curselection())
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        def on_select_all() -> None:
            listbox.select_set(0, tk.END)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=(0, 5))
        tk.Button(btn_frame, text="Select All", command=on_select_all).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

        dialog.grab_set()
        dialog.wait_window()

        if not selected:
            return None
        return selected

    # ------------------------------------------------------------------
    def _select_export_traces(self) -> list[int] | None:
        """Prompt the user to pick analysed traces for export."""

        eligible = [
            i
            for i, (res, lor) in enumerate(zip(self.results_all, self.lorentz_all))
            if res or lor
        ]
        if not eligible:
            messagebox.showinfo("Export Analysis", "No analysed traces available to export.")
            return None

        if self.root is None:
            return eligible.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Export Analysis")
        dialog.transient(self.root)
        tk.Label(dialog, text="Select analysed traces to export:").pack(padx=10, pady=5)

        listbox = tk.Listbox(dialog, selectmode=tk.EXTENDED, exportselection=False)
        for idx in eligible:
            listbox.insert(tk.END, self.labels[idx] if idx < len(self.labels) else f"Trace {idx + 1}")
        listbox.pack(padx=10, pady=(5, 0), fill=tk.BOTH, expand=True)
        listbox.focus_set()
        tk.Label(
            dialog, text="Tip: Shift-click to select a range; Ctrl-click to toggle items."
        ).pack(padx=10, pady=(2, 8), anchor="w")

        selected: list[int] = []

        def on_ok() -> None:
            selected.extend(eligible[i] for i in listbox.curselection())
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        def on_select_all() -> None:
            listbox.select_set(0, tk.END)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=(0, 5))
        tk.Button(btn_frame, text="Select All", command=on_select_all).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

        dialog.grab_set()
        dialog.wait_window()

        if not selected:
            return None
        return selected

    # ------------------------------------------------------------------
    def batch_process(self) -> None:
        """Automatically analyse all loaded spectra with progress feedback."""

        if not self.spectra:
            return

        indices = self._select_batch_spectra()
        if not indices:
            return

        had_silent = hasattr(self, "_silent_auto")
        previous_silent = getattr(self, "_silent_auto", False)
        self._silent_auto = True

        total = len(indices)
        progress_win = None
        progress_bar = None
        status_label = None
        errors: list[str] = []

        if self.root is not None:
            try:
                progress_win = tk.Toplevel(self.root)
                progress_win.title("Batch Process")
                status_label = tk.Label(progress_win, text="Processing spectra...")
                status_label.pack(padx=10, pady=10)
                progress_bar = ttk.Progressbar(
                    progress_win, orient=tk.HORIZONTAL, length=300, mode="determinate"
                )
                progress_bar.pack(padx=10, pady=(0, 10))
                progress_bar["value"] = 0
            except Exception:
                progress_win = None
                progress_bar = None
                status_label = None

        try:
            for n, i in enumerate(indices):
                label = self.labels[i] if 0 <= i < len(self.labels) else f"Trace {i + 1}"
                if status_label is not None:
                    status_label.config(text=f"Processing {label} ({n + 1}/{total})")
                self.current = i
                self.spectrum = self.spectra[i]
                self.results = self.results_all[i]
                self.lorentz_results = self.lorentz_all[i]
                self.ranges = self.ranges_all[i]
                self.auto_peaks = self.auto_peaks_all[i]
                self.abs_peaks = self.abs_peaks_all[i]
                try:
                    self.peak_finder(auto=True)
                    self.start_peak_to_peak(auto=True)
                    self.start_analysis(auto=True)
                    self.fit_lorentzian(auto=True)
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
                    print(f"[Batch Process] Error processing {label}: {exc}")
                if progress_bar is not None:
                    progress_bar["value"] = (n + 1) / total * 100.0
                    progress_win.update_idletasks()
        finally:
            if had_silent:
                self._silent_auto = previous_silent
            else:
                try:
                    del self._silent_auto
                except AttributeError:
                    pass

        if progress_win is not None:
            if status_label is not None:
                if errors:
                    status_label.config(text="Completed with errors. Check console for details.")
                else:
                    status_label.config(text="Batch process completed successfully.")
                progress_win.update_idletasks()
            progress_win.destroy()

        if errors and self.root is None:
            print(f"[Batch Process] Completed with errors: {'; '.join(errors)}")

        first = indices[0]
        self.current = first
        self.spectrum = self.spectra[first]
        self.results = self.results_all[first]
        self.lorentz_results = self.lorentz_all[first]
        self.ranges = self.ranges_all[first]
        self.auto_peaks = self.auto_peaks_all[first]
        self.abs_peaks = self.abs_peaks_all[first]
        self._refresh_tables()

    # ------------------------------------------------------------------
    def export_analysis_data(self) -> None:
        """Export analysed results and metadata for selected traces to a CSV file."""

        if not self.spectra:
            messagebox.showinfo("Export Analysis", "No spectra loaded.")
            return

        indices = self._select_export_traces()
        if not indices:
            return

        path = filedialog.asksaveasfilename(
            title="Export Analysis Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        metadata_keys: set[str] = set()
        metadata_maps: dict[int, dict[str, object]] = {}
        for idx in indices:
            if 0 <= idx < len(self.spectra):
                meta_raw = getattr(self.spectra[idx], "metadata", None)
                meta_map: dict[str, object] = {}
                if isinstance(meta_raw, dict):
                    for key, value in meta_raw.items():
                        key_str = str(key)
                        meta_map[key_str] = value
                        metadata_keys.add(key_str)
                metadata_maps[idx] = meta_map
            else:
                metadata_maps[idx] = {}

        meta_fields = [f"meta_{key}" for key in sorted(metadata_keys)]
        fieldnames = [
            "trace",
            "category",
            "analysis",
            "peak",
            "pos_x",
            "pos_y",
            "neg_x",
            "neg_y",
            "width",
            "h_res",
            "delta",
            "A",
            "B",
            "area",
            "g",
            "kind",
        ] + meta_fields

        rows: list[dict[str, object]] = []

        for idx in indices:
            label = self.labels[idx] if 0 <= idx < len(self.labels) else f"Trace {idx + 1}"
            metadata_map = metadata_maps.get(idx, {})

            def _base_row() -> dict[str, object]:
                row = {key: "" for key in fieldnames}
                row["trace"] = label
                for meta_key in metadata_keys:
                    row[f"meta_{meta_key}"] = metadata_map.get(meta_key, "")
                return row

            for result in self.results_all[idx]:
                row = _base_row()
                row["category"] = "analysis"
                row["analysis"] = result.get("analysis", "")
                row["peak"] = result.get("peak", "")
                row["pos_x"] = result.get("pos_x", "")
                row["pos_y"] = result.get("pos_y", "")
                row["neg_x"] = result.get("neg_x", "")
                row["neg_y"] = result.get("neg_y", "")
                row["width"] = result.get("width", "")
                rows.append(row)

            for lor in self.lorentz_all[idx]:
                row = _base_row()
                row["category"] = "lorentzian"
                row["analysis"] = lor.get("analysis", "")
                row["peak"] = lor.get("peak", "")
                row["h_res"] = lor.get("h_res", "")
                row["delta"] = lor.get("delta", "")
                row["A"] = lor.get("A", "")
                row["B"] = lor.get("B", "")
                row["area"] = lor.get("area", "")
                row["g"] = lor.get("g", "")
                row["kind"] = lor.get("kind", "")
                rows.append(row)

        if not rows:
            messagebox.showinfo("Export Analysis", "No analysis results available for the selected traces.")
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            messagebox.showerror("Export Analysis", str(exc))
            return

        messagebox.showinfo("Export Analysis", f"Exported {len(rows)} rows to {path}")

    # ------------------------------------------------------------------
    def _get_baseline_options(self) -> tuple[bool, bool] | None:
        """Return user-selected baseline options.

        When running within a Tk application a dedicated dialog presents both
        options at once.  In headless mode the previous messagebox prompts are
        used as a fallback.
        """

        if self.root is None:
            use_poly = messagebox.askyesno(
                "Baseline Correction",
                "Use polynomial baseline?\nSelect 'No' for linear baseline.",
            )
            use_auto = messagebox.askyesno(
                "Baseline Points",
                "Use automatic baseline fit?\nSelect 'No' for manual placement.",
            )
            return use_poly, use_auto

        dialog = BaselineOptionsDialog(self.root)
        dialog.wait_window()
        return dialog.result

    # ------------------------------------------------------------------
    def baseline_correction(self) -> None:
        """Apply a baseline correction to the currently selected trace."""
        self._save_state()

        if self.ax is None:
            return

        options = self._get_baseline_options()
        if options is None:
            return
        use_poly, use_auto = options
        if self.spectrum is None:
            message = "No spectrum selected. Load data before applying baseline correction."
            if self.root is None:
                print(f"[Baseline Correction] {message}")
            else:
                messagebox.showwarning("Baseline Correction", message)
            return

        field = self.spectrum.field
        intensity = self.spectrum.intensity
        degree = 3 if use_poly else 1

        if use_auto:
            n = min(5, len(field) // 2)
            pts = list(zip(field[:n], intensity[:n])) + list(
                zip(field[-n:], intensity[-n:])
            )
            corrected, _baseline = baseline_correct(
                field, intensity, points=pts, degree=degree
            )
            (preview_line,) = self.ax.plot(
                field, _baseline, "k--", linewidth=1
            )
            self.ax.figure.canvas.draw_idle()
            confirm = messagebox.askyesno(
                "Baseline Correction", "Use this automatically generated fit?"
            )
            preview_line.remove()
            self.ax.figure.canvas.draw_idle()
            if not confirm:
                return
        else:
            pts: list[tuple[float, float]] = []
            if self.root is None:
                fig = self.ax.figure
                plt.figure(fig.number)
                try:
                    raw = plt.ginput(n=-1, timeout=-1)
                except Exception:
                    raw = []
                if len(raw) < 2:
                    messagebox.showwarning(
                        "Baseline Correction",
                        "At least two points are required for manual baseline correction.",
                    )
                    return
                for x, _y in raw:
                    idx = int(np.argmin((field - x) ** 2))
                    pts.append((float(field[idx]), float(intensity[idx])))
            else:
                count_var = tk.StringVar(value="Selected points: 0")
                dialog = tk.Toplevel(self.root)
                dialog.title("Baseline Points")
                tk.Label(dialog, textvariable=count_var).pack(padx=10, pady=5)
                confirmed = False

                def confirm() -> None:
                    nonlocal confirmed
                    confirmed = True
                    dialog.destroy()

                def cancel() -> None:
                    dialog.destroy()

                btn = tk.Frame(dialog)
                btn.pack(padx=10, pady=5)
                tk.Button(btn, text="Confirm selection", command=confirm).pack(
                    side=tk.LEFT, padx=5
                )
                tk.Button(btn, text="Cancel", command=cancel).pack(
                    side=tk.LEFT, padx=5
                )
                dialog.protocol("WM_DELETE_WINDOW", cancel)

                editor = BaselinePointEditor(self.ax, field, intensity, degree)

                def _update_count() -> None:
                    count_var.set(f"Selected points: {len(editor.get_points())}")

                editor.on_update = _update_count
                dialog.wait_window()
                editor.disconnect()
                pts = editor.get_points()
                if not confirmed:
                    editor.clear_artists()
                    return
                if len(pts) < 2:
                    editor.clear_artists()
                    messagebox.showwarning(
                        "Baseline Correction",
                        "At least two points are required for manual baseline correction.",
                    )
                    return
            corrected, _baseline = baseline_correct(
                field, intensity, points=pts, degree=degree
            )
            if self.root is not None:
                editor.clear_artists()

        label = f"{self.labels[self.current]} (baseline corrected)"
        (line,) = self.ax.plot(field, corrected, label=label)
        try:
            line.set_gid("trace")
        except Exception:
            pass
        self.trace_lines.append(line)
        self.spectra.append(
            ESRSpectrum(
                field=field.copy(),
                intensity=corrected.copy(),
                metadata=self.spectrum.metadata,
            )
        )
        self.labels.append(label)
        self.results_all.append([])
        self.lorentz_all.append([])
        self.ranges_all.append([])
        self.auto_peaks_all.append([])
        self.abs_peaks_all.append([])

        if (
            self.trace_combo is None
            and self.control_frame is not None
            and self.root is not None
        ):
            insert_before = self.delete_btn if self.delete_btn is not None else None
            self.trace_var = tk.StringVar(value=self.labels[0])
            self.trace_combo = ttk.Combobox(
                self.control_frame,
                textvariable=self.trace_var,
                values=self.labels,
                state="readonly",
            )
            combo_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                combo_pack["before"] = insert_before
            self.trace_combo.bind("<<ComboboxSelected>>", self._on_trace_change)
            self.trace_combo.pack(**combo_pack)

            self.toggle_frame = tk.Frame(self.control_frame)
            toggle_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                toggle_pack["before"] = insert_before
            self.toggle_frame.pack(**toggle_pack)
            tk.Label(self.toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, lbl in enumerate(self.labels):
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=lbl,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)
        else:
            if self.trace_combo is not None and self.trace_var is not None:
                self.trace_combo["values"] = self.labels
            if self.delete_btn is not None:
                self.delete_btn.config(
                    state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED
                )
            if self.toggle_frame is not None:
                var = tk.BooleanVar(value=True)
                idx = len(self.trace_lines) - 1
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=label,
                    variable=var,
                    command=lambda i=idx, v=var: self._toggle_trace(i, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)

        if self.delete_btn is not None:
            self.delete_btn.config(state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED)

        self.update_legend()
        self._rescale()

    # ------------------------------------------------------------------
    def integrate_trace(self) -> None:
        """Integrate the selected derivative trace and plot the absorption spectrum."""
        self._save_state()
        # Operate on the currently selected spectrum
        self.spectrum = self.spectra[self.current]
        absorption = cumulative_trapezoid(
            self.spectrum.intensity, self.spectrum.field, initial=0
        )
        absorption -= float(np.mean(absorption))

        if self.ax is None:
            return

        label = f"{self.labels[self.current]} (absorption)"
        (line,) = self.ax.plot(
            self.spectrum.field,
            absorption,
            label=label,
        )
        try:
            line.set_gid("trace")
        except Exception:
            pass
        self.trace_lines.append(line)
        self.spectra.append(
            ESRSpectrum(
                field=self.spectrum.field.copy(),
                intensity=absorption.copy(),
                metadata=self.spectrum.metadata,
            )
        )
        self.labels.append(label)
        self.results_all.append([])
        self.lorentz_all.append([])
        self.ranges_all.append([])
        self.auto_peaks_all.append([])
        self.abs_peaks_all.append([])

        if (
            self.trace_combo is None
            and self.control_frame is not None
            and self.root is not None
        ):
            insert_before = self.delete_btn if self.delete_btn is not None else None
            self.trace_var = tk.StringVar(value=self.labels[0])
            self.trace_combo = ttk.Combobox(
                self.control_frame,
                textvariable=self.trace_var,
                values=self.labels,
                state="readonly",
            )
            combo_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                combo_pack["before"] = insert_before
            self.trace_combo.bind("<<ComboboxSelected>>", self._on_trace_change)
            self.trace_combo.pack(**combo_pack)

            self.toggle_frame = tk.Frame(self.control_frame)
            toggle_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                toggle_pack["before"] = insert_before
            self.toggle_frame.pack(**toggle_pack)
            tk.Label(self.toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, lbl in enumerate(self.labels):
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=lbl,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)
        else:
            if self.trace_combo is not None and self.trace_var is not None:
                self.trace_combo["values"] = self.labels
            if self.delete_btn is not None:
                self.delete_btn.config(
                    state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED
                )
            if self.toggle_frame is not None:
                var = tk.BooleanVar(value=True)
                idx = len(self.trace_lines) - 1
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=label,
                    variable=var,
                    command=lambda i=idx, v=var: self._toggle_trace(i, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)

        if self.delete_btn is not None:
            self.delete_btn.config(state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED)

        self.update_legend()
        self._rescale()

    # ------------------------------------------------------------------
    def save_results(self, path: Path) -> None:
        """Save analysed peak data to a CSV file.

        Parameters
        ----------
        path:
            Destination file path. Existing files will be overwritten.
        """

        import pandas as pd

        pd.DataFrame(self.results).to_csv(Path(path), index=False)

    def _format_table_title(self, base: str, detail: str | None) -> str:
        """Return a caption with an optional detail suffix."""

        return f"{base} ({detail})" if detail else base

    def _update_table_titles(self, trace_label: str | None) -> None:
        """Update per-trace table headings with the active trace label."""

        for widget, base in (
            (self.peak_table_label, self._peak_table_title),
            (self.results_table_label, self._analysis_table_title),
            (self.lorentz_table_label, self._lorentz_table_title),
        ):
            if widget is not None:
                widget.config(text=self._format_table_title(base, trace_label))

    def _refresh_tables(self) -> None:
        """Refresh the analysis tables for the currently active trace."""

        trace_label: str | None = None
        if self.labels and 0 <= self.current < len(self.labels):
            trace_label = self.labels[self.current]
        self._update_table_titles(trace_label)

        if self.peak_tree is not None:
            for item in self.peak_tree.get_children():
                self.peak_tree.delete(item)
            # Without spectra loaded there is nothing to populate
            if not self.spectra or self.spectrum is None or not self.labels:
                pass
            else:
                label = self.labels[self.current]
                idx = 0
                for i, (p, n) in enumerate(self.auto_peaks):
                    self.peak_tree.insert(
                        "",
                        tk.END,
                        values=(
                            label,
                            f"{i + 1}",
                            f"{self.spectrum.field[p]:.3f}",
                            f"{self.spectrum.field[n]:.3f}",
                        ),
                    )
                    idx = i + 1
                for j, p in enumerate(self.abs_peaks, start=idx + 1):
                    self.peak_tree.insert(
                        "",
                        tk.END,
                        values=(
                            label,
                            f"{j}",
                            f"{self.spectrum.field[p]:.3f}",
                            "",
                        ),
                    )

        if self.tree is not None:
            for item in self.tree.get_children():
                self.tree.delete(item)
            for r in self.results:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        r["analysis"],
                        f"{r['peak']}",
                        f"{r['pos_x']:.3f}",
                        f"{r['pos_y']:.3f}",
                        f"{r['neg_x']:.3f}",
                        f"{r['neg_y']:.3f}",
                        f"{r['width']:.3f}",
                    ),
                )

        if self.lorentz_tree is not None:
            for item in self.lorentz_tree.get_children():
                self.lorentz_tree.delete(item)
            for r in self.lorentz_results:
                g_val = r.get("g")
                g_str = f"{g_val:.3f}" if isinstance(g_val, (int, float)) else ""
                area_val = r.get("area")
                area_str = (
                    f"{area_val:.3f}" if isinstance(area_val, (int, float)) else ""
                )
                self.lorentz_tree.insert(
                    "",
                    tk.END,
                    values=(
                        r["analysis"],
                        f"{r['peak']}",
                        f"{r['h_res']:.3f}",
                        f"{r['delta']:.3f}",
                        f"{r['A']:.3f}",
                        f"{r['B']:.3f}",
                        area_str,
                        g_str,
                    ),
                )

        self._update_batch_table()
        self._update_button_states()

    def _update_batch_table(self) -> None:
        """Populate the batch comparison table with H_res and FWHM values."""

        if self.batch_tree is None or not hasattr(self.batch_tree, "insert"):
            return

        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

        for label, res_list, lor_list in zip(
            self.labels, self.results_all, self.lorentz_all
        ):
            fwhm_vals: dict[int, float] = {}
            for r in res_list:
                if r.get("analysis") == "FWHM":
                    try:
                        fwhm_vals[int(r.get("peak", 0))] = float(r.get("width", 0.0))
                    except Exception:
                        continue
            hres_vals: dict[int, float] = {}
            for r in lor_list:
                if r.get("analysis") == "Lorentzian":
                    try:
                        hres_vals[int(r.get("peak", 0))] = float(r.get("h_res", 0.0))
                    except Exception:
                        continue

            row = [label]
            for peak in (1, 2):
                h = hres_vals.get(peak)
                fwhm = fwhm_vals.get(peak)
                row.append(f"{h:.3f}" if h is not None else "")
                row.append(f"{fwhm:.3f}" if fwhm is not None else "")
            self.batch_tree.insert("", tk.END, values=row)

    def _format_metadata(self, meta: dict[str, object] | None) -> str:
        """Return a human readable string for the acquisition metadata."""

        if not meta:
            return ""

        lines: list[str] = []
        if (freq := meta.get("Frequency")) is not None:
            lines.append(f"Frequency: {freq}")
        if (mod := meta.get("Modulation")) is not None:
            lines.append(f"Modulation: {mod}")
        if (mod_f := meta.get("ModulationFreq")) is not None:
            lines.append(f"Mod. Freq.: {mod_f}")
        if (b_from := meta.get("Bfrom")) is not None and (b_to := meta.get("Bto")) is not None:
            lines.append(f"B Sweep: {b_from}-{b_to}")
        if (mw := meta.get("MicrowavePower")) is not None:
            lines.append(f"MW Power: {mw}")
        if (st := meta.get("SweepTime")) is not None:
            lines.append(f"Sweep Time: {st}")
        if (temp := meta.get("Temperature")) is not None:
            lines.append(f"Temperature: {temp}")
        return "\n".join(lines)

    def _update_metadata_display(self) -> None:
        """Update the metadata label for the currently selected spectrum."""
        if self.spectrum is None:
            text = ""
        else:
            text = self._format_metadata(self.spectrum.metadata)
        self.metadata_text = text
        if self.meta_label is not None:
            self.meta_label.config(text=text)

    def _rescale(self) -> None:
        """Rescale axes to ensure all visible traces are fully shown."""
        if self.ax is None:
            return
        self.ax.relim(visible_only=True)
        self.ax.autoscale()
        self.ax.figure.canvas.draw_idle()

    def _get_canvas_pixel_size(self) -> tuple[int, int]:
        if self._canvas_pixel_size:
            return self._canvas_pixel_size
        if self.figure_canvas is not None:
            fig = self.figure_canvas.figure
            dpi = fig.dpi
            return (int(round(fig.get_figwidth() * dpi)), int(round(fig.get_figheight() * dpi)))
        return (self._min_canvas_width, self._min_canvas_height)

    def _set_canvas_pixel_size(self, width: int, height: int, redraw: bool = True) -> None:
        if self.figure_canvas is None or self.figure_container is None:
            return
        width = max(self._min_canvas_width, int(width))
        height = max(self._min_canvas_height, int(height))
        self._canvas_pixel_size = (width, height)
        try:
            self.figure_container.config(width=width, height=height)
        except Exception:
            pass
        widget = self.figure_canvas.get_tk_widget()
        try:
            widget.configure(width=width, height=height)
        except Exception:
            pass
        fig = self.figure_canvas.figure
        dpi = fig.dpi
        try:
            fig.set_size_inches(width / dpi, height / dpi, forward=True)
        except Exception:
            pass
        if redraw:
            self.figure_canvas.draw_idle()
        self._position_resize_handles()

    def _install_resize_handles(self) -> None:
        container = self.figure_container
        if container is None:
            return
        if getattr(self, "_resize_handles", None):
            self._position_resize_handles()
            return
        highlight_bg = container.cget("highlightbackground") if int(container.cget("highlightthickness") or 0) else ""
        bg = highlight_bg or container.cget("bg")
        handles: list[tk.Widget] = []
        configs = [
            ("east", {"relx": 1.0, "rely": 0.5, "anchor": "e", "relheight": 1.0, "width": 6}),
            ("south", {"relx": 0.5, "rely": 1.0, "anchor": "s", "relwidth": 1.0, "height": 6}),
            ("corner", {"relx": 1.0, "rely": 1.0, "anchor": "se", "width": 14, "height": 14}),
        ]
        cursors = {"east": "sb_h_double_arrow", "south": "sb_v_double_arrow", "corner": "size_nw_se"}
        for mode, opts in configs:
            handle = tk.Frame(container, bg=bg, cursor=cursors.get(mode, "fleur"))
            handle.place(**opts)
            handle.bind("<ButtonPress-1>", lambda event, m=mode: self._start_resize(event, m))
            handle.bind("<B1-Motion>", lambda event, m=mode: self._perform_resize(event, m))
            handle.bind("<ButtonRelease-1>", self._finish_resize)
            handles.append(handle)
        self._resize_handles = handles
        container.bind("<Configure>", lambda _event: self._position_resize_handles(), add="+")
        self._position_resize_handles()

    def _position_resize_handles(self) -> None:
        for handle in getattr(self, "_resize_handles", []):
            try:
                handle.lift()
            except Exception:
                pass

    def _start_resize(self, event: tk.Event, mode: str) -> None:
        container = self.figure_container
        if container is None:
            return
        container.update_idletasks()
        self._resize_state = {
            "mode": mode,
            "start_x": event.x_root,
            "start_y": event.y_root,
            "start_width": container.winfo_width(),
            "start_height": container.winfo_height(),
        }

    def _perform_resize(self, event: tk.Event, mode: str) -> None:
        state = self._resize_state
        if state is None:
            return
        dx = event.x_root - state["start_x"]
        dy = event.y_root - state["start_y"]
        width = state["start_width"]
        height = state["start_height"]
        if mode in ("east", "corner"):
            width = max(self._min_canvas_width, int(state["start_width"] + dx))
        if mode in ("south", "corner"):
            height = max(self._min_canvas_height, int(state["start_height"] + dy))
        self._set_canvas_pixel_size(width, height)

    def _finish_resize(self, _event: tk.Event) -> None:
        self._resize_state = None
        self._position_resize_handles()

    def _toggle_dark_mode(self) -> None:
        if self._dark_mode_var is None or getattr(self, '_updating_theme', False):
            return
        theme = 'dark' if bool(self._dark_mode_var.get()) else 'light'
        self._apply_theme(theme)

    def _configure_button_style(self, palette: dict[str, str]) -> None:
        if self.root is None:
            return
        button_bg = palette.get('button_bg', palette.get('accent', '#4a90e2'))
        button_fg = palette.get('button_fg', palette.get('text', '#ffffff'))
        button_active = palette.get('button_active_bg', button_bg)
        try:
            if isinstance(self._button_cls, type) and issubclass(self._button_cls, ttk.Button):
                style = ttk.Style(self.root)
                for style_name in ('Modern.TButton', 'Compact.TButton', 'TButton', 'Toolbutton'):
                    try:
                        style.configure(style_name, background=button_bg, foreground=button_fg)
                        style.map(style_name, background=[('pressed', button_active), ('active', button_active), ('!disabled', button_bg)], foreground=[('disabled', '#9a9a9a'), ('!disabled', button_fg)])
                    except Exception:
                        continue
            else:
                self._button_kwargs['bg'] = button_bg
                self._button_kwargs['fg'] = button_fg
                self._button_kwargs['activebackground'] = button_active
                self._button_kwargs['activeforeground'] = button_fg
                self._button_kwargs['highlightbackground'] = palette.get('accent', button_bg)
        except Exception:
            pass

    def _iter_control_buttons(self) -> list[tk.Widget]:
        return [
            self.delete_btn,
            self.analyze_btn,
            self.analyse_btn,
            self.dhpp_btn,
            self.find_btn,
            self.find_abs_btn,
            self.fit_btn,
            self.integrate_btn,
            self.baseline_btn,
            self.compare_btn,
            self.g_btn,
            self.area_btn,
            self.export_btn,
            self.batch_btn,
            self.undo_btn,
        ]

    def _style_toggle_checkbutton(self, widget: tk.Checkbutton) -> None:
        palette = self._get_theme_palette()
        try:
            widget.configure(
                bg=palette['panel_bg'],
                fg=palette['text'],
                selectcolor=palette['panel_bg'],
                activebackground=palette['panel_bg'],
                activeforeground=palette['text'],
            )
        except Exception:
            pass

    def _iter_control_buttons(self) -> list[tk.Widget]:
        return [
            self.delete_btn,
            self.analyze_btn,
            self.analyse_btn,
            self.dhpp_btn,
            self.find_btn,
            self.find_abs_btn,
            self.fit_btn,
            self.integrate_btn,
            self.baseline_btn,
            self.compare_btn,
            self.g_btn,
            self.area_btn,
            self.export_btn,
            self.batch_btn,
            self.undo_btn,
        ]

    def _apply_theme(self, theme: str) -> None:
        if self.root is None:
            self._theme = theme if theme in self._THEMES else 'light'
            return
        theme_key = 'dark' if theme == 'dark' else 'light'
        palette = self._THEMES.get(theme_key, self._THEMES['light'])
        self._theme = theme_key
        if self._dark_mode_var is not None and bool(self._dark_mode_var.get()) != (theme_key == 'dark'):
            self._updating_theme = True
            try:
                self._dark_mode_var.set(theme_key == 'dark')
            finally:
                self._updating_theme = False
        button_bg = palette.get('button_bg', palette.get('accent', '#4a90e2'))
        button_fg = palette.get('button_fg', palette.get('text', '#ffffff'))
        button_active = palette.get('button_active_bg', button_bg)
        self._configure_button_style(palette)
        try:
            combo_style = ttk.Style(self.root)
            combo_style.configure('PlotEditor.TCombobox', fieldbackground=palette['entry_bg'], foreground=palette['text'], background=palette['entry_bg'])
            combo_style.map('PlotEditor.TCombobox', fieldbackground=[('readonly', palette['entry_bg'])], foreground=[('readonly', palette['text'])])
        except Exception:
            pass
        try:
            self.root.configure(bg=palette['bg'])
        except Exception:
            pass

        frames = [
            self.plot_container,
            self.plot_frame,
            self.panel_container,
            self.panel_frame,
            self.control_frame,
            self.meta_frame,
            self.results_frame,
            self.lorentz_frame,
            self.compare_frame,
            self.batch_frame,
        ] + getattr(self, '_button_rows', [])
        for frame in frames:
            if frame is None:
                continue
            try:
                frame.configure(bg=palette['panel_bg'])
            except Exception:
                pass
            for child in getattr(frame, 'winfo_children', lambda: [])():
                try:
                    child.configure(bg=palette['panel_bg'])
                except Exception:
                    pass
                if isinstance(child, tk.Label):
                    try:
                        child.configure(bg=palette['panel_bg'], fg=palette['text'])
                    except Exception:
                        pass
                elif isinstance(child, tk.Entry):
                    try:
                        child.configure(bg=palette['entry_bg'], fg=palette['text'], insertbackground=palette['text'])
                    except Exception:
                        pass
                elif isinstance(child, tk.Button):
                    try:
                        child.configure(bg=button_bg, fg=button_fg, activebackground=button_active, activeforeground=button_fg)
                    except Exception:
                        pass

        if self.toggle_frame is not None:
            try:
                self.toggle_frame.configure(bg=palette['panel_bg'])
            except Exception:
                pass
            for child in self.toggle_frame.winfo_children():
                try:
                    child.configure(
                        bg=palette['panel_bg'],
                        fg=palette['text'],
                        selectcolor=palette['panel_bg'],
                        activebackground=palette['panel_bg'],
                        activeforeground=palette['text'],
                    )
                except Exception:
                    pass

        for label in (
            self.meta_label,
            self.peak_table_label,
            self.results_table_label,
            self.lorentz_table_label,
            self.compare_table_label,
        ):
            if label is None:
                continue
            try:
                label.configure(bg=palette['panel_bg'], fg=palette['text'])
            except Exception:
                pass

        if isinstance(self.trace_combo, ttk.Combobox):
            try:
                self.trace_combo.configure(style='PlotEditor.TCombobox')
            except Exception:
                pass
        elif self.trace_combo is not None:
            try:
                self.trace_combo.configure(bg=palette['panel_bg'], fg=palette['text'], activebackground=palette['button_active_bg'], activeforeground=palette['button_fg'])
            except Exception:
                pass

        for extra_canvas in getattr(self, 'extra_canvases', []):
            try:
                self._apply_mpl_theme(palette, extra_canvas.figure)
                widget = extra_canvas.get_tk_widget()
                widget.configure(bg=palette['panel_bg'], highlightthickness=0)
            except Exception:
                pass

        if self.figure_container is not None:
            try:
                self.figure_container.configure(
                    bg=palette['panel_bg'],
                    highlightbackground=palette['accent'],
                    highlightcolor=palette['accent'],
                )
            except Exception:
                pass
        if self.figure_widget is not None:
            try:
                self.figure_widget.configure(bg=palette['plot_face'])
            except Exception:
                pass

        if self.toolbar is not None:
            try:
                self.toolbar.update_theme(palette)  # type: ignore[attr-defined]
            except Exception:
                pass

        try:
            style = ttk.Style(self.root)
            style.configure('Treeview', background=palette['panel_bg'], fieldbackground=palette['panel_bg'], foreground=palette['text'])
            style.configure('Treeview.Heading', foreground=palette['text'], background=palette['panel_bg'])
            style.map('Treeview', background=[('selected', palette['accent'])], foreground=[('selected', palette['text'])])
        except Exception:
            pass

        for handle in getattr(self, '_resize_handles', []):
            try:
                handle.configure(bg=palette['accent'])
            except Exception:
                pass
        self._position_resize_handles()

        self._apply_mpl_theme(palette)

    def _apply_mpl_theme(self, palette: dict[str, str], fig: Figure | None = None) -> None:
        if fig is None:
            if self.figure_canvas is None:
                return
            fig = self.figure_canvas.figure
        try:
            fig.patch.set_facecolor(palette['plot_face'])
        except Exception:
            pass
        for ax in fig.axes:
            ax.set_facecolor(palette['axes_face'])
            for spine in ax.spines.values():
                spine.set_color(palette['axes_edge'])
            ax.tick_params(colors=palette['axes_edge'])
            for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                tick.set_color(palette['axes_edge'])
            ax.xaxis.label.set_color(palette['axes_edge'])
            ax.yaxis.label.set_color(palette['axes_edge'])
            ax.title.set_color(palette['axes_edge'])
            legend = ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor(palette['axes_face'])
                frame.set_edgecolor(palette['axes_edge'])
                for txt in legend.get_texts():
                    txt.set_color(palette['axes_edge'])
        try:
            canvas = fig.canvas
            if canvas is not None:
                canvas.draw_idle()
        except Exception:
            pass

    def _on_trace_change(self, _event: object | None = None) -> None:
        """Update state when the user selects a different trace."""

        if self.trace_var is None:
            return

        label = self.trace_var.get()
        if label not in self.labels:
            return

        self.current = self.labels.index(label)
        self.spectrum = self.spectra[self.current]
        self.results = self.results_all[self.current]
        self.lorentz_results = self.lorentz_all[self.current]
        self.ranges = self.ranges_all[self.current]
        self.auto_peaks = self.auto_peaks_all[self.current]
        self.abs_peaks = self.abs_peaks_all[self.current]

        self._refresh_tables()
        self._update_metadata_display()
        self._rescale()

    # ------------------------------------------------------------------
    def _toggle_trace(self, index: int, show: bool) -> None:
        """Show or hide the trace at the given index."""

        if not (0 <= index < len(self.trace_lines)):
            return

        self.trace_lines[index].set_visible(show)

        self.update_legend()
        self._rescale()
        self._rescale()

    def _set_label(self, index: int, text: str) -> None:
        """Update the stored label for a trace."""

        if 0 <= index < len(self.labels):
            self.labels[index] = text
            if self.trace_combo is not None:
                self.trace_combo["values"] = self.labels
                if self.trace_var is not None and index == self.current:
                    self.trace_var.set(text)

    # ------------------------------------------------------------------
    def delete_trace(self) -> None:
        """Remove one or more traces selected by the user."""

        if len(self.spectra) == 0:
            return

        # Ask the user which traces to delete (multi-select)
        indices = self._select_delete_traces()
        if not indices:
            return

        self._save_state()

        # Delete from highest index to lowest to avoid reindexing issues
        for idx in sorted(indices, reverse=True):
            if not (0 <= idx < len(self.spectra)):
                continue
            del self.spectra[idx]
            del self.labels[idx]
            del self.results_all[idx]
            del self.lorentz_all[idx]
            del self.ranges_all[idx]
            del self.auto_peaks_all[idx]
            del self.abs_peaks_all[idx]
            line = self.trace_lines.pop(idx)
            try:
                line.remove()
            except Exception:
                pass
        if self.toggle_frame is not None:
            # Preserve current visibility of remaining traces
            vis_states = [ln.get_visible() for ln in self.trace_lines]
            for child in self.toggle_frame.winfo_children():
                child.destroy()
            tk.Label(self.toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, (lbl, visible) in enumerate(zip(self.labels, vis_states)):
                var = tk.BooleanVar(value=bool(visible))
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=lbl,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)
        if len(self.spectra) == 0:
            # No spectra left – reset state
            self.current = 0
            self.spectrum = None
            self.results = []
            self.lorentz_results = []
            self.ranges = []
            self.auto_peaks = []
            self.abs_peaks = []
            if self.trace_combo is not None and self.trace_var is not None:
                self.trace_combo["values"] = []
                try:
                    self.trace_var.set("")
                except Exception:
                    pass
        else:
            if self.current >= len(self.spectra):
                self.current = len(self.spectra) - 1
            self.spectrum = self.spectra[self.current]
            self.results = self.results_all[self.current]
            self.lorentz_results = self.lorentz_all[self.current]
            self.ranges = self.ranges_all[self.current]
            self.auto_peaks = self.auto_peaks_all[self.current]
            self.abs_peaks = self.abs_peaks_all[self.current]
            if self.trace_combo is not None and self.trace_var is not None:
                self.trace_combo["values"] = self.labels
                try:
                    self.trace_var.set(self.labels[self.current])
                except Exception:
                    pass
        if self.delete_btn is not None:
            self.delete_btn.config(
                state=tk.NORMAL if len(self.spectra) > 0 else tk.DISABLED
            )
        self._refresh_tables()
        self._update_metadata_display()
        self.update_legend()
        self._rescale()
        self._update_button_states()

    # ------------------------------------------------------------------
    def _append_spectrum(self, spectrum: ESRSpectrum, label: str) -> None:
        """Append a new spectrum to the plot and UI dynamically.

        This mirrors the logic used by integration and fitting to keep all
        per-trace state vectors in sync and the controls up to date.
        """

        if self.ax is None:
            return

        # Plot the new trace
        (line,) = self.ax.plot(spectrum.field, spectrum.intensity, label=label)
        try:
            line.set_gid("trace")
        except Exception:
            pass
        self.trace_lines.append(line)

        # Append to data/model lists
        self.spectra.append(spectrum)
        self.labels.append(label)
        self.results_all.append([])
        self.lorentz_all.append([])
        self.ranges_all.append([])
        self.auto_peaks_all.append([])
        self.abs_peaks_all.append([])

        # If the combobox/toggles are not present yet, create them; otherwise, extend
        if (
            self.trace_combo is None
            and self.control_frame is not None
            and self.root is not None
        ):
            insert_before = self.delete_btn if self.delete_btn is not None else None
            self.trace_var = tk.StringVar(value=self.labels[0])
            self.trace_combo = ttk.Combobox(
                self.control_frame,
                textvariable=self.trace_var,
                values=self.labels,
                state="readonly",
            )
            combo_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                combo_pack["before"] = insert_before
            self.trace_combo.bind("<<ComboboxSelected>>", self._on_trace_change)
            self.trace_combo.pack(**combo_pack)

            self.toggle_frame = tk.Frame(self.control_frame)
            toggle_pack = {"fill": tk.X, "padx": 5, "pady": (0, 5)}
            if insert_before is not None:
                toggle_pack["before"] = insert_before
            self.toggle_frame.pack(**toggle_pack)
            tk.Label(self.toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, lbl in enumerate(self.labels):
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=lbl,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)
        else:
            if self.trace_combo is not None and self.trace_var is not None:
                self.trace_combo["values"] = self.labels
            if self.delete_btn is not None:
                self.delete_btn.config(
                    state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED
                )
            if self.toggle_frame is not None:
                var = tk.BooleanVar(value=True)
                idx = len(self.trace_lines) - 1
                chk = tk.Checkbutton(
                    self.toggle_frame,
                    text=label,
                    variable=var,
                    command=lambda i=idx, v=var: self._toggle_trace(i, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)

        # Keep UI consistent
        if self.delete_btn is not None:
            self.delete_btn.config(state=tk.NORMAL if len(self.labels) > 0 else tk.DISABLED)
        self.update_legend()
        self._rescale()
        self._update_button_states()

    # ------------------------------------------------------------------
    def update_legend(self) -> None:
        """Redraw the legend to reflect current line styles."""

        if self.ax is None:
            return

        handles: list[Line2D] = []
        labels: list[str] = []
        for line, label in zip(self.trace_lines, self.labels):
            line.set_label(label)
            if line.get_visible():
                handles.append(line)
                labels.append(label)

        if handles:
            leg = self.ax.legend(handles, labels)
            leg.set_draggable(True)
        else:
            leg = self.ax.get_legend()
            if leg is not None:
                leg.remove()

        self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _update_button_states(self) -> None:
        """Enable/disable controls depending on whether spectra are loaded."""

        has_data = len(self.spectra) > 0
        state = tk.NORMAL if has_data else tk.DISABLED

        to_toggle = [
            getattr(self, "analyze_btn", None),
            getattr(self, "analyse_btn", None),
            getattr(self, "dhpp_btn", None),
            getattr(self, "find_btn", None),
            getattr(self, "find_abs_btn", None),
            getattr(self, "fit_btn", None),
            getattr(self, "integrate_btn", None),
            getattr(self, "baseline_btn", None),
            getattr(self, "g_btn", None),
            getattr(self, "area_btn", None),
            getattr(self, "batch_btn", None),
        ]
        for btn in to_toggle:
            try:
                if btn is not None:
                    btn.config(state=state)
            except Exception:
                pass

        has_results = any(self.results_all) or any(self.lorentz_all)
        try:
            if getattr(self, "export_btn", None) is not None:
                export_state = tk.NORMAL if (has_data and has_results) else tk.DISABLED
                self.export_btn.config(state=export_state)
        except Exception:
            pass

        # Compare requires at least two spectra
        try:
            if getattr(self, "compare_btn", None) is not None:
                self.compare_btn.config(
                    state=tk.NORMAL if len(self.spectra) >= 2 else tk.DISABLED
                )
        except Exception:
            pass

        # Delete is enabled when at least one spectrum is present
        try:
            if getattr(self, "delete_btn", None) is not None:
                self.delete_btn.config(
                    state=tk.NORMAL if len(self.spectra) > 0 else tk.DISABLED
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _show_readme(self) -> None:
        """Display the project README in a message box."""
        try:
            readme_path = Path(__file__).resolve().parent.parent / "README.md"
            text = readme_path.read_text(encoding="utf-8")
        except Exception as exc:
            text = f"Unable to load README: {exc}"
        messagebox.showinfo("README", text)

    # ------------------------------------------------------------------
    def _show_workflow(self) -> None:
        """Show a brief description of the typical workflow."""
        workflow = (
            "1. Load one or more CSV files containing ESR spectra.\n"
            "2. Use the controls to select peaks and perform analyses.\n"
            "3. Review the results in the tables on the right.\n"
            "4. Optionally fit Lorentzian lines or compare spectra."
        )
        messagebox.showinfo("Workflow", workflow)

    # ------------------------------------------------------------------
    def _show_functions(self) -> None:
        """List available analysis functions with short descriptions."""
        lines: list[str] = []
        for name, (desc, formula) in FUNCTION_DETAILS.items():
            lines.append(name)
            lines.append(f"    {desc}")
            if formula is not None:
                if isinstance(formula, sp.Basic):
                    pretty_lines = sp.pretty(formula, use_unicode=True).splitlines()
                else:
                    pretty_lines = str(formula).splitlines()
                for fl in pretty_lines:
                    lines.append(f"    {fl}")
            lines.append("")
        messagebox.showinfo("Functions", "\n".join(lines).rstrip())

    # ------------------------------------------------------------------
    def _open_file(self) -> None:
        """Open one or more CSV files and add them as new traces."""
        try:
            paths = filedialog.askopenfilenames(
                title="Open ESR CSV File(s)",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            )
        except Exception:
            paths = []

        if not paths:
            return

        for fp in paths:
            try:
                p = Path(fp)
                spec = ESRLoader.load_csv(p)
                label = p.name
                self._append_spectrum(spec, label)
            except Exception as exc:
                try:
                    messagebox.showerror("Open", f"Failed to load {fp}: {exc}")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _view_settings(self) -> None:
        """Placeholder callback for the View menu."""
        try:
            messagebox.showinfo("View", "No view options available.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _create_menu(self) -> None:
        """Create the menu bar with File, View and Help menus."""
        if self.root is None:
            return
        try:
            menubar = tk.Menu(self.root)
            file_menu = tk.Menu(menubar, tearoff=0)
            file_menu.add_command(label="Open", command=self._open_file)
            if hasattr(file_menu, "add_separator"):
                file_menu.add_separator()
            file_menu.add_command(label="Exit", command=getattr(self.root, "quit", lambda: None))

            view_menu = tk.Menu(menubar, tearoff=0)
            if self._dark_mode_var is None:
                self._dark_mode_var = tk.BooleanVar(master=self.root, value=self._theme == 'dark')
            view_menu.add_checkbutton(
                label="Dark Mode",
                variable=self._dark_mode_var,
                command=self._toggle_dark_mode,
            )
            view_menu.add_separator()
            view_menu.add_command(label="Reset View", command=self._view_settings)

            help_menu = tk.Menu(menubar, tearoff=0)
            help_menu.add_command(label="Readme", command=self._show_readme)
            help_menu.add_command(label="Workflow", command=self._show_workflow)
            help_menu.add_command(label="Functions", command=self._show_functions)

            menubar.add_cascade(label="File", menu=file_menu)
            menubar.add_cascade(label="View", menu=view_menu)
            menubar.add_cascade(label="Help", menu=help_menu)
            self.root.config(menu=menubar)
        except Exception:
            # In headless environments or tests the Tk primitives may not be
            # fully initialised.  Failing silently keeps the rest of the GUI
            # functional while still allowing the menu to appear when running
            # interactively.
            pass

    # ------------------------------------------------------------------
    def show(self) -> None:
        """Start the Tkinter main loop and display the analysis GUI."""

        # ``ButtonCls`` and ``button_kwargs`` allow us to swap out the widget
        # implementation depending on whether ``ttkbootstrap`` is available.  The
        # themed widgets from ``ttkbootstrap`` feature rounded corners which give
        # the interface a softer appearance reminiscent of modern "flat" GUI
        # design.
        ButtonCls: type[tk.Button] | type[ttk.Button]
        button_kwargs: dict[str, object]

        style = None
        self.root = None

        if ttkbootstrap is not None:  # pragma: no cover - depends on optional lib
            try:
                style = ttkbootstrap.Style(theme="flatly")
                self.root = style.master
                ButtonCls = ttkbootstrap.Button
                button_kwargs = {"bootstyle": ("primary", "round")}
                style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
            except Exception:
                # If ``ttkbootstrap`` cannot initialise (e.g. no display), fall
                # back to the classic ``ttk`` widgets below.
                self.root = None

        if self.root is None:
            self.root = tk.Tk()
            ButtonCls = ttk.Button
            button_kwargs = {"style": "Modern.TButton"}
            try:
                style = ttk.Style(self.root)
                style.theme_use("clam")
                style.configure(
                    "Modern.TButton",
                    font=("TkDefaultFont", 9, "bold"),
                    relief="raised",
                    borderwidth=3,
                    background="#4a90e2",
                    foreground="white",
                    padding=(5, 2),
                )
                style.map(
                    "Modern.TButton",
                    background=[("active", "#357ab7"), ("!disabled", "#4a90e2")],
                    foreground=[("!disabled", "white")],
                )
                style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
            except Exception:
                ButtonCls = tk.Button
                button_kwargs = {
                    "font": ("TkDefaultFont", 9, "bold"),
                    "relief": tk.RAISED,
                    "bd": 3,
                    "bg": "#4a90e2",
                    "fg": "white",
                    "activebackground": "#357ab7",
                }

        self._button_cls = ButtonCls
        self._button_kwargs = dict(button_kwargs)
        self._dark_mode_var = tk.BooleanVar(master=self.root, value=self._theme == 'dark')
        palette = self._THEMES.get(self._theme, self._THEMES['light'])
        self._configure_button_style(palette)

        # Basic window housekeeping such as maximising the window if supported.
        self.root.title("SimpleESR")
        try:
            self.root.update_idletasks()
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                try:
                    width = self.root.winfo_screenwidth()
                    height = self.root.winfo_screenheight()
                    self.root.geometry(f"{width}x{height}+0+0")
                except Exception:
                    pass

        self._create_menu()

        # Keep the analysis panel at roughly a quarter of the window width.  A
        # simple two-column grid layout with weights of 3:1 ensures that the
        # plot takes up 75% of the available space while the panel receives the
        # remaining 25%.  The lightweight dummy Tk classes used in tests do not
        # implement ``grid`` methods, so we only configure the grid if available
        # and fall back to ``pack`` otherwise.
        use_grid = all(
            hasattr(self.root, attr)
            for attr in ("grid_rowconfigure", "grid_columnconfigure")
        )
        if use_grid:
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=3)
            self.root.grid_columnconfigure(1, weight=1)

        def _wrap_buttons(frame: tk.Frame) -> None:
            """Reposition buttons when the available width changes.

            Widgets are arranged from left to right and wrapped onto a new row if
            the next button would exceed the frame's current width.  This emulates
            a flow layout so that controls remain accessible even when the
            analysis panel becomes narrow."""

            if not (hasattr(frame, "bind") and hasattr(frame, "winfo_children")):
                return

            def _do_wrap(event: tk.Event | None = None) -> None:
                width = event.width if event else getattr(frame, "winfo_width", lambda: 0)()
                if width <= 1:
                    return
                x = 0
                row = 0
                col = 0
                pad = 4
                for w in frame.winfo_children():
                    getattr(w, "update_idletasks", lambda: None)()
                    w_width = getattr(w, "winfo_reqwidth", lambda: 0)()
                    if x + w_width > width and x > 0:
                        row += 1
                        col = 0
                        x = 0
                    if hasattr(w, "grid"):
                        w.grid(row=row, column=col, padx=2, pady=2, sticky="w")
                    col += 1
                    x += w_width + pad

            frame.bind("<Configure>", _do_wrap)
            _do_wrap()

        try:
            plot_container = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
            if use_grid and hasattr(plot_container, "grid"):
                plot_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            else:
                plot_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            plot_canvas = tk.Canvas(plot_container, highlightthickness=0)
            plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            plot_scroll = tk.Scrollbar(plot_container, orient=tk.VERTICAL, command=plot_canvas.yview)
            plot_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            plot_canvas.configure(yscrollcommand=plot_scroll.set)

            plot_frame = tk.Frame(plot_canvas)
            plot_id = plot_canvas.create_window((0, 0), window=plot_frame, anchor="nw")

            def _on_plot_configure(_event: tk.Event) -> None:
                plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

            plot_frame.bind("<Configure>", _on_plot_configure)

            def _on_plot_canvas_configure(event: tk.Event) -> None:
                plot_canvas.itemconfigure(plot_id, width=event.width)

            plot_canvas.bind("<Configure>", _on_plot_canvas_configure)

            def _on_plot_mousewheel(event: tk.Event) -> None:
                plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            plot_canvas.bind(
                "<Enter>", lambda _e: plot_canvas.bind_all("<MouseWheel>", _on_plot_mousewheel)
            )
            plot_canvas.bind(
                "<Leave>", lambda _e: plot_canvas.unbind_all("<MouseWheel>")
            )
        except Exception:
            plot_frame = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
            if use_grid and hasattr(plot_frame, "grid"):
                plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            else:
                plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            plot_container = plot_frame
        self.plot_container = plot_container
        self.plot_frame = plot_frame

        try:
            panel_container = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
            if use_grid and hasattr(panel_container, "grid"):
                panel_container.grid(row=0, column=1, sticky="nsew")
            else:
                panel_container.pack(side=tk.RIGHT, fill=tk.Y)
            panel_canvas = tk.Canvas(panel_container, highlightthickness=0)
            panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = tk.Scrollbar(panel_container, orient=tk.VERTICAL, command=panel_canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            panel_canvas.configure(yscrollcommand=scrollbar.set)

            panel = tk.Frame(panel_canvas)
            panel_id = panel_canvas.create_window((0, 0), window=panel, anchor="nw")

            def _on_panel_configure(_event: tk.Event) -> None:
                panel_canvas.configure(scrollregion=panel_canvas.bbox("all"))

            panel.bind("<Configure>", _on_panel_configure)

            def _on_canvas_configure(event: tk.Event) -> None:
                panel_canvas.itemconfigure(panel_id, width=event.width)

            panel_canvas.bind("<Configure>", _on_canvas_configure)

            def _on_mousewheel(event: tk.Event) -> None:
                panel_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            panel_canvas.bind(
                "<Enter>", lambda _e: panel_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            )
            panel_canvas.bind(
                "<Leave>", lambda _e: panel_canvas.unbind_all("<MouseWheel>")
            )
        except Exception:
            panel = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
            if use_grid and hasattr(panel, "grid"):
                panel.grid(row=0, column=1, sticky="nsew")
            else:
                panel.pack(side=tk.RIGHT, fill=tk.Y)
            panel_container = panel

        self.panel_container = panel_container
        self.panel_frame = panel

        # ------------------------------------------------------------------
        # Metadata panel
        meta_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        meta_frame.pack(fill=tk.X, pady=(0, 10))
        self.meta_frame = meta_frame
        tk.Label(meta_frame, text="Metadata", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=5, pady=(5, 0)
        )
        self.meta_label = tk.Label(meta_frame, justify=tk.LEFT)
        self.meta_label.pack(anchor="w", padx=5, pady=(0, 5))
        self._update_metadata_display()

        # ------------------------------------------------------------------
        # Plot area with toolbar on top
        fig, self.ax = plt.subplots()
        self.trace_lines = []
        for spec in self.spectra:
            line, = self.ax.plot(spec.field, spec.intensity)
            try:
                line.set_gid("trace")
            except Exception:
                pass
            self.trace_lines.append(line)
        self.ax.set_xlabel("Magnetic Field")
        self.ax.set_ylabel("Intensity")
        self.update_legend()

        border_color = palette.get("accent", "#4a90e2")
        base_bg = plot_frame.cget("bg") if hasattr(plot_frame, "cget") else palette.get("panel_bg", "#f0f0f0")
        self.figure_container = tk.Frame(
            plot_frame,
            bd=2,
            relief=tk.GROOVE,
            highlightthickness=2,
            highlightbackground=border_color,
            highlightcolor=border_color,
            bg=base_bg,
        )
        self.figure_container.pack_propagate(False)
        canvas = FigureCanvasTkAgg(fig, master=self.figure_container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbarNoSubplots(
            canvas,
            plot_frame,
            get_active_index=lambda: self.current,
            update_legend=self.update_legend,
            set_label=self._set_label,
            get_trace_line=lambda i: self.trace_lines[i] if 0 <= i < len(self.trace_lines) else None,
            get_canvas_size=self._get_canvas_pixel_size,
            set_canvas_size=self._set_canvas_pixel_size,
            get_theme_palette=lambda: self._THEMES.get(self._theme, self._THEMES['light']),
            pack_toolbar=False,
        )
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = toolbar
        self.figure_container.pack(side=tk.TOP, anchor="nw", padx=5, pady=5)
        try:
            self.figure_container.update_idletasks()
        except Exception:
            pass

        self.figure_canvas = canvas
        self.figure_widget = canvas_widget
        self._resize_handles = []
        self._canvas_pixel_size = None

        initial_width = int(round(fig.get_figwidth() * fig.dpi))
        initial_height = int(round(fig.get_figheight() * fig.dpi))
        self._set_canvas_pixel_size(initial_width, initial_height, redraw=False)
        canvas.draw()
        self._install_resize_handles()

        # ------------------------------------------------------------------
        # Controls
        control_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        self.control_frame = control_frame
        tk.Label(control_frame, text="Controls", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=5, pady=(5, 0)
        )

        if len(self.spectra) > 1:
            tk.Label(control_frame, text="Graph to be edited/analyzed").pack(
                padx=5, pady=(0, 5)
            )
            self.trace_var = tk.StringVar(value=self.labels[0])
            self.trace_combo = ttk.Combobox(
                control_frame,
                textvariable=self.trace_var,
                values=self.labels,
                state="readonly",
            )
            self.trace_combo.bind("<<ComboboxSelected>>", self._on_trace_change)
            self.trace_combo.pack(fill=tk.X, padx=5, pady=(0, 5))
            toggle_frame = tk.Frame(control_frame)
            toggle_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            tk.Label(toggle_frame, text="Visible traces").pack(anchor="w")
            self.trace_vars = []
            for i, label in enumerate(self.labels):
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(
                    toggle_frame,
                    text=label,
                    variable=var,
                    command=lambda idx=i, v=var: self._toggle_trace(idx, v.get()),
                )
                chk.pack(anchor="w")
                self._style_toggle_checkbutton(chk)
                self.trace_vars.append(var)
            self.toggle_frame = toggle_frame

        # Always present a Delete button; enabled when at least one trace exists
        self.delete_btn = ButtonCls(
            control_frame,
            text="Delete Trace",
            command=self.delete_trace,
            **button_kwargs,
        )
        self.delete_btn.pack(fill=tk.X, padx=5, pady=(0, 5))
        try:
            self.delete_btn.config(state=tk.NORMAL if len(self.spectra) > 0 else tk.DISABLED)
        except Exception:
            pass

        # Button rows for compact layout
        button_row1 = tk.Frame(control_frame)
        button_row1.pack(fill=tk.X, padx=5, pady=2)

        self.analyze_btn = ButtonCls(
            button_row1,
            text="Auto Analyze",
            command=self.analyze_spectra,
            **button_kwargs,
        )
        self.analyse_btn = ButtonCls(
            button_row1,
            text="Analyse FWHM",
            command=self.start_analysis,
            **button_kwargs,
        )
        self.dhpp_btn = ButtonCls(
            button_row1,
            text="Analyse \u0394H_pp",
            command=self.start_peak_to_peak,
            **button_kwargs,
        )
        self.find_btn = ButtonCls(
            button_row1,
            text="Find Peaks",
            command=self.peak_finder,
            **button_kwargs,
        )
        self.find_abs_btn = ButtonCls(
            button_row1,
            text="Find Absorption Peaks",
            command=self.peak_finder_absorption,
            **button_kwargs,
        )
        _wrap_buttons(button_row1)

        button_row2 = tk.Frame(control_frame)
        button_row2.pack(fill=tk.X, padx=5, pady=(2, 5))

        self.fit_btn = ButtonCls(
            button_row2,
            text="Fit Lorentzian",
            command=self.fit_lorentzian,
            **button_kwargs,
        )
        self.integrate_btn = ButtonCls(
            button_row2,
            text="Integrate Trace",
            command=self.integrate_trace,
            **button_kwargs,
        )
        self.baseline_btn = ButtonCls(
            button_row2,
            text="Baseline Correct",
            command=self.baseline_correction,
            **button_kwargs,
        )
        self.compare_btn = ButtonCls(
            button_row2,
            text="Compare Spectra",
            command=self.compare_spectra,
            **button_kwargs,
        )
        _wrap_buttons(button_row2)

        button_row3 = tk.Frame(control_frame)
        button_row3.pack(fill=tk.X, padx=5, pady=(2, 5))
        self._button_rows = [button_row1, button_row2, button_row3]

        self.g_btn = ButtonCls(
            button_row3,
            text="Calculate g",
            command=self.calculate_g,
            **button_kwargs,
        )

        self.area_btn = ButtonCls(
            button_row3,
            text="Area Integral",
            command=self.calculate_area,
            **button_kwargs,
        )

        self.export_btn = ButtonCls(
            button_row3,
            text="Export Analysis",
            command=self.export_analysis_data,
            **button_kwargs,
        )

        self.batch_btn = ButtonCls(
            button_row3,
            text="Batch Process",
            command=self.batch_process,
            **button_kwargs,
        )

        self.undo_btn = ButtonCls(
            button_row3,
            text="Undo",
            command=self.undo,
            **button_kwargs,
        )
        self.undo_btn.config(state=tk.DISABLED)
        _wrap_buttons(button_row3)

        # ------------------------------------------------------------------
        # Peak position table
        peak_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        peak_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.peak_table_label = tk.Label(
            peak_frame, text=self._peak_table_title, font=("TkDefaultFont", 10, "bold")
        )
        self.peak_table_label.pack(anchor="w", padx=5, pady=(5, 0))
        peak_columns = ("trace", "peak", "pos", "neg")
        self.peak_tree = ttk.Treeview(
            peak_frame, columns=peak_columns, show="headings", height=5
        )
        peak_headings = {
            "trace": "Trace",
            "peak": "Peak",
            "pos": "Pos X",
            "neg": "Neg X",
        }
        for col, text in peak_headings.items():
            self.peak_tree.heading(col, text=text)
            self.peak_tree.column(col, anchor=tk.CENTER)
        self.peak_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # ------------------------------------------------------------------
        # Results tables
        result_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.results_frame = result_frame
        self.results_table_label = tk.Label(
            result_frame, text=self._analysis_table_title, font=("TkDefaultFont", 10, "bold")
        )
        self.results_table_label.pack(anchor="w", padx=5, pady=(5, 0))
        columns = ("analysis", "peak", "pos_x", "pos_y", "neg_x", "neg_y", "width")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=5)
        headings = {
            "analysis": "Analysis",
            "peak": "Peak",
            "pos_x": "Pos X",
            "pos_y": "Pos Y",
            "neg_x": "Neg X",
            "neg_y": "Neg Y",
            "width": "Value",
        }
        for col, text in headings.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        lorentz_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        lorentz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.lorentz_frame = lorentz_frame
        self.lorentz_table_label = tk.Label(
            lorentz_frame, text=self._lorentz_table_title, font=("TkDefaultFont", 10, "bold")
        )
        self.lorentz_table_label.pack(anchor="w", padx=5, pady=(5, 0))
        lorentz_columns = ("analysis", "peak", "h_res", "delta", "A", "B", "area", "g")
        self.lorentz_tree = ttk.Treeview(
            lorentz_frame, columns=lorentz_columns, show="headings", height=5
        )
        lorentz_headings = {
            "analysis": "Analysis",
            "peak": "Peak",
            "h_res": "H_res",
            "delta": "Delta",
            "A": "A",
            "B": "B",
            "area": "Area",
            "g": "g",
        }
        for col, text in lorentz_headings.items():
            self.lorentz_tree.heading(col, text=text)
            self.lorentz_tree.column(col, anchor=tk.CENTER)
        self.lorentz_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        compare_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        compare_frame.pack(fill=tk.BOTH, expand=True)
        self.compare_frame = compare_frame
        self.compare_table_label = tk.Label(
            compare_frame, text=self._compare_table_title, font=("TkDefaultFont", 10, "bold")
        )
        self.compare_table_label.pack(anchor="w", padx=5, pady=(5, 0))
        compare_cols = ("param", "first", "second", "diff")
        self.compare_tree = ttk.Treeview(
            compare_frame, columns=compare_cols, show="headings", height=6
        )
        compare_headings = {
            "param": "Parameter",
            "first": "Trace 1",
            "second": "Trace 2",
            "diff": "Diff",
        }
        for col, text in compare_headings.items():
            self.compare_tree.heading(col, text=text)
            self.compare_tree.column(col, anchor=tk.CENTER)
        self.compare_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        batch_frame = tk.Frame(panel, bd=2, relief=tk.GROOVE)
        batch_frame.pack(fill=tk.BOTH, expand=True)
        self.batch_frame = batch_frame
        tk.Label(
            batch_frame, text="Batch Results", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", padx=5, pady=(5, 0))
        batch_cols = ("trace", "h1", "f1", "h2", "f2")
        self.batch_tree = ttk.Treeview(
            batch_frame, columns=batch_cols, show="headings", height=6
        )
        batch_headings = {
            "trace": "Trace",
            "h1": "H_res P1",
            "f1": "FWHM P1",
            "h2": "H_res P2",
            "f2": "FWHM P2",
        }
        for col, text in batch_headings.items():
            self.batch_tree.heading(col, text=text)
            self.batch_tree.column(col, anchor=tk.CENTER)
        self.batch_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self._apply_theme(self._theme)

        # Ensure the tables reflect any results already calculated before the GUI
        self._refresh_tables()
        # Disable analysis controls if no spectra loaded yet
        self._update_button_states()
        self.root.mainloop()


def main() -> None:
    """Start the analyser GUI; load files via File > Open."""

    try:
        app = SpanPeakSelector()
        app.show()
    except Exception as exc:  # pragma: no cover - GUI error handling
        messagebox.showerror("Error", str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
