"""Formula text used by GUI2 tooltips."""

from __future__ import annotations

from typing import Any

ANALYSIS_HEADER_TOOLTIPS: dict[str, str] = {
    "analysis": (
        "Analysis kind for the selected peak pair. "
        "dH_pp = abs(H_pos - H_neg). "
        "FWHM (derivative Lorentzian) = sqrt(3) * dH_pp."
    ),
    "peak": "1-based peak index from automated peak detection.",
    "pos_x": "H_pos = field[pos_idx] from detected positive extremum.",
    "neg_x": "H_neg = field[neg_idx] from detected negative extremum.",
    "width": "Derived linewidth value for the selected analysis method.",
}

FIT_HEADER_TOOLTIPS: dict[str, str] = {
    "peak": "1-based peak index from automated peak detection.",
    "kind": "Fit model: derivative or absorption Lorentzian.",
    "h_res": "Resonance field from nonlinear least-squares fit (mT).",
    "delta": "Lorentzian half-width at half-maximum parameter (mT).",
    "A": "Fit amplitude parameter from the Lorentzian model.",
    "B": "Derivative mode uses B (dispersive mix). Absorption mode uses C (offset).",
    "g": (
        "g = h * nu / (mu_B * H_res). "
        "Use nu in Hz and H_res in T."
    ),
    "g_err": (
        "sigma_g = |g| * sqrt((sigma_H/H_res)^2 + (sigma_nu/nu)^2). "
        "sigma_H comes from fit stderr(H_res)."
    ),
    "g_err_pct": "%g_err = 100 * sigma_g / |g|.",
    "area": "Absorption area = pi * A * delta.",
    "area_err": (
        "sigma_area = sqrt((pi*delta)^2*sigma_A^2 + (pi*A)^2*sigma_delta^2 + "
        "2*(pi*delta)*(pi*A)*cov(A, delta))."
    ),
    "area_err_pct": "%area_err = 100 * sigma_area / |area|.",
    "chi2": (
        "Reduced chi2: chi2 = (1/dof) * sum(((y_i - yhat_i)/sigma)^2), "
        "dof = N - p, sigma ~= 1.4826 * MAD(residuals). "
        "If sigma ~ 0, fallback is sum(residual^2)/dof."
    ),
    "stderr": "Parameter standard errors: stderr = sqrt(diag(covariance)).",
}

GROUP_HEADER_TOOLTIPS: dict[str, str] = {
    "group": "Replicate group key inferred from filename by removing R# token.",
    "peak": "Peak index within the fit model.",
    "kind": "Model type for grouped fit rows (derivative/absorption).",
    "n_total": "Total replicate rows found for this group and peak.",
    "n_used": "Rows retained after QC and outlier screening.",
    "h_res_mean": "Arithmetic mean of retained H_res values.",
    "h_res_std": "Sample standard deviation of retained H_res values.",
    "h_res_sem": "Standard error of mean: std / sqrt(n_used).",
    "h_res_wmean": "Weighted mean H_res with weights w_i = 1 / sigma_i^2 from stderr(H_res).",
    "h_res_wsem": "Weighted SEM = sqrt(1 / sum(w_i)).",
    "h_res_err_total": "Combined uncertainty = sqrt(SEM^2 + mean(sigma_i^2)/n_used).",
    "delta_mean": "Mean delta across retained replicates.",
    "delta_std": "Sample std of delta across retained replicates.",
    "g_mean": "Mean g across retained replicates.",
    "g_std": "Sample std of g across retained replicates.",
    "area_mean": "Mean area across retained replicates (absorption fits).",
    "area_std": "Sample std of area across retained replicates.",
    "chi2_mean": "Mean reduced chi2 across retained replicates.",
    "included": "Trace labels included in this group summary.",
    "rejected": "Trace labels rejected by QC (invalid fit, high chi2, or outlier).",
}

CONTROL_TOOLTIPS: dict[str, str] = {
    "open_btn": "Import one or more CSV traces for batch analysis.",
    "analyze_btn": "Run full pipeline on active trace: peaks, widths, fits, g, and uncertainties.",
    "batch_btn": "Launch the analysis wizard, then run the pipeline on all loaded traces.",
    "export_btn": "Export analysis and fit values, including uncertainty columns, to CSV.",
    "theme_btn": "Toggle UI theme.",
    "trace_list": "Select active trace. Checkbox toggles visibility in the live plot.",
    "show_selected_btn": "Set selected trace(s) visible.",
    "hide_selected_btn": "Set selected trace(s) hidden.",
    "progress": "Progress percent = 100 * completed / total jobs.",
    "analysis_table": "Hover headers or cells to see formulas for each analysis value.",
    "analysis_filters": "Toggle which critical points (Delta H_pp vs. FWHM) remain visible in the table and plot markers.",
    "analysis_filter_dhpp": "Show or hide Delta H_pp rows and their plot markers.",
    "analysis_filter_fwhm": "Show or hide FWHM rows and their plot markers.",
    "fit_table": "Hover headers or cells to see fitting and uncertainty equations.",
    "group_table": "Grouped replicate summaries from filenames with R# tokens.",
    "plot_widget": "Live traces with optional Lorentzian fit overlays.",
}


def analysis_cell_tooltip(row: dict[str, Any], key: str) -> str | None:
    """Return row-aware analysis tooltip text."""

    if key != "width":
        return ANALYSIS_HEADER_TOOLTIPS.get(key)
    kind = str(row.get("analysis", "")).strip().lower()
    if kind == "dh_pp":
        return "dH_pp = abs(H_pos - H_neg)."
    if kind == "fwhm":
        return "FWHM = sqrt(3) * abs(H_pos - H_neg) for derivative Lorentzian peaks."
    return ANALYSIS_HEADER_TOOLTIPS.get("width")


def fit_cell_tooltip(_row: dict[str, Any], key: str) -> str | None:
    """Return fit tooltip text for one column key."""

    return FIT_HEADER_TOOLTIPS.get(key)
