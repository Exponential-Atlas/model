"""
Exponential Atlas v6 — Backtesting
====================================

Validates the model by fitting only on pre-cutoff data and comparing
predictions against actual post-cutoff observations.

This is the **gold standard** for model validation: can the model, given
only data available at time T, accurately predict what happened after T?

The v5 model had a critical bug: backtesting used only log-linear fits.
v6 uses ``select_best_fit()`` — the same BIC-based model selection used
for forward projections — ensuring backtest accuracy reflects real model
performance, not a simplified version.

Methodology
-----------
For each cutoff year and each domain:

1. Filter data points to those with ``year <= cutoff_year``.
2. Require at least 3 pre-cutoff points (below that, fitting is unreliable
   and the model would rightfully decline to project).
3. Fit using ``select_best_fit()`` with the domain's configuration.
4. Predict for every post-cutoff actual data point.
5. Compute error metrics:
   - **MAPE** (Mean Absolute Percentage Error) — interpretable as
     "on average, predictions are X% off".
   - **Log error** — better for exponential quantities because a 2x
     overshoot and a 0.5x undershoot are symmetric in log-space.
   - **Bias direction** — does the model systematically over-predict
     (optimistic) or under-predict (pessimistic)?

Calibration
-----------
If the backtest reveals systematic bias (e.g., 20% average overshoot),
we can compute a calibration factor to apply to forward projections.
This is disclosed transparently: "Our model historically over-predicted
by X%, so we apply a Y correction factor."
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class DomainBacktestResult:
    """Backtest result for a single domain at a single cutoff year.

    Attributes
    ----------
    domain_id : str
        Domain identifier, e.g. ``"solar_module"``.
    cutoff_year : int
        Year at which data was cut off.
    n_pre_cutoff : int
        Number of data points used for fitting.
    n_post_cutoff : int
        Number of data points used for evaluation.
    fit_method : str
        Which fit method was selected by BIC.
    r_squared_fit : float
        R-squared of the fit on pre-cutoff data.
    predictions : list[dict]
        One entry per post-cutoff data point::

            {
                "year": float,
                "actual": float,
                "predicted": float,
                "pct_error": float,
                "log_error": float,
            }
    mape : float
        Mean Absolute Percentage Error across post-cutoff points.
    mean_log_error : float
        Mean log error (signed: positive = overshoot).
    bias_direction : str
        ``"over"`` if mean_log_error > 0.05,
        ``"under"`` if < -0.05,
        ``"neutral"`` otherwise.
    skipped : bool
        True if the domain was skipped (insufficient data).
    skip_reason : str
        Reason for skipping, if applicable.
    """

    domain_id: str
    cutoff_year: int
    n_pre_cutoff: int = 0
    n_post_cutoff: int = 0
    fit_method: str = ""
    r_squared_fit: float = 0.0
    predictions: list[dict] = field(default_factory=list)
    mape: float = float("nan")
    mean_log_error: float = float("nan")
    bias_direction: str = "neutral"
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class CutoffYearResult:
    """Aggregated backtest results for a single cutoff year.

    Attributes
    ----------
    cutoff_year : int
        The cutoff year.
    domains_tested : int
        Number of domains with enough data.
    domains_skipped : int
        Number of domains skipped.
    overall_mape : float
        Weighted-average MAPE across all domains.
    overall_mean_log_error : float
        Average log error.
    overall_bias : str
        ``"over"`` / ``"under"`` / ``"neutral"``.
    per_domain : dict[str, DomainBacktestResult]
        Results for each domain.
    """

    cutoff_year: int
    domains_tested: int = 0
    domains_skipped: int = 0
    overall_mape: float = float("nan")
    overall_mean_log_error: float = float("nan")
    overall_bias: str = "neutral"
    per_domain: dict[str, DomainBacktestResult] = field(default_factory=dict)


@dataclass
class FullBacktestResult:
    """Complete backtest results across all cutoff years.

    This is the top-level result object returned by ``run_full_backtest()``.
    """

    summary: dict = field(default_factory=dict)
    by_year: dict[int, CutoffYearResult] = field(default_factory=dict)
    by_domain: dict[str, dict] = field(default_factory=dict)
    by_method: dict[str, dict] = field(default_factory=dict)
    calibration_factor: float = 1.0
    calibration_direction: str = "none"

    def summary_table(self) -> str:
        """Human-readable summary table."""
        lines = [
            "Exponential Atlas v6 — Backtest Summary",
            "=" * 55,
            "",
            f"{'Cutoff':<10} {'Domains':<10} {'MAPE':>8} {'Log Err':>10} "
            f"{'Bias':<10}",
            f"{'-'*50}",
        ]
        for year in sorted(self.by_year.keys()):
            r = self.by_year[year]
            lines.append(
                f"{year:<10} {r.domains_tested:<10} "
                f"{r.overall_mape:>7.1f}% "
                f"{r.overall_mean_log_error:>+10.3f} "
                f"{r.overall_bias:<10}"
            )
        lines.append("")
        lines.append(f"Overall MAPE: {self.summary.get('overall_mape', 'N/A')}")
        lines.append(
            f"Calibration factor: {self.calibration_factor:.3f} "
            f"({self.calibration_direction})"
        )
        lines.append(
            f"Total comparisons: {self.summary.get('n_comparisons', 0)}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core backtest function for a single cutoff year
# ---------------------------------------------------------------------------

def backtest_at_year(
    domains: dict[str, dict],
    cutoff_year: int,
    fit_module,
    min_pre_points: int = 3,
    min_post_points: int = 1,
) -> CutoffYearResult:
    """Backtest: fit model using only data up to cutoff_year, compare to actuals.

    Parameters
    ----------
    domains : dict[str, dict]
        Mapping of domain_id to domain data dict (as loaded by
        ``loader.load_all_domains()``).
    cutoff_year : int
        Only data with ``year <= cutoff_year`` is used for fitting.
    fit_module
        The fits module or any object with a ``select_best_fit(years, values,
        config)`` function.  In practice, pass ``model.fits`` or the
        ``select_best_fit`` function directly.
    min_pre_points : int
        Minimum data points required before cutoff (default 3).
    min_post_points : int
        Minimum post-cutoff data points required (default 1).

    Returns
    -------
    CutoffYearResult
        Aggregated results for this cutoff year.
    """
    # Resolve fit function
    if hasattr(fit_module, "select_best_fit"):
        select_fn = fit_module.select_best_fit
    elif callable(fit_module):
        select_fn = fit_module
    else:
        raise TypeError(
            "fit_module must have a select_best_fit attribute or be callable."
        )

    result = CutoffYearResult(cutoff_year=cutoff_year)
    all_mapes: list[float] = []
    all_log_errors: list[float] = []

    for domain_id, domain_data in sorted(domains.items()):
        dr = DomainBacktestResult(
            domain_id=domain_id, cutoff_year=cutoff_year
        )

        # Extract data points
        data_points = domain_data.get("data_points", [])
        if not data_points:
            dr.skipped = True
            dr.skip_reason = "No data points"
            result.per_domain[domain_id] = dr
            result.domains_skipped += 1
            continue

        # Split into pre-cutoff and post-cutoff
        pre_years, pre_values = [], []
        post_years, post_values = [], []

        for pt in data_points:
            y = float(pt["year"])
            v = float(pt["value"])
            if y <= cutoff_year:
                pre_years.append(y)
                pre_values.append(v)
            else:
                post_years.append(y)
                post_values.append(v)

        dr.n_pre_cutoff = len(pre_years)
        dr.n_post_cutoff = len(post_years)

        # Check minimum data requirements
        if len(pre_years) < min_pre_points:
            dr.skipped = True
            dr.skip_reason = (
                f"Insufficient pre-cutoff data: {len(pre_years)} < "
                f"{min_pre_points} required"
            )
            result.per_domain[domain_id] = dr
            result.domains_skipped += 1
            continue

        if len(post_years) < min_post_points:
            dr.skipped = True
            dr.skip_reason = (
                f"Insufficient post-cutoff data: {len(post_years)} < "
                f"{min_post_points} required"
            )
            result.per_domain[domain_id] = dr
            result.domains_skipped += 1
            continue

        # Build domain config for fitting
        # We use a simplified config with the pre-cutoff data
        domain_config = _build_domain_config(
            domain_data, pre_years, pre_values
        )

        # Fit on pre-cutoff data
        try:
            fit_result = select_fn(
                np.array(pre_years),
                np.array(pre_values),
                domain_config,
            )
            # Handle ModelSelectionResult (has .best) or direct FitResult
            if hasattr(fit_result, "best"):
                best_fit = fit_result.best
            else:
                best_fit = fit_result

            dr.fit_method = best_fit.method
            dr.r_squared_fit = best_fit.r_squared

        except Exception as exc:
            dr.skipped = True
            dr.skip_reason = f"Fit failed: {exc}"
            result.per_domain[domain_id] = dr
            result.domains_skipped += 1
            continue

        # Predict for post-cutoff years
        pct_errors = []
        log_errors = []

        for y_post, v_actual in zip(post_years, post_values):
            try:
                v_predicted = float(best_fit.predict(y_post))
            except Exception:
                continue

            # Skip invalid predictions
            if not np.isfinite(v_predicted) or v_predicted <= 0:
                continue
            if v_actual <= 0:
                continue

            # Percentage error
            pct_err = abs(v_predicted - v_actual) / abs(v_actual) * 100.0

            # Log error (positive = over-prediction)
            log_err = np.log(v_predicted / v_actual)

            pct_errors.append(pct_err)
            log_errors.append(log_err)

            dr.predictions.append({
                "year": y_post,
                "actual": v_actual,
                "predicted": v_predicted,
                "pct_error": round(pct_err, 2),
                "log_error": round(float(log_err), 4),
            })

        if pct_errors:
            dr.mape = float(np.mean(pct_errors))
            dr.mean_log_error = float(np.mean(log_errors))
            all_mapes.extend(pct_errors)
            all_log_errors.extend(log_errors)

            if dr.mean_log_error > 0.05:
                dr.bias_direction = "over"
            elif dr.mean_log_error < -0.05:
                dr.bias_direction = "under"
            else:
                dr.bias_direction = "neutral"

            result.domains_tested += 1
        else:
            dr.skipped = True
            dr.skip_reason = "No valid predictions could be computed"
            result.domains_skipped += 1

        result.per_domain[domain_id] = dr

    # Aggregate (using median for robustness to outliers)
    if all_mapes:
        result.overall_mape = float(np.median(all_mapes))
        result.overall_mean_log_error = float(np.median(all_log_errors))

        if result.overall_mean_log_error > 0.05:
            result.overall_bias = "over"
        elif result.overall_mean_log_error < -0.05:
            result.overall_bias = "under"
        else:
            result.overall_bias = "neutral"

    return result


def _build_domain_config(
    domain_data: dict,
    pre_years: list[float],
    pre_values: list[float],
) -> dict:
    """Build a simplified domain config dict suitable for select_best_fit.

    This uses the domain's metadata but constrains it to what would be
    available at the cutoff year.
    """
    config: dict = {
        "direction": domain_data.get("direction", "decreasing"),
        "physical_floor": domain_data.get("physical_floor"),
        "physical_ceiling": domain_data.get("physical_ceiling"),
        "best_fit": domain_data.get("best_fit"),
    }

    # Piecewise breakpoint — only include if it falls within pre-cutoff range
    pw = domain_data.get("piecewise", {})
    if pw and isinstance(pw, dict):
        bp = pw.get("breakpoint_year")
        if bp is not None and bp <= max(pre_years):
            config["piecewise_breakpoint"] = bp

    # Wright's Law — only include production data up to the cutoff
    wl = domain_data.get("wrights_law")
    if wl and isinstance(wl, dict):
        cum_prod = wl.get("cumulative_production", [])
        if cum_prod:
            cutoff = max(pre_years)
            filtered_cp = [
                entry for entry in cum_prod
                if float(entry.get("year", 9999)) <= cutoff
            ]
            if len(filtered_cp) >= 3:
                config["wrights_law"] = {
                    "cumulative_production": [
                        float(e["value"]) for e in filtered_cp
                    ],
                    "production_years": [
                        float(e["year"]) for e in filtered_cp
                    ],
                    "prices": pre_values,
                    "learning_rate_hint": wl.get("learning_rate"),
                }

    return config


# ---------------------------------------------------------------------------
# Full backtest across multiple cutoff years
# ---------------------------------------------------------------------------

def run_full_backtest(
    domains: dict[str, dict],
    cutoff_years: Optional[list[int]] = None,
    fit_module=None,
    min_pre_points: int = 3,
    min_post_points: int = 1,
) -> FullBacktestResult:
    """Run backtest at all cutoff years and compute aggregate statistics.

    Parameters
    ----------
    domains : dict[str, dict]
        Mapping of domain_id to domain data dict.
    cutoff_years : list[int], optional
        Years at which to cut off data.  Default: ``[2005, 2010, 2015, 2020]``.
    fit_module
        The fits module (must have ``select_best_fit``).  If ``None``,
        attempts to import ``model.fits``.
    min_pre_points : int
        Minimum pre-cutoff data points per domain.
    min_post_points : int
        Minimum post-cutoff data points per domain.

    Returns
    -------
    FullBacktestResult
        Complete backtest results.

    Notes
    -----
    The calibration factor is computed as ``exp(-mean_log_error)`` from
    all backtest comparisons.  If the model systematically over-predicts
    by 15% (mean log error = +0.14), the calibration factor would be
    ~0.87, meaning we should multiply forward projections by 0.87.
    """
    if cutoff_years is None:
        cutoff_years = [2005, 2010, 2015, 2020]

    if fit_module is None:
        try:
            from model import fits as fit_module
        except ImportError:
            raise ImportError(
                "Could not import model.fits. Pass fit_module explicitly."
            )

    full_result = FullBacktestResult()

    # --- Run backtest at each cutoff year ---
    all_mapes: list[float] = []
    all_log_errors: list[float] = []
    n_comparisons = 0

    for year in sorted(cutoff_years):
        year_result = backtest_at_year(
            domains=domains,
            cutoff_year=year,
            fit_module=fit_module,
            min_pre_points=min_pre_points,
            min_post_points=min_post_points,
        )
        full_result.by_year[year] = year_result

        # Collect all comparisons for overall statistics
        for dr in year_result.per_domain.values():
            if not dr.skipped and dr.predictions:
                for pred in dr.predictions:
                    all_mapes.append(pred["pct_error"])
                    all_log_errors.append(pred["log_error"])
                    n_comparisons += 1

    # --- Aggregate by domain ---
    domain_stats: dict[str, dict] = {}
    for year in sorted(cutoff_years):
        yr = full_result.by_year[year]
        for did, dr in yr.per_domain.items():
            if dr.skipped:
                continue
            if did not in domain_stats:
                domain_stats[did] = {
                    "mapes": [],
                    "log_errors": [],
                    "methods_used": [],
                    "n_cutoffs_tested": 0,
                }
            if np.isfinite(dr.mape):
                domain_stats[did]["mapes"].append(dr.mape)
                domain_stats[did]["log_errors"].append(dr.mean_log_error)
                domain_stats[did]["methods_used"].append(dr.fit_method)
                domain_stats[did]["n_cutoffs_tested"] += 1

    for did, stats in domain_stats.items():
        if stats["mapes"]:
            avg_mape = float(np.mean(stats["mapes"]))
            avg_log_err = float(np.mean(stats["log_errors"]))
            full_result.by_domain[did] = {
                "avg_mape": round(avg_mape, 2),
                "avg_log_error": round(avg_log_err, 4),
                "n_cutoffs_tested": stats["n_cutoffs_tested"],
                "methods_used": list(set(stats["methods_used"])),
                "bias": (
                    "over" if avg_log_err > 0.05
                    else "under" if avg_log_err < -0.05
                    else "neutral"
                ),
            }

    # --- Aggregate by method ---
    method_stats: dict[str, dict] = {}
    for year in sorted(cutoff_years):
        yr = full_result.by_year[year]
        for dr in yr.per_domain.values():
            if dr.skipped or not dr.predictions:
                continue
            m = dr.fit_method
            if m not in method_stats:
                method_stats[m] = {"mapes": [], "log_errors": [], "count": 0}
            method_stats[m]["mapes"].append(dr.mape)
            method_stats[m]["log_errors"].append(dr.mean_log_error)
            method_stats[m]["count"] += 1

    for method, stats in method_stats.items():
        if stats["mapes"]:
            full_result.by_method[method] = {
                "avg_mape": round(float(np.mean(stats["mapes"])), 2),
                "avg_log_error": round(float(np.mean(stats["log_errors"])), 4),
                "n_domains": stats["count"],
                "bias": (
                    "over" if np.mean(stats["log_errors"]) > 0.05
                    else "under" if np.mean(stats["log_errors"]) < -0.05
                    else "neutral"
                ),
            }

    # --- Overall summary ---
    if all_mapes:
        mape_arr = np.array(all_mapes)
        log_err_arr = np.array(all_log_errors)

        # Use MEDIAN MAPE as the primary metric — robust to outliers.
        # A single domain with wildly wrong extrapolation (e.g., fusion_q
        # predicting 237M when actual was 0.33) produces billion-percent
        # MAPE that destroys arithmetic mean. Median is the honest metric.
        median_mape = float(np.median(mape_arr))

        # Also compute trimmed mean (5th-95th percentile) for a
        # mean-like metric that's robust to extreme outliers.
        p5, p95 = np.percentile(mape_arr, [5, 95])
        trimmed = mape_arr[(mape_arr >= p5) & (mape_arr <= p95)]
        trimmed_mean_mape = float(np.mean(trimmed)) if len(trimmed) > 0 else median_mape

        # Keep raw mean for full transparency
        raw_mean_mape = float(np.mean(mape_arr))

        # Log error uses median too (more robust for calibration)
        median_log_err = float(np.median(log_err_arr))
        mean_log_err = float(np.mean(log_err_arr))

        overall_bias = (
            "over" if median_log_err > 0.05
            else "under" if median_log_err < -0.05
            else "neutral"
        )

        # Calibration factor from MEDIAN log error (robust to outliers)
        calibration_factor = float(np.exp(-median_log_err))

        full_result.summary = {
            "overall_mape": round(median_mape, 2),
            "overall_mape_trimmed_mean": round(trimmed_mean_mape, 2),
            "overall_mape_raw_mean": round(raw_mean_mape, 2),
            "overall_median_log_error": round(median_log_err, 4),
            "overall_mean_log_error": round(mean_log_err, 4),
            "n_comparisons": n_comparisons,
            "n_domains_tested": len(domain_stats),
            "n_cutoff_years": len(cutoff_years),
            "bias_direction": overall_bias,
            "note": (
                "overall_mape uses MEDIAN (robust to outliers). "
                "Raw arithmetic mean is also reported for transparency."
            ),
        }
        full_result.calibration_factor = round(calibration_factor, 4)
        full_result.calibration_direction = overall_bias
    else:
        full_result.summary = {
            "overall_mape": None,
            "overall_mean_log_error": None,
            "n_comparisons": 0,
            "n_domains_tested": 0,
            "n_cutoff_years": len(cutoff_years),
            "bias_direction": "insufficient_data",
        }

    return full_result
