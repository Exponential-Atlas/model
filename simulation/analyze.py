"""
Exponential Atlas v6 -- Domain Analysis
========================================
Runs full analysis on all loaded domains: fitting, projection,
acceleration detection.  Replaces v5's analyze_all() function.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from model.data.loader import load_all_domains, get_domain_data_points, get_wrights_law_data
from model.data.domain_registry import (
    SIMULATION_DOMAINS,
    SIM_TO_DATA_MAP,
    DATA_TO_SIM_MAP,
    ALL_DATA_DOMAINS,
)
from model.fits.model_selection import select_best_fit, ModelSelectionResult
from model.fits.base import FitResult


def _detect_acceleration(
    fit: FitResult,
    years: np.ndarray,
    values: np.ndarray,
) -> dict:
    """Detect whether the domain is accelerating, decelerating, or steady.

    Compares the slope in the early half vs the late half of the data.
    For piecewise fits, uses the pre/post breakpoint slopes directly.

    Parameters
    ----------
    fit : FitResult
        The best fit result.
    years : np.ndarray
        Data years.
    values : np.ndarray
        Data values.

    Returns
    -------
    dict
        {
            "status": "accelerating" | "decelerating" | "steady",
            "early_slope": float,
            "late_slope": float,
            "acceleration_ratio": float,  # late/early (>1 = accelerating)
            "note": str
        }
    """
    if fit.method == "piecewise":
        early = abs(fit.params.get("slope_pre", 0))
        late = abs(fit.params.get("slope_post", 0))
    else:
        # Split data in half and compare log-space slopes
        n = len(years)
        mid = n // 2
        if mid < 2 or (n - mid) < 2:
            return {
                "status": "insufficient_data",
                "early_slope": 0,
                "late_slope": 0,
                "acceleration_ratio": 1.0,
                "note": "Not enough data to detect acceleration",
            }

        log_v = np.log(np.maximum(values, 1e-30))

        # Early half
        from scipy.stats import linregress
        early_slope, _, _, _, _ = linregress(years[:mid], log_v[:mid])
        early = abs(early_slope)

        # Late half
        late_slope, _, _, _, _ = linregress(years[mid:], log_v[mid:])
        late = abs(late_slope)

    if early < 1e-10:
        ratio = float("inf") if late > 1e-10 else 1.0
    else:
        ratio = late / early

    if ratio > 1.5:
        status = "accelerating"
        note = f"Late-period slope is {ratio:.1f}x the early-period slope"
    elif ratio < 0.67:
        status = "decelerating"
        note = f"Late-period slope is {ratio:.1f}x the early-period slope (slowing)"
    else:
        status = "steady"
        note = "Rate of change is roughly constant across the data range"

    return {
        "status": status,
        "early_slope": round(float(early), 6),
        "late_slope": round(float(late), 6),
        "acceleration_ratio": round(float(ratio), 3),
        "note": note,
    }


def _compute_projections(
    fit: FitResult,
    projection_years: list[int],
) -> dict[int, float]:
    """Compute projected values at future years.

    Parameters
    ----------
    fit : FitResult
        The best fit.
    projection_years : list[int]
        Years to project to (e.g., [2026, 2030, 2035, 2039]).

    Returns
    -------
    dict[int, float]
        Year -> projected value.
    """
    projections = {}

    for year in projection_years:
        try:
            # For Wright's Law, try predict_by_year first
            if fit.method == "wrights_law":
                predict_by_year = fit.params.get("predict_by_year")
                if predict_by_year is not None:
                    projections[year] = float(predict_by_year(year))
                    continue

            # All other methods: predict takes year directly
            projections[year] = float(fit.predict(year))
        except Exception:
            projections[year] = None

    return projections


def analyze_domain(
    domain_id: str,
    domain_data: dict,
    projection_years: list[int] | None = None,
) -> dict:
    """Run full analysis on a single data domain.

    Parameters
    ----------
    domain_id : str
        Domain identifier.
    domain_data : dict
        Loaded domain JSON data.
    projection_years : list[int], optional
        Years to project to.  Defaults to [2026, 2030, 2035, 2039].

    Returns
    -------
    dict
        Comprehensive analysis including fit results, projections,
        acceleration detection, and metadata.
    """
    if projection_years is None:
        projection_years = [2026, 2030, 2035, 2039]

    years, values = get_domain_data_points(domain_data)
    if len(years) < 2:
        return {
            "domain_id": domain_id,
            "error": "Insufficient data points",
            "n_points": len(years),
        }

    years_arr = np.array(years)
    values_arr = np.array(values)

    # Build domain config for model selection
    domain_config = {
        "direction": "d" if domain_data.get("direction") == "decreasing" else "g",
        "physical_floor": domain_data.get("physical_floor"),
        "physical_ceiling": domain_data.get("physical_ceiling"),
        "best_fit": domain_data.get("best_fit"),
    }

    # Wright's Law config
    wl = domain_data.get("wrights_law")
    if wl and wl.get("cumulative_production"):
        wl_data = get_wrights_law_data(domain_data)
        if wl_data is not None:
            wl_years, wl_cum_prod, wl_prices = wl_data
            domain_config["wrights_law"] = {
                "prices": wl_prices,
                "cumulative_production": wl_cum_prod,
                "learning_rate_hint": wl.get("learning_rate"),
                "production_years": wl_years,
            }

    # Piecewise config
    pw = domain_data.get("piecewise")
    if pw and isinstance(pw, dict):
        domain_config["piecewise_breakpoint"] = pw.get("breakpoint_year")

    # Run model selection
    try:
        selection = select_best_fit(years, values, domain_config)
    except Exception as exc:
        return {
            "domain_id": domain_id,
            "error": f"Model selection failed: {exc}",
            "n_points": len(years),
        }

    best_fit = selection.best

    # Projections
    projections = _compute_projections(best_fit, projection_years)

    # Acceleration detection
    acceleration = _detect_acceleration(best_fit, years_arr, values_arr)

    # Compute annual improvement rate (for display)
    slope = abs(best_fit.params.get("slope", best_fit.params.get("slope_post", 0)))
    annual_improvement = math.exp(slope) if slope > 0 else 1.0

    return {
        "domain_id": domain_id,
        "name": domain_data.get("name", domain_id),
        "category": domain_data.get("category", "Unknown"),
        "direction": domain_data.get("direction", "unknown"),
        "confidence": domain_data.get("confidence", "low"),
        "n_points": len(years),
        "year_range": [float(years_arr.min()), float(years_arr.max())],
        "value_range": [float(values_arr.min()), float(values_arr.max())],
        "sim_domain": DATA_TO_SIM_MAP.get(domain_id, "unknown"),

        # Fit results
        "best_fit_method": best_fit.method,
        "r_squared": round(best_fit.r_squared, 4),
        "aic": round(best_fit.aic, 2),
        "bic": round(best_fit.bic, 2),
        "fit_params": {
            k: v for k, v in best_fit.params.items()
            if not callable(v)  # Exclude predict functions
        },
        "all_fits_summary": [
            {
                "method": f.method,
                "r_squared": round(f.r_squared, 4),
                "bic": round(f.bic, 2),
            }
            for f in selection.all_fits
        ],
        "selection_notes": selection.notes,

        # Derived metrics
        "annual_improvement_factor": round(annual_improvement, 4),
        "projections": projections,
        "acceleration": acceleration,
    }


def analyze_domains(
    domains: dict[str, dict] | None = None,
    projection_years: list[int] | None = None,
) -> dict:
    """Run full analysis on all loaded domains.

    For each domain:
    1. Load data points
    2. Run model selection (best fit by BIC)
    3. Compute projections from the best fit
    4. Detect acceleration (early vs late slope comparison)
    5. Return comprehensive analysis dict

    This replaces v5's analyze_all() function.

    Parameters
    ----------
    domains : dict[str, dict], optional
        Mapping of domain_id -> domain data.  If None, loads all domains
        from disk.
    projection_years : list[int], optional
        Years to project to.  Defaults to [2026, 2030, 2035, 2039].

    Returns
    -------
    dict
        {
            "domains": dict[str, analysis_dict],
            "summary": {
                "total_domains": int,
                "successful_fits": int,
                "failed_fits": int,
                "accelerating_count": int,
                "mean_r_squared": float,
            },
            "sim_domain_rates": dict[str, float],
        }
    """
    if domains is None:
        domains = load_all_domains(validate=False)

    if projection_years is None:
        projection_years = [2026, 2030, 2035, 2039]

    results = {}
    successful = 0
    failed = 0
    accelerating = 0
    r_squared_sum = 0.0

    for domain_id, domain_data in sorted(domains.items()):
        analysis = analyze_domain(domain_id, domain_data, projection_years)
        results[domain_id] = analysis

        if "error" in analysis:
            failed += 1
        else:
            successful += 1
            r_squared_sum += analysis.get("r_squared", 0)
            if analysis.get("acceleration", {}).get("status") == "accelerating":
                accelerating += 1

    mean_r2 = r_squared_sum / successful if successful > 0 else 0

    return {
        "domains": results,
        "summary": {
            "total_domains": len(domains),
            "successful_fits": successful,
            "failed_fits": failed,
            "accelerating_count": accelerating,
            "mean_r_squared": round(mean_r2, 4),
        },
    }
