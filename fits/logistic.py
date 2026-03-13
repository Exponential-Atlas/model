"""
Exponential Atlas v6 — Logistic (S-curve) fit.

Model
-----
Some technology metrics approach a **physical limit**: VR resolution
caps at ~120 ppd (retinal limit), compute efficiency has thermodynamic
bounds, etc.  For these domains, exponential extrapolation is wrong —
the trend must saturate.

The logistic (sigmoid) model captures saturation:

**Growing domains** (value increases toward a ceiling):

    value(t) = floor + (ceiling - floor) / (1 + exp(-k * (t - t_mid)))

**Decreasing domains** (value decreases toward a floor):

    value(t) = ceiling - (ceiling - floor) / (1 + exp(-k * (t - t_mid)))

    equivalently:
    value(t) = floor + (ceiling - floor) / (1 + exp(+k * (t - t_mid)))

Parameters:
    floor     — lower asymptote
    ceiling   — upper asymptote
    k         — steepness (growth rate)
    t_mid     — inflection point (year of fastest change)

Fitting is done via ``scipy.optimize.curve_fit`` (Levenberg-Marquardt
with bounds) on the **natural scale** — not log-transformed, since the
logistic is inherently bounded and log-transform would distort the
asymptotic behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from .base import (
    FitResult,
    compute_aic,
    compute_bic,
    _r_squared,
)


# ---------------------------------------------------------------------------
# Logistic model functions
# ---------------------------------------------------------------------------
def _logistic_growing(t, floor, ceiling, k, t_mid):
    """Standard logistic for growing domains."""
    return floor + (ceiling - floor) / (1.0 + np.exp(-k * (t - t_mid)))


def _logistic_decreasing(t, floor, ceiling, k, t_mid):
    """Logistic for decreasing domains (falls from ceiling to floor)."""
    return floor + (ceiling - floor) / (1.0 + np.exp(k * (t - t_mid)))


# ---------------------------------------------------------------------------
# Fit function
# ---------------------------------------------------------------------------
def fit_logistic(
    years: list | np.ndarray,
    values: list | np.ndarray,
    ceiling: float | None = None,
    floor: float | None = None,
) -> FitResult:
    """Fit a logistic (S-curve) model to *(years, values)*.

    The function auto-detects whether the domain is **growing** or
    **decreasing** based on whether the last value is larger or smaller
    than the first.

    Parameters
    ----------
    years : array-like
        Calendar years.
    values : array-like
        Observed metric values.  Must be positive.
    ceiling : float, optional
        Known upper asymptote (physical limit).  If ``None``, the ceiling
        is estimated from data — but a warning is emitted because this is
        often unstable with sparse data.
    floor : float, optional
        Known lower asymptote.  If ``None``, estimated from data.

    Returns
    -------
    FitResult
        The ``predict`` callable takes a year (or array of years).

    Raises
    ------
    ValueError
        If fewer than 4 data points are provided (4 parameters to fit).
    RuntimeError
        If curve_fit fails to converge.
    """
    y = np.asarray(years, dtype=float)
    v = np.asarray(values, dtype=float)

    if len(y) < 4:
        raise ValueError(
            f"fit_logistic requires >= 4 data points, got {len(y)}"
        )

    # Sort by year
    order = np.argsort(y)
    y = y[order]
    v = v[order]

    # Detect direction
    decreasing = v[-1] < v[0]

    # --- Determine initial guesses and bounds ---
    v_min, v_max = float(v.min()), float(v.max())
    v_range = v_max - v_min if v_max > v_min else 1.0
    t_range = float(y[-1] - y[0]) if y[-1] > y[0] else 1.0

    estimated_ceiling = False
    estimated_floor = False

    if decreasing:
        # Decreasing: ceiling ~ max observed, floor ~ below min observed
        if floor is None:
            floor_est = max(v_min * 0.1, v_min - 0.5 * v_range)
            if floor_est <= 0:
                floor_est = v_min * 0.01
            estimated_floor = True
        else:
            floor_est = floor

        if ceiling is None:
            ceiling_est = v_max * 1.5
            estimated_ceiling = True
        else:
            ceiling_est = ceiling

        # Initial guesses
        k0 = 2.0 / t_range  # reasonable steepness
        t_mid0 = float(np.median(y))

        # Bounds
        floor_lo = 0.0
        floor_hi = v_min * 1.1 if not estimated_floor else v_max
        ceil_lo = v_max * 0.5
        ceil_hi = v_max * 10.0 if estimated_ceiling else ceiling_est * 2.0

        p0 = [floor_est, ceiling_est, k0, t_mid0]
        bounds_lo = [floor_lo, ceil_lo, 1e-6, y[0] - 50]
        bounds_hi = [floor_hi, ceil_hi, 10.0, y[-1] + 50]

        model_fn = _logistic_decreasing

    else:
        # Growing: floor ~ below min, ceiling ~ above max
        if floor is None:
            floor_est = max(0.0, v_min - 0.5 * v_range)
            estimated_floor = True
        else:
            floor_est = floor

        if ceiling is None:
            ceiling_est = v_max * 3.0
            estimated_ceiling = True
        else:
            ceiling_est = ceiling

        k0 = 2.0 / t_range
        t_mid0 = float(np.median(y))

        floor_lo = 0.0
        floor_hi = v_min * 1.1 if (not estimated_floor and v_min > 0) else v_max
        ceil_lo = v_max * 0.8
        ceil_hi = ceiling_est * 10.0 if estimated_ceiling else ceiling_est * 2.0

        p0 = [floor_est, ceiling_est, k0, t_mid0]
        bounds_lo = [floor_lo, ceil_lo, 1e-6, y[0] - 50]
        bounds_hi = [floor_hi, ceil_hi, 10.0, y[-1] + 50]

        model_fn = _logistic_growing

    # Sanitize bounds: ensure lo < hi for every parameter
    for i in range(len(bounds_lo)):
        if bounds_lo[i] >= bounds_hi[i]:
            bounds_hi[i] = bounds_lo[i] + abs(bounds_lo[i]) * 0.1 + 1.0
        # Clamp initial guess to bounds
        p0[i] = max(bounds_lo[i] + 1e-10, min(p0[i], bounds_hi[i] - 1e-10))

    # --- Fit ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                model_fn,
                y,
                v,
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=10000,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Logistic fit failed to converge: {exc}. "
            f"Try providing explicit ceiling/floor values."
        ) from exc

    fitted_floor, fitted_ceiling, fitted_k, fitted_t_mid = popt

    # --- Diagnostics ---
    predicted = model_fn(y, *popt)
    residuals = v - predicted
    rss = float(np.sum(residuals ** 2))
    r_squared = _r_squared(v, predicted)

    n = len(y)
    # Count free parameters: only count estimated ones
    k_params = 2  # k and t_mid are always fitted
    if estimated_ceiling:
        k_params += 1
    else:
        pass  # ceiling was fixed — not a free parameter
    if estimated_floor:
        k_params += 1
    else:
        pass  # floor was fixed
    # For AIC/BIC fairness, always count all params that curve_fit optimised
    # (which is always 4) — this is the honest count.
    k_params = 4

    aic = compute_aic(n, k_params, rss)
    bic = compute_bic(n, k_params, rss)

    # Warnings
    warn_parts = []
    year_min, year_max = float(y.min()), float(y.max())
    warn_parts.append(
        f"Logistic model fitted on [{year_min:.1f}, {year_max:.1f}]."
    )
    if estimated_ceiling:
        warn_parts.append(
            f"Ceiling was estimated from data ({fitted_ceiling:.4g}) — "
            f"this is often unreliable with sparse observations. "
            f"Provide an explicit physical ceiling for better results."
        )
    if estimated_floor:
        warn_parts.append(
            f"Floor was estimated from data ({fitted_floor:.4g})."
        )
    extrap_warning = " ".join(warn_parts)

    # Predict closure
    _popt = tuple(popt)
    _model_fn = model_fn

    def _predict(year):
        year = np.asarray(year, dtype=float)
        scalar = year.ndim == 0
        year = np.atleast_1d(year)
        result = _model_fn(year, *_popt)
        return float(result[0]) if scalar else result

    return FitResult(
        method="logistic",
        params={
            "floor": float(fitted_floor),
            "ceiling": float(fitted_ceiling),
            "k": float(fitted_k),
            "t_mid": float(fitted_t_mid),
            "direction": "decreasing" if decreasing else "growing",
            "ceiling_estimated": estimated_ceiling,
            "floor_estimated": estimated_floor,
            "year_min": year_min,
            "year_max": year_max,
        },
        r_squared=r_squared,
        aic=aic,
        bic=bic,
        residuals=residuals,
        n_params=k_params,
        n_points=n,
        predict=_predict,
        extrapolation_warning=extrap_warning,
    )
