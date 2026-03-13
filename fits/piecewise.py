"""
Exponential Atlas v6 — Piecewise log-linear fit with automated breakpoint
detection.

Model
-----
Some technology domains experience a *regime change* — a discontinuity in
their exponential trend.  Genome sequencing is the canonical example: it
followed one cost curve until ~2008, then next-gen sequencing created a
much steeper decline.

We model this as two separate log-linear regressions joined at a
breakpoint year ``t_bp``:

    For t <= t_bp:   ln(value) = slope_1 * t + intercept_1
    For t >  t_bp:   ln(value) = slope_2 * t + intercept_2

The two segments are fitted **independently** (no continuity constraint),
because real technology discontinuities are exactly that — discontinuous.

Breakpoint detection
--------------------
If no ``breakpoint_hint`` is given, we search exhaustively over candidate
breakpoints (every observed year from the 3rd to the (n−3)rd) and select
the one that minimises the total RSS (sum of RSS from both segments).

This is equivalent to a maximum-likelihood search for a single change-point
in a piecewise-linear-in-log model.

The model has 5 parameters: slope_1, intercept_1, slope_2, intercept_2,
and the breakpoint year.  AIC/BIC are computed accordingly.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress

from .base import (
    FitResult,
    compute_aic,
    compute_bic,
    _safe_log,
    _r_squared,
)


def _fit_segment(y: np.ndarray, log_v: np.ndarray):
    """Fit OLS on a single segment.  Returns (slope, intercept, rss)."""
    if len(y) < 2:
        return None, None, np.inf
    slope, intercept, r_value, p_value, std_err = linregress(y, log_v)
    predicted = intercept + slope * y
    rss = float(np.sum((log_v - predicted) ** 2))
    return slope, intercept, rss


def _search_breakpoint(
    y: np.ndarray,
    log_v: np.ndarray,
    min_segment: int = 3,
) -> tuple[float, float]:
    """Find the breakpoint year that minimises total RSS.

    We try every observed year that leaves at least ``min_segment`` points
    on each side.

    Returns
    -------
    best_bp : float
        Optimal breakpoint year.
    best_rss : float
        Combined RSS at the optimal breakpoint.
    """
    n = len(y)
    best_bp = None
    best_rss = np.inf

    # Candidates: indices from min_segment to n - min_segment (inclusive)
    # The breakpoint is placed such that points with year <= bp go to
    # segment 1 and points with year > bp go to segment 2.
    for idx in range(min_segment, n - min_segment + 1):
        # breakpoint between y[idx-1] and y[idx]
        # we use y[idx-1] as the breakpoint value (last point of segment 1)
        bp_candidate = y[idx - 1]

        mask_left = y <= bp_candidate
        mask_right = y > bp_candidate

        if mask_left.sum() < 2 or mask_right.sum() < 2:
            continue

        _, _, rss_left = _fit_segment(y[mask_left], log_v[mask_left])
        _, _, rss_right = _fit_segment(y[mask_right], log_v[mask_right])
        total_rss = rss_left + rss_right

        if total_rss < best_rss:
            best_rss = total_rss
            best_bp = bp_candidate

    # Also try midpoints between consecutive years for finer resolution
    for idx in range(min_segment, n - min_segment + 1):
        if idx < n:
            bp_candidate = (y[idx - 1] + y[idx]) / 2.0
        else:
            continue

        mask_left = y <= bp_candidate
        mask_right = y > bp_candidate

        if mask_left.sum() < 2 or mask_right.sum() < 2:
            continue

        _, _, rss_left = _fit_segment(y[mask_left], log_v[mask_left])
        _, _, rss_right = _fit_segment(y[mask_right], log_v[mask_right])
        total_rss = rss_left + rss_right

        if total_rss < best_rss:
            best_rss = total_rss
            best_bp = bp_candidate

    if best_bp is None:
        # Fallback: midpoint
        best_bp = float(np.median(y))
        best_rss = np.inf

    return float(best_bp), float(best_rss)


# ---------------------------------------------------------------------------
# Fit function
# ---------------------------------------------------------------------------
def fit_piecewise(
    years: list | np.ndarray,
    values: list | np.ndarray,
    breakpoint_hint: float | None = None,
) -> FitResult:
    """Fit a piecewise log-linear model with one breakpoint.

    Parameters
    ----------
    years : array-like
        Calendar years (possibly fractional).
    values : array-like
        Observed values.  Must be positive.
    breakpoint_hint : float, optional
        If given, used directly as the breakpoint.  Otherwise the optimal
        breakpoint is found by exhaustive search over candidate years.

    Returns
    -------
    FitResult
        The ``predict`` callable takes a year (or array of years) and
        uses the appropriate segment's parameters.

    Raises
    ------
    ValueError
        If fewer than 6 data points are provided (need >= 3 per side).
    """
    y = np.asarray(years, dtype=float)
    v = np.asarray(values, dtype=float)

    if len(y) < 6:
        raise ValueError(
            f"fit_piecewise requires >= 6 data points, got {len(y)}"
        )

    # Sort by year
    order = np.argsort(y)
    y = y[order]
    v = v[order]

    log_v = _safe_log(v)

    # --- Determine breakpoint ---
    if breakpoint_hint is not None:
        bp = float(breakpoint_hint)
    else:
        bp, _ = _search_breakpoint(y, log_v, min_segment=3)

    # --- Fit both segments ---
    mask_left = y <= bp
    mask_right = y > bp

    # Fallback: if one side has < 2 points, adjust breakpoint
    if mask_left.sum() < 2:
        bp = float(y[1])
        mask_left = y <= bp
        mask_right = y > bp
    if mask_right.sum() < 2:
        bp = float(y[-3])
        mask_left = y <= bp
        mask_right = y > bp

    s1, i1, rss1 = _fit_segment(y[mask_left], log_v[mask_left])
    s2, i2, rss2 = _fit_segment(y[mask_right], log_v[mask_right])

    if s1 is None or s2 is None:
        raise ValueError(
            "Could not fit both segments — not enough points on one side "
            f"of breakpoint {bp:.1f}."
        )

    # Combined residuals
    pred_left = i1 + s1 * y[mask_left]
    pred_right = i2 + s2 * y[mask_right]
    residuals_left = log_v[mask_left] - pred_left
    residuals_right = log_v[mask_right] - pred_right
    residuals = np.concatenate([residuals_left, residuals_right])

    total_rss = float(np.sum(residuals ** 2))

    # R² over the entire dataset
    predicted_all = np.empty_like(log_v)
    predicted_all[mask_left] = pred_left
    predicted_all[mask_right] = pred_right
    r_squared = _r_squared(log_v, predicted_all)

    n = len(y)
    k = 5  # slope_1, intercept_1, slope_2, intercept_2, breakpoint

    aic = compute_aic(n, k, total_rss)
    bic = compute_bic(n, k, total_rss)

    year_min, year_max = float(y.min()), float(y.max())
    warning = (
        f"Piecewise model with breakpoint at {bp:.1f}. "
        f"Fitted on [{year_min:.1f}, {year_max:.1f}]. "
        f"Predictions beyond this range use the post-break segment's slope."
    )

    # Predict closure — uses post-break slope for extrapolation
    _bp = bp
    _s1, _i1, _s2, _i2 = s1, i1, s2, i2

    def _predict(year):
        year = np.asarray(year, dtype=float)
        scalar = year.ndim == 0
        year = np.atleast_1d(year)
        result = np.empty_like(year)
        left = year <= _bp
        right = ~left
        result[left] = np.exp(_i1 + _s1 * year[left])
        result[right] = np.exp(_i2 + _s2 * year[right])
        return float(result[0]) if scalar else result

    return FitResult(
        method="piecewise",
        params={
            "breakpoint": bp,
            "slope_pre": float(s1),
            "intercept_pre": float(i1),
            "slope_post": float(s2),
            "intercept_post": float(i2),
            "n_left": int(mask_left.sum()),
            "n_right": int(mask_right.sum()),
            "rss_left": float(rss1),
            "rss_right": float(rss2),
            "year_min": year_min,
            "year_max": year_max,
            "halving_time_post": (
                float(-np.log(2) / s2) if s2 < 0 else None
            ),
            "doubling_time_post": (
                float(np.log(2) / s2) if s2 > 0 else None
            ),
        },
        r_squared=r_squared,
        aic=aic,
        bic=bic,
        residuals=residuals,
        n_params=k,
        n_points=n,
        predict=_predict,
        extrapolation_warning=warning,
    )
