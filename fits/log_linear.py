"""
Exponential Atlas v6 — Log-linear fit.

Model
-----
The most basic exponential trend model.  We assume that the *logarithm* of
the measured value evolves linearly in time:

    ln(value) = slope * year + intercept

so on the natural scale:

    value(year) = exp(intercept) * exp(slope * year)

For **decreasing** domains (e.g. cost in $/kWh) ``slope < 0``;
for **growing** domains (e.g. FLOP/s per $) ``slope > 0``.

Fitting is done via ``scipy.stats.linregress`` on log-transformed values,
which is equivalent to ordinary least-squares in log-space.
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


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_log_linear(
    year: float | np.ndarray,
    slope: float,
    intercept: float,
) -> float | np.ndarray:
    """Predict value at *year* given log-linear parameters.

    .. math::

        \\hat{v}(t) = \\exp(\\text{intercept} + \\text{slope} \\cdot t)

    Parameters
    ----------
    year : float or array-like
        Year(s) to predict at.
    slope : float
        Fitted slope in log-space (per year).
    intercept : float
        Fitted intercept in log-space.

    Returns
    -------
    float or np.ndarray
        Predicted value(s) on the natural scale.
    """
    year = np.asarray(year, dtype=float)
    result = np.exp(intercept + slope * year)
    # Return scalar if input was scalar
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Fit function
# ---------------------------------------------------------------------------
def fit_log_linear(
    years: list | np.ndarray,
    values: list | np.ndarray,
) -> FitResult:
    """Fit a log-linear (exponential) model to *(years, values)*.

    Parameters
    ----------
    years : array-like
        Independent variable (calendar years, possibly fractional).
    values : array-like
        Observed values.  Must be positive (we take the logarithm).

    Returns
    -------
    FitResult
        Fully populated result including AIC, BIC, predict function,
        and residuals (in log-space, where the fitting happens).

    Raises
    ------
    ValueError
        If fewer than 2 data points are provided.
    """
    y = np.asarray(years, dtype=float)
    v = np.asarray(values, dtype=float)

    if len(y) < 2:
        raise ValueError(
            f"fit_log_linear requires >= 2 data points, got {len(y)}"
        )

    log_v = _safe_log(v)

    # OLS in log-space
    slope, intercept, r_value, p_value, std_err = linregress(y, log_v)
    r_squared = float(r_value ** 2)

    # Residuals in log-space (the domain where we actually minimised RSS)
    predicted_log = intercept + slope * y
    residuals = log_v - predicted_log
    rss = float(np.sum(residuals ** 2))

    n = len(y)
    k = 2  # slope + intercept

    aic = compute_aic(n, k, rss)
    bic = compute_bic(n, k, rss)

    # Extrapolation warning
    year_min, year_max = float(y.min()), float(y.max())
    warning = (
        f"Model fitted on years [{year_min:.1f}, {year_max:.1f}]. "
        f"Predictions outside this range are extrapolations."
    )

    # Build predict closure
    _slope, _intercept = slope, intercept

    def _predict(year):
        return predict_log_linear(year, _slope, _intercept)

    return FitResult(
        method="log_linear",
        params={
            "slope": float(slope),
            "intercept": float(intercept),
            "std_err": float(std_err),
            "p_value": float(p_value),
            "year_min": year_min,
            "year_max": year_max,
            "halving_time": float(-np.log(2) / slope) if slope < 0 else None,
            "doubling_time": float(np.log(2) / slope) if slope > 0 else None,
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
