"""
Exponential Atlas v6 — Wright's Law fit.

Model
-----
Wright's Law (1936) states that unit cost declines as a power of
cumulative production:

    cost = c0 * cumulative_production ^ (-alpha)

The *learning rate* is the fractional cost reduction per doubling of
cumulative production:

    learning_rate = 1 - 2^(-alpha)

For example, a learning rate of 0.20 means a 20 % cost reduction every
time cumulative production doubles — which is the historic rate for solar PV.

Fitting is done by OLS on the log-log transform:

    ln(cost) = ln(c0) - alpha * ln(cumulative_production)

For projecting into the future we also need to model how cumulative
production grows over time.  We fit a secondary log-linear regression
on ``(year, ln(cumulative_production))`` so that we can convert a
target year into a projected cumulative production, then feed that
into the Wright's Law curve.
"""

from __future__ import annotations

import warnings

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
def predict_wrights_law(
    cumulative_production: float | np.ndarray,
    alpha: float,
    c0: float,
) -> float | np.ndarray:
    """Predict cost at a given cumulative production level.

    .. math::

        \\hat{c}(Q) = c_0 \\, Q^{-\\alpha}

    Parameters
    ----------
    cumulative_production : float or array-like
        Cumulative production (same units as the training data).
    alpha : float
        Learning exponent (positive means cost decreases with production).
    c0 : float
        Normalisation constant — notional cost at ``Q = 1``.

    Returns
    -------
    float or np.ndarray
        Predicted cost on the natural scale.
    """
    q = np.asarray(cumulative_production, dtype=float)
    # Guard against non-positive production
    q = np.where(q > 0, q, np.finfo(float).tiny)
    result = c0 * q ** (-alpha)
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Fit function
# ---------------------------------------------------------------------------
def fit_wrights_law(
    prices: list | np.ndarray,
    cumulative_production: list | np.ndarray,
    learning_rate_hint: float | None = None,
    *,
    production_years: list | np.ndarray | None = None,
) -> FitResult:
    """Fit Wright's Law to observed (price, cumulative_production) pairs.

    Parameters
    ----------
    prices : array-like
        Observed unit costs / prices.
    cumulative_production : array-like
        Cumulative production corresponding to each price observation.
    learning_rate_hint : float, optional
        A prior learning rate (e.g. 0.20 for solar).  Currently used only
        for diagnostics — the fit is always data-driven.
    production_years : array-like, optional
        Calendar years matching each cumulative production value.  If
        provided, a secondary log-linear fit of cumulative production
        over time is included in the result so that ``predict`` can
        accept a *year* as well.

    Returns
    -------
    FitResult
        The ``predict`` callable takes **cumulative_production** (not year)
        by default.  If ``production_years`` was supplied, an additional
        ``predict_by_year`` callable is stored in ``params``.

    Raises
    ------
    ValueError
        If fewer than 3 data points are provided.
    """
    p = np.asarray(prices, dtype=float)
    q = np.asarray(cumulative_production, dtype=float)

    if len(p) < 3:
        raise ValueError(
            f"fit_wrights_law requires >= 3 data points, got {len(p)}"
        )
    if len(p) != len(q):
        raise ValueError(
            f"prices and cumulative_production must have the same length "
            f"({len(p)} != {len(q)})"
        )

    log_p = _safe_log(p)
    log_q = _safe_log(q)

    # OLS in log-log space:  ln(price) = ln(c0) - alpha * ln(Q)
    slope, intercept, r_value, p_value, std_err = linregress(log_q, log_p)
    alpha = -slope          # positive alpha means cost decreases
    c0 = float(np.exp(intercept))
    r_squared = float(r_value ** 2)

    # Observed learning rate
    learning_rate = 1.0 - 2.0 ** (-alpha)

    # Residuals in log-space
    predicted_log = intercept + slope * log_q
    residuals = log_p - predicted_log
    rss = float(np.sum(residuals ** 2))

    n = len(p)
    k = 2  # alpha + c0

    aic = compute_aic(n, k, rss)
    bic = compute_bic(n, k, rss)

    # Extrapolation warning
    q_min, q_max = float(q.min()), float(q.max())
    extrap_warning = (
        f"Wright's Law fitted on cumulative production [{q_min:.4g}, "
        f"{q_max:.4g}]. Projections beyond {q_max:.4g} are "
        f"extrapolations — learning rates may change."
    )

    # Build primary predict closure (takes cumulative production)
    _alpha, _c0 = alpha, c0

    def _predict(cumulative_prod):
        return predict_wrights_law(cumulative_prod, _alpha, _c0)

    # Optional: production growth model for year-based projection
    prod_growth_params = None
    predict_by_year_fn = None
    if production_years is not None:
        py = np.asarray(production_years, dtype=float)
        if len(py) != len(q):
            warnings.warn(
                "production_years length does not match cumulative_production; "
                "skipping production-growth fit.",
                stacklevel=2,
            )
        elif len(py) >= 2:
            log_q_for_time = _safe_log(q)
            ps, pi, pr, pp, pse = linregress(py, log_q_for_time)
            prod_growth_params = {
                "slope": float(ps),
                "intercept": float(pi),
                "r_squared": float(pr ** 2),
                "std_err": float(pse),
            }

            # Year-based predict:  year -> cumulative_prod -> cost
            _ps, _pi = ps, pi

            def _predict_by_year(year):
                year = np.asarray(year, dtype=float)
                projected_q = np.exp(_pi + _ps * year)
                return predict_wrights_law(projected_q, _alpha, _c0)

            predict_by_year_fn = _predict_by_year

    # Diagnostic: compare observed LR to hint
    lr_warning = None
    if learning_rate_hint is not None:
        delta = abs(learning_rate - learning_rate_hint)
        if delta > 0.10:
            lr_warning = (
                f"Fitted learning rate ({learning_rate:.2%}) differs "
                f"substantially from hint ({learning_rate_hint:.2%})."
            )

    params = {
        "alpha": float(alpha),
        "c0": c0,
        "learning_rate": float(learning_rate),
        "learning_rate_hint": learning_rate_hint,
        "q_min": q_min,
        "q_max": q_max,
    }
    if prod_growth_params is not None:
        params["production_growth"] = prod_growth_params
    if predict_by_year_fn is not None:
        params["predict_by_year"] = predict_by_year_fn
    if lr_warning is not None:
        params["learning_rate_warning"] = lr_warning

    return FitResult(
        method="wrights_law",
        params=params,
        r_squared=r_squared,
        aic=aic,
        bic=bic,
        residuals=residuals,
        n_params=k,
        n_points=n,
        predict=_predict,
        extrapolation_warning=extrap_warning,
    )
