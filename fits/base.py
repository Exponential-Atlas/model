"""
Exponential Atlas v6 — Base types and helpers for curve fitting.

Defines the FitResult dataclass and information criterion computations
(AIC, AICc, BIC) used by all fit methods.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# FitResult — the universal return type for every fit function
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    """Container for a single curve-fit result.

    Every fit function in the ``fits`` package returns one of these so that
    model selection can compare methods on a level playing field.

    Attributes
    ----------
    method : str
        One of ``"log_linear"``, ``"wrights_law"``, ``"piecewise"``,
        ``"logistic"``.
    params : dict
        Method-specific fitted parameters.  See each fit module for the
        keys it stores.
    r_squared : float
        Coefficient of determination (R²) computed on the *fitted* domain
        (log-transformed values for log-linear/piecewise/Wright's,
        natural scale for logistic).
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    residuals : np.ndarray
        Residual vector (observed − predicted) on the scale used for fitting.
    n_params : int
        Number of free parameters in the model.
    n_points : int
        Number of data points used in the fit.
    predict : Callable
        ``predict(x)`` returns the predicted value for *x* where *x* is a
        year (log-linear, piecewise, logistic) or cumulative production
        (Wright's Law).  Accepts scalars **and** numpy arrays.
    extrapolation_warning : str | None
        Human-readable warning if the predict function would be
        extrapolating beyond the observed range of the independent variable.
    """

    method: str
    params: dict
    r_squared: float
    aic: float
    bic: float
    residuals: np.ndarray
    n_params: int
    n_points: int
    predict: Callable
    extrapolation_warning: Optional[str] = None

    # Allow storing extra diagnostics without polluting the main fields
    _extra: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (
            f"FitResult(method={self.method!r}, R²={self.r_squared:.4f}, "
            f"AIC={self.aic:.2f}, BIC={self.bic:.2f}, "
            f"n_params={self.n_params}, n_points={self.n_points})"
        )


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------
def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion with small-sample correction (AICc).

    .. math::

        \\text{AIC}   = n \\ln(\\text{RSS}/n) + 2k
        \\text{AICc}  = \\text{AIC} + \\frac{2k(k+1)}{n - k - 1}

    The corrected version (AICc) is returned whenever *n* is large enough
    relative to *k*.  When ``n - k - 1 <= 0`` the correction is undefined and
    only the base AIC is returned (with a warning).

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of estimated parameters (including the residual variance,
        if that's how you count — but for consistency we follow the common
        convention where *k* is the number of regression coefficients).
    rss : float
        Residual sum of squares.

    Returns
    -------
    float
        AICc (or plain AIC if correction is undefined).
    """
    if n <= 0:
        return np.inf
    if rss <= 0:
        # Perfect fit — log(0) undefined; return −inf to mark as best
        return -np.inf

    aic = n * np.log(rss / n) + 2 * k

    # Small-sample correction (AICc)
    denom = n - k - 1
    if denom > 0:
        aic += 2 * k * (k + 1) / denom
    else:
        warnings.warn(
            f"AICc correction undefined (n={n}, k={k}). Returning plain AIC.",
            stacklevel=2,
        )

    return float(aic)


def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion.

    .. math::

        \\text{BIC} = n \\ln(\\text{RSS}/n) + k \\ln(n)

    BIC penalises extra parameters more heavily than AIC for any
    sample size ``n >= 8``, making it more conservative for model
    selection on small datasets (which we have in most Atlas domains).

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of estimated parameters.
    rss : float
        Residual sum of squares.

    Returns
    -------
    float
        BIC value.
    """
    if n <= 0:
        return np.inf
    if rss <= 0:
        return -np.inf

    bic = n * np.log(rss / n) + k * np.log(n)
    return float(bic)


# ---------------------------------------------------------------------------
# Small helpers used across fit modules
# ---------------------------------------------------------------------------
def _safe_log(values: np.ndarray) -> np.ndarray:
    """Return ``np.log(values)`` after clamping zeros/negatives to a tiny
    positive number so we never hit ``-inf`` or ``nan``."""
    v = np.asarray(values, dtype=float)
    v = np.where(v > 0, v, np.finfo(float).tiny)
    return np.log(v)


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """R² (coefficient of determination).

    Returns 0.0 if the total sum of squares is zero (all observations
    identical), to avoid division-by-zero.
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
