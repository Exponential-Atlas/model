"""
Exponential Atlas v6 — Fitting Package
=======================================

This package provides four curve-fitting methods and an automated model
selection function for technology-trend data.

Fit methods
-----------
- **log_linear** — exponential trend via OLS in log-space.
- **wrights_law** — cost as a power-law of cumulative production.
- **piecewise** — two-segment log-linear with automated breakpoint detection.
- **logistic** — S-curve for domains approaching physical limits.

Model selection
---------------
``select_best_fit()`` runs all applicable methods and picks the winner by
BIC (Bayesian Information Criterion).

Quick usage::

    from model.fits import fit_log_linear, select_best_fit, FitResult

    result = fit_log_linear(years, values)
    print(result.r_squared, result.aic, result.bic)
    print(result.predict(2030))

    selection = select_best_fit(years, values, domain_config)
    print(selection.best)
    print(selection.summary_table())
"""

# Base types and helpers
from .base import FitResult, compute_aic, compute_bic

# Individual fit functions
from .log_linear import fit_log_linear, predict_log_linear
from .wrights_law import fit_wrights_law, predict_wrights_law
from .piecewise import fit_piecewise
from .logistic import fit_logistic

# Automated model selection
from .model_selection import select_best_fit, ModelSelectionResult

__all__ = [
    # Types
    "FitResult",
    "ModelSelectionResult",
    # Helpers
    "compute_aic",
    "compute_bic",
    # Fit functions
    "fit_log_linear",
    "predict_log_linear",
    "fit_wrights_law",
    "predict_wrights_law",
    "fit_piecewise",
    "fit_logistic",
    # Model selection
    "select_best_fit",
]
