"""
Exponential Atlas v6 — Automated model selection.

Strategy
--------
For each domain we run **every applicable** fit method and select the best
model using **BIC** (Bayesian Information Criterion).

BIC is preferred over AIC for small-sample model selection because it
penalises extra parameters more heavily, guarding against overfitting —
which matters a lot when we have 4–17 data points per domain.

The caller provides a ``domain_config`` dict describing the domain's
characteristics, which determines which fit methods are applicable:

- **log_linear** — always attempted (baseline).
- **piecewise** — attempted if ``n >= 6`` (need 3+ points per segment).
- **wrights_law** — attempted if cumulative-production data is available.
- **logistic** — attempted if ``ceiling`` or ``floor`` is specified AND
  ``n >= 5``.

The result is a ``ModelSelectionResult`` containing the best fit and all
candidate fits for transparency (we show all of them in the methodology
panel on the website).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import FitResult
from .log_linear import fit_log_linear
from .wrights_law import fit_wrights_law
from .piecewise import fit_piecewise
from .logistic import fit_logistic


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ModelSelectionResult:
    """Outcome of running all applicable fits and selecting the best.

    Attributes
    ----------
    best : FitResult
        The fit with the lowest BIC among all successful candidates.
    all_fits : list[FitResult]
        Every fit that succeeded, sorted by BIC (best first).
    selection_criterion : str
        Always ``"BIC"`` in this implementation.
    notes : list[str]
        Informational messages about skipped methods or ties.
    """

    best: FitResult
    all_fits: list[FitResult] = field(default_factory=list)
    selection_criterion: str = "BIC"
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        methods = [f.method for f in self.all_fits]
        return (
            f"ModelSelectionResult(best={self.best.method!r}, "
            f"BIC={self.best.bic:.2f}, "
            f"candidates={methods})"
        )

    def summary_table(self) -> str:
        """Return a human-readable comparison table."""
        lines = [
            f"{'Method':<16} {'R²':>8} {'AIC':>10} {'BIC':>10} {'k':>4} {'n':>4}",
            "-" * 56,
        ]
        for f in self.all_fits:
            marker = " *" if f is self.best else ""
            lines.append(
                f"{f.method:<16} {f.r_squared:>8.4f} {f.aic:>10.2f} "
                f"{f.bic:>10.2f} {f.n_params:>4} {f.n_points:>4}{marker}"
            )
        lines.append("")
        lines.append(f"* = selected by {self.selection_criterion}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------
def select_best_fit(
    years: list | np.ndarray,
    values: list | np.ndarray,
    domain_config: dict,
) -> ModelSelectionResult:
    """Run all applicable fit methods and select the best by BIC.

    Parameters
    ----------
    years : array-like
        Calendar years for each observation.
    values : array-like
        Observed metric values.
    domain_config : dict
        Domain metadata controlling which fits to try.  Expected keys:

        - ``direction`` : ``"g"`` (growing) or ``"d"`` (decreasing).
        - ``physical_floor`` : float or ``None``.
        - ``physical_ceiling`` : float or ``None``.
        - ``best_fit`` : str or ``None`` — hint from domain JSON
          (``"ll"``, ``"wl"``, ``"pw"``, ``"logistic"``).  Does **not**
          override BIC selection, but is noted.
        - ``wrights_law`` : dict or ``None`` — if present, must contain
          ``"prices"`` and ``"cumulative_production"`` arrays plus
          optionally ``"learning_rate_hint"`` and ``"production_years"``.
        - ``piecewise_breakpoint`` : float or ``None`` — hint for
          breakpoint year (passed to ``fit_piecewise``).

    Returns
    -------
    ModelSelectionResult
        Contains ``.best`` (the winner) and ``.all_fits`` (all candidates).
    """
    y = np.asarray(years, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(y)

    candidates: list[FitResult] = []
    notes: list[str] = []

    # --- 1. Log-linear (always) ---
    try:
        ll = fit_log_linear(y, v)
        candidates.append(ll)
    except Exception as exc:
        notes.append(f"log_linear failed: {exc}")

    # --- 2. Piecewise (if n >= 6) ---
    if n >= 6:
        bp_hint = domain_config.get("piecewise_breakpoint")
        try:
            pw = fit_piecewise(y, v, breakpoint_hint=bp_hint)
            candidates.append(pw)
        except Exception as exc:
            notes.append(f"piecewise failed: {exc}")
    else:
        notes.append(
            f"piecewise skipped: need >= 6 data points, have {n}."
        )

    # --- 3. Wright's Law (if production data available) ---
    wl_config = domain_config.get("wrights_law")
    if wl_config is not None:
        try:
            prices = wl_config.get("prices", v)
            cum_prod = wl_config["cumulative_production"]
            lr_hint = wl_config.get("learning_rate_hint")
            prod_years = wl_config.get("production_years")
            wl = fit_wrights_law(
                prices,
                cum_prod,
                learning_rate_hint=lr_hint,
                production_years=prod_years,
            )
            candidates.append(wl)
        except Exception as exc:
            notes.append(f"wrights_law failed: {exc}")
    else:
        notes.append("wrights_law skipped: no cumulative production data.")

    # --- 4. Logistic (if ceiling or floor specified AND n >= 5) ---
    phys_ceiling = domain_config.get("physical_ceiling")
    phys_floor = domain_config.get("physical_floor")
    if (phys_ceiling is not None or phys_floor is not None) and n >= 5:
        try:
            lg = fit_logistic(
                y, v,
                ceiling=phys_ceiling,
                floor=phys_floor,
            )
            candidates.append(lg)
        except Exception as exc:
            notes.append(f"logistic failed: {exc}")
    elif n < 5:
        notes.append(
            f"logistic skipped: need >= 5 data points, have {n}."
        )
    else:
        notes.append(
            "logistic skipped: no physical ceiling or floor specified."
        )

    # --- Selection by BIC ---
    if not candidates:
        raise RuntimeError(
            "All fit methods failed. Cannot select a model. "
            f"Notes: {notes}"
        )

    # Sort by BIC (lower is better)
    candidates.sort(key=lambda f: f.bic)
    best = candidates[0]

    # Check if the hint agrees
    hint = domain_config.get("best_fit")
    if hint is not None:
        hint_map = {"ll": "log_linear", "wl": "wrights_law",
                    "pw": "piecewise", "logistic": "logistic"}
        hint_method = hint_map.get(hint, hint)
        if best.method != hint_method:
            notes.append(
                f"Domain hint suggested '{hint_method}' but BIC selected "
                f"'{best.method}' (BIC {best.bic:.2f} vs "
                + ", ".join(
                    f"{f.method}={f.bic:.2f}" for f in candidates
                    if f.method == hint_method
                )
                + ")."
            )

    return ModelSelectionResult(
        best=best,
        all_fits=candidates,
        selection_criterion="BIC",
        notes=notes,
    )
