"""
Exponential Atlas v6 -- Simulation Configuration
=================================================
Central configuration dataclass for the Monte Carlo simulation engine.

The most important architectural change from v5: base rates are DERIVED
from fitted curves, not hardcoded.  This connects the curve-fitting
pipeline (Phase 1D) to the simulation engine (Phase 2A) so that the
model is self-consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from model.data.domain_registry import (
    SIMULATION_DOMAINS,
    SIM_TO_DATA_MAP,
    AGGREGATION_METHOD,
)
from model.data.loader import load_domain, get_domain_data_points, load_all_domains
from model.fits.model_selection import select_best_fit, ModelSelectionResult
from model.fits.base import FitResult


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Full configuration for one simulation run.

    Attributes
    ----------
    n_years : int
        Number of simulation years (default 14 for 2026-2039).
    start_year : int
        First year of the simulation.
    scenario : str
        One of 'conservative', 'moderate', 'aggressive'.
    n_runs : int
        Number of Monte Carlo runs.
    seed : int
        Random seed for reproducibility.
    gamma_mode : str
        Coupling decay mode: 'adaptive' or 'fixed_decay'.
    base_gamma : float
        Starting coupling strength.
    sim_domains : list[str]
        Ordered list of simulation domain IDs.
    adoption_mode : str
        'bass' for Bass diffusion, 'fixed_lag' for v5-compatible integer lag.
    base_rates : np.ndarray or None
        Per-domain annual improvement rates.  If None, must be derived
        before simulation via derive_base_rates().
    rate_accelerations : np.ndarray or None
        Per-domain annual acceleration of the base rate, measured from
        historical data.  The second derivative of the log-space trajectory.
        Values > 0 mean the improvement rate itself is accelerating.
        Applied as: effective_rate(t) = base_rate * exp(acceleration * t).
    recursive_self_improvement : float
        Power-law exponent controlling AI recursive self-improvement.
        AI's current improvement level feeds back into its own base rate:
        rsi_multiplier = state[ai]^rsi_exponent.  Default 0.15 (optimistic).
        Set to 0.0 to disable.  Website slider range: [0, 0.30].
    noise_std : float
        Standard deviation of log-normal noise per domain per year.
    """
    n_years: int = 14
    start_year: int = 2026
    scenario: str = "moderate"
    n_runs: int = 10000
    seed: int = 42
    gamma_mode: str = "adaptive"
    base_gamma: float = 0.06
    sim_domains: list = field(default_factory=lambda: list(SIMULATION_DOMAINS))
    adoption_mode: str = "bass"
    base_rates: Optional[np.ndarray] = None
    rate_accelerations: Optional[np.ndarray] = None
    recursive_self_improvement: float = 0.15
    noise_std: float = 0.05

    def __post_init__(self):
        if self.sim_domains is None:
            self.sim_domains = list(SIMULATION_DOMAINS)
        if self.scenario not in ("conservative", "moderate", "aggressive"):
            raise ValueError(
                f"scenario must be 'conservative', 'moderate', or 'aggressive', "
                f"got '{self.scenario}'"
            )
        if self.gamma_mode not in ("adaptive", "fixed_decay"):
            raise ValueError(
                f"gamma_mode must be 'adaptive' or 'fixed_decay', "
                f"got '{self.gamma_mode}'"
            )
        if self.adoption_mode not in ("bass", "fixed_lag"):
            raise ValueError(
                f"adoption_mode must be 'bass' or 'fixed_lag', "
                f"got '{self.adoption_mode}'"
            )


# ---------------------------------------------------------------------------
# Scenario multipliers
# ---------------------------------------------------------------------------

SCENARIO_MULTIPLIERS = {
    "conservative": 0.7,
    "moderate": 1.0,
    "aggressive": 1.5,
}


# ---------------------------------------------------------------------------
# Derive base rates from fitted curves
# ---------------------------------------------------------------------------

def _get_slope_from_fit(fit: FitResult) -> float:
    """Extract the annual log-space slope from any fit method.

    For log-linear and piecewise, the slope is directly the log-space
    rate of change per year.  For Wright's Law and logistic, we
    approximate by evaluating the predict function at two years near the
    end of the data range and computing the implied annual change.

    Returns the absolute value of the slope (always positive, representing
    the annual rate of improvement regardless of direction).

    The slope is capped at MAX_ANNUAL_LOG_SLOPE to prevent extreme
    short-window trends (like AI inference costs 2023-2025) from producing
    unrealistic base rates.  Even the fastest technology domains do not
    sustain >10x annual improvement as a base rate -- the extreme recent
    rates in AI are partially captured by interaction effects and
    breakthroughs in the simulation, not purely base rates.
    """
    # Cap: ln(10) ~ 2.30 corresponds to 10x/year.  We cap at ln(3) ~ 1.10
    # (3x/year) for the base rate because interaction effects and
    # breakthroughs will amplify this further in the simulation.
    MAX_ANNUAL_LOG_SLOPE = math.log(3.0)  # ~1.099

    method = fit.method
    params = fit.params

    if method == "log_linear":
        # slope in ln-space per year
        raw = abs(params["slope"])
        return min(raw, MAX_ANNUAL_LOG_SLOPE)

    elif method == "piecewise":
        # Use the post-breakpoint slope (most recent trend)
        # For piecewise, we also look at the pre-breakpoint slope as a
        # sanity anchor.  If the post-break slope is extreme (>5x the
        # pre-break), we blend them to avoid extrapolating a very short
        # acceleration window as the permanent base rate.
        slope_post = abs(params["slope_post"])
        slope_pre = abs(params.get("slope_pre", slope_post))

        if slope_pre > 0 and slope_post > 5.0 * slope_pre:
            # Extreme recent acceleration: blend 70% post + 30% pre
            # This acknowledges acceleration while damping extrapolation risk
            blended = 0.7 * slope_post + 0.3 * slope_pre
            return min(blended, MAX_ANNUAL_LOG_SLOPE)

        return min(slope_post, MAX_ANNUAL_LOG_SLOPE)

    elif method == "wrights_law":
        # Use predict_by_year if available, otherwise approximate
        predict_by_year = params.get("predict_by_year")
        if predict_by_year is not None:
            # Evaluate at two years near the end of the data
            try:
                v1 = predict_by_year(2023.0)
                v2 = predict_by_year(2024.0)
                if v1 > 0 and v2 > 0:
                    raw = abs(math.log(v2 / v1))
                    return min(raw, MAX_ANNUAL_LOG_SLOPE)
            except Exception:
                pass
        # Fallback: use the learning rate to approximate
        alpha = params.get("alpha", 0.30)
        # Conservative: assume production doubles every 2.5 years
        raw = alpha * math.log(2) / 2.5
        return min(raw, MAX_ANNUAL_LOG_SLOPE)

    elif method == "logistic":
        # For logistic, slope varies with time.  Use the steepness k
        # and compute the instantaneous rate at the midpoint (max rate).
        k = params.get("k", 0.1)
        # Max rate of change at midpoint: k * (ceiling - floor) / 4
        # As a fraction of the midpoint value: k / 2
        # Convert to log-space rate: approximately k / 2
        raw = k / 2.0
        return min(raw, MAX_ANNUAL_LOG_SLOPE)

    # Unknown method -- return a conservative default
    return 0.10


def fit_all_data_domains() -> dict[str, ModelSelectionResult]:
    """Run model selection on every data domain.

    Returns
    -------
    dict[str, ModelSelectionResult]
        Mapping of data_domain_id -> best fit result.
    """
    all_domains = load_all_domains(validate=False)
    fits: dict[str, ModelSelectionResult] = {}

    for domain_id, domain_data in all_domains.items():
        years, values = get_domain_data_points(domain_data)
        if len(years) < 2:
            continue

        # Build domain_config for model selection
        domain_config = {
            "direction": "d" if domain_data.get("direction") == "decreasing" else "g",
            "physical_floor": domain_data.get("physical_floor"),
            "physical_ceiling": domain_data.get("physical_ceiling"),
            "best_fit": domain_data.get("best_fit"),
        }

        # Add Wright's Law config if available
        wl = domain_data.get("wrights_law")
        if wl and wl.get("cumulative_production"):
            from model.data.loader import get_wrights_law_data
            wl_data = get_wrights_law_data(domain_data)
            if wl_data is not None:
                wl_years, wl_cum_prod, wl_prices = wl_data
                domain_config["wrights_law"] = {
                    "prices": wl_prices,
                    "cumulative_production": wl_cum_prod,
                    "learning_rate_hint": wl.get("learning_rate"),
                    "production_years": wl_years,
                }

        # Add piecewise config if available
        pw = domain_data.get("piecewise")
        if pw and isinstance(pw, dict):
            domain_config["piecewise_breakpoint"] = pw.get("breakpoint_year")

        try:
            result = select_best_fit(years, values, domain_config)
            fits[domain_id] = result
        except Exception as exc:
            # Skip domains where all fits fail (log warning in production)
            pass

    return fits


def derive_base_rates(
    domain_fits: dict[str, ModelSelectionResult],
    sim_domains: list[str],
    scenario: str = "moderate",
) -> np.ndarray:
    """Derive annual improvement rates from fitted curves.

    For each simulation domain:
    1. Get the fitted slope(s) from Phase 1D fits for each data domain
       that maps to this sim domain.
    2. Convert to annual improvement factor: exp(abs(slope)).
    3. For domains with multiple data domains, aggregate per
       AGGREGATION_METHOD (geometric mean, min, etc.).
    4. Apply scenario multiplier:
       - conservative: 0.7x fitted rate
       - moderate: 1.0x (the data itself)
       - aggressive: 1.5x fitted rate

    Parameters
    ----------
    domain_fits : dict[str, ModelSelectionResult]
        Mapping of data_domain_id -> model selection result.
    sim_domains : list[str]
        Ordered list of simulation domain IDs.
    scenario : str
        'conservative', 'moderate', or 'aggressive'.

    Returns
    -------
    np.ndarray
        Shape (N,) array of annual improvement factors for each sim domain.
        Values > 1.0 mean improvement (e.g., 1.15 = 15% annual improvement).
    """
    multiplier = SCENARIO_MULTIPLIERS[scenario]
    n = len(sim_domains)
    rates = np.ones(n, dtype=np.float64)

    for idx, sim_dom in enumerate(sim_domains):
        data_domains = SIM_TO_DATA_MAP.get(sim_dom, [])
        agg_method = AGGREGATION_METHOD.get(sim_dom, "primary")

        if agg_method == "derived" or not data_domains:
            # Derived domains (e.g., materials) have no data -- use a
            # conservative default rate that will be amplified by interactions
            rates[idx] = 1.02  # 2% base improvement
            continue

        # Collect slopes from all data domains for this sim domain
        slopes = []
        for dd in data_domains:
            if dd in domain_fits:
                fit_result = domain_fits[dd]
                slope = _get_slope_from_fit(fit_result.best)
                slopes.append(slope)

        if not slopes:
            # No fits available -- use conservative default
            rates[idx] = 1.05
            continue

        # Convert slopes to improvement factors: exp(slope)
        improvement_factors = [math.exp(s) for s in slopes]

        # Aggregate according to method
        if agg_method == "geometric_mean":
            # Geometric mean of improvement factors
            log_mean = sum(math.log(f) for f in improvement_factors) / len(improvement_factors)
            base_factor = math.exp(log_mean)
        elif agg_method == "min":
            # Conservative: use the slowest-improving sub-domain
            base_factor = min(improvement_factors)
        elif agg_method == "primary":
            # Single data domain (or just take the first)
            base_factor = improvement_factors[0]
        else:
            base_factor = improvement_factors[0]

        # Apply scenario multiplier to the *excess* improvement
        # (i.e., the improvement above 1.0)
        excess = base_factor - 1.0
        adjusted = 1.0 + excess * multiplier

        # Sanity bounds: improvement factor should be in [1.0, 3.0]
        # (no domain improves more than 3x per year from base rate alone;
        # interaction effects, breakthroughs, and compounding will amplify
        # beyond this in the simulation)
        rates[idx] = max(1.001, min(adjusted, 3.0))

    return rates


# ---------------------------------------------------------------------------
# Derive rate accelerations from historical data (second derivative)
# ---------------------------------------------------------------------------

ACCELERATION_SCENARIO_MULTIPLIERS = {
    "conservative": 0.5,
    "moderate": 1.0,
    "aggressive": 1.5,
}


def derive_rate_accelerations(
    sim_domains: list[str],
    scenario: str = "moderate",
    domains: dict | None = None,
) -> np.ndarray:
    """Derive per-domain rate acceleration from historical data.

    For each domain, fits a quadratic in log-space:
        log(value) = a * year² + b * year + c

    The coefficient 2*a gives the annual acceleration of the log-space
    slope — i.e., how fast the rate of improvement itself is improving.

    This is the "second derivative" feature: domains where historical data
    shows the improvement rate ACCELERATING will have their simulation
    base rates increase over time, rather than staying constant.

    Applied in compute_step() as:
        effective_rate(t) = base_rate * exp(acceleration * t)

    Parameters
    ----------
    sim_domains : list[str]
        Ordered list of simulation domain IDs.
    scenario : str
        'conservative', 'moderate', or 'aggressive'.
    domains : dict, optional
        Pre-loaded domain data. If None, loads from disk.

    Returns
    -------
    np.ndarray
        Shape (N,) array of rate acceleration values per domain.
        Values >= 0.  Zero means no acceleration (constant rate).
    """
    if domains is None:
        domains = load_all_domains(validate=False)

    multiplier = ACCELERATION_SCENARIO_MULTIPLIERS.get(scenario, 1.0)
    n = len(sim_domains)
    accelerations = np.zeros(n, dtype=np.float64)

    for idx, sim_dom in enumerate(sim_domains):
        data_domains = SIM_TO_DATA_MAP.get(sim_dom, [])
        if not data_domains:
            continue

        domain_accels = []

        for dd in data_domains:
            if dd not in domains:
                continue
            domain_data = domains[dd]
            years, values = get_domain_data_points(domain_data)
            if len(years) < 5:
                # Need at least 5 points for a meaningful quadratic fit
                continue

            years_arr = np.array(years, dtype=float)
            values_arr = np.array(values, dtype=float)
            log_vals = np.log(np.maximum(values_arr, 1e-30))

            # Center years for numerical stability
            t_center = years_arr.mean()
            t_norm = years_arr - t_center

            try:
                coeffs = np.polyfit(t_norm, log_vals, 2)
                a_quad = coeffs[0]
            except Exception:
                continue

            direction = domain_data.get("direction", "increasing")

            # For increasing domains: positive a = slope getting steeper (accelerating)
            # For decreasing domains: negative a = decline getting steeper (accelerating)
            if direction == "decreasing":
                rate_accel = -2.0 * a_quad
            else:
                rate_accel = 2.0 * a_quad

            # Only keep positive acceleration (improvement rate is speeding up)
            if rate_accel > 0.002:  # Minimum threshold to avoid noise
                domain_accels.append(rate_accel)

        if domain_accels:
            # Use maximum — the fastest-accelerating signal is the
            # leading indicator.  This is the optimistic approach.
            accelerations[idx] = max(domain_accels)

    # Apply scenario multiplier
    accelerations *= multiplier

    # Cap at maximum reasonable acceleration.
    # 0.10 means log-slope increases by 0.10/year.
    # Over 14 years: rate multiplied by exp(0.10 * 14) = 4.1x.
    # This is aggressive but data-grounded for domains like AI.
    MAX_ACCEL = 0.12
    accelerations = np.minimum(accelerations, MAX_ACCEL)

    return accelerations
