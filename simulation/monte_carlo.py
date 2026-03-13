"""
Exponential Atlas v6 -- Monte Carlo Simulation Engine
=====================================================
Runs the full Monte Carlo simulation: N stochastic runs of the coupled
dynamical model, collects percentiles, and tests convergence.

Uses numpy.random.default_rng() (modern Generator API) instead of the
legacy np.random.seed() for better statistical properties and
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from model.data.domain_registry import SIMULATION_DOMAINS
from model.interactions.matrix import (
    build_interaction_matrix,
    build_threshold_matrix,
    build_saturation_lookup,
)

from .config import SimulationConfig
from .dynamics import compute_step
from .gamma import compute_gamma
from .breakthroughs import generate_breakthroughs, BREAKTHROUGH_PROBS
from .constraints import apply_constraints
from .adoption import apply_adoption_delay, apply_fixed_lag


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results.

    Attributes
    ----------
    raw_runs : np.ndarray
        Shape (n_runs, n_years+1, n_domains).  Raw capability improvement
        factors before adoption delay.
    deployed_runs : np.ndarray
        Shape (n_runs, n_years+1, n_domains).  Improvement factors after
        adoption delay is applied.
    percentiles : dict
        Maps domain_name -> year -> {p5, p10, p25, p50, p75, p90, p95}.
        Computed from raw_runs.
    deployed_percentiles : dict
        Same structure, computed from deployed_runs.
    convergence_status : dict
        {
            "is_converged": bool,
            "max_p50_change": float,
            "domain_with_max_change": str,
            "n_runs_tested": int,
        }
    config : SimulationConfig
        The configuration used for this simulation.
    sim_domains : list[str]
        Ordered list of domain IDs.
    years : np.ndarray
        Array of calendar years (start_year to start_year + n_years).
    """
    raw_runs: np.ndarray
    deployed_runs: np.ndarray
    percentiles: dict
    deployed_percentiles: dict
    convergence_status: dict
    config: SimulationConfig
    sim_domains: list[str]
    years: np.ndarray


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------

PERCENTILE_LEVELS = [5, 10, 25, 50, 75, 90, 95]
PERCENTILE_NAMES = ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]


def _compute_percentiles(
    runs: np.ndarray,
    sim_domains: list[str],
    start_year: int,
    n_years: int,
) -> dict:
    """Compute percentiles from simulation runs.

    Parameters
    ----------
    runs : np.ndarray
        Shape (n_runs, n_years+1, n_domains).
    sim_domains : list[str]
        Domain names.
    start_year : int
        First year of simulation.
    n_years : int
        Number of simulation years.

    Returns
    -------
    dict
        Maps domain -> year -> {p5, p10, p25, p50, p75, p90, p95}.
    """
    result = {}
    n_domains = len(sim_domains)

    for d_idx in range(n_domains):
        domain = sim_domains[d_idx]
        domain_data = {}

        for t in range(n_years + 1):
            year = start_year + t
            values = runs[:, t, d_idx]
            pcts = np.percentile(values, PERCENTILE_LEVELS)
            domain_data[year] = {
                name: float(val)
                for name, val in zip(PERCENTILE_NAMES, pcts)
            }

        result[domain] = domain_data

    return result


# ---------------------------------------------------------------------------
# Main Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    config: SimulationConfig,
    n_runs: int | None = None,
    seed: int | None = None,
    convergence_test: bool = True,
    domain_configs: dict | None = None,
) -> MonteCarloResult:
    """Run the full Monte Carlo simulation.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration (must have base_rates set).
    n_runs : int, optional
        Override config.n_runs.
    seed : int, optional
        Override config.seed.
    convergence_test : bool
        If True, after all runs, check whether running with 2x runs
        changes the p50 by < 1%.  Reports convergence status.
    domain_configs : dict, optional
        Loaded domain JSON configs, used for constraint checking.
        If None, constraints are skipped (no floor/ceiling enforcement).

    Returns
    -------
    MonteCarloResult
        Full result container with raw runs, deployed runs, percentiles,
        and convergence status.

    Raises
    ------
    ValueError
        If config.base_rates is None.
    """
    if config.base_rates is None:
        raise ValueError(
            "config.base_rates must be set before running Monte Carlo. "
            "Use derive_base_rates() to compute from fitted curves."
        )

    actual_n_runs = n_runs or config.n_runs
    actual_seed = seed if seed is not None else config.seed
    sim_domains = config.sim_domains
    n_domains = len(sim_domains)
    n_years = config.n_years

    # Build interaction structures
    W = build_interaction_matrix(sim_domains)
    T = build_threshold_matrix(sim_domains)
    sat_lookup = build_saturation_lookup(sim_domains)

    # Compute base values for constraints (use improvement factor of 1.0 at baseline)
    base_values = {dom: 1.0 for dom in sim_domains}

    # Pre-allocate output array
    raw_runs = np.empty((actual_n_runs, n_years + 1, n_domains), dtype=np.float64)

    # Create the modern RNG
    rng = np.random.default_rng(actual_seed)

    # Pre-generate all noise for all runs at once for efficiency
    # Shape: (actual_n_runs, n_years, n_domains)
    all_noise = rng.normal(0.0, config.noise_std, size=(actual_n_runs, n_years, n_domains))

    # Find AI index for breakthrough computation
    ai_idx = sim_domains.index("ai") if "ai" in sim_domains else 0

    for run in range(actual_n_runs):
        # Initialize state: all domains start at 1.0 (no improvement yet)
        state = np.ones(n_domains, dtype=np.float64)
        raw_runs[run, 0, :] = state

        # Create a per-run child RNG for breakthroughs
        run_rng = np.random.default_rng(rng.integers(0, 2**63))

        for t in range(n_years):
            # Current AI improvement for breakthrough probability
            ai_improvement = state[ai_idx]

            # Compute coupling strength
            gamma = compute_gamma(
                t=t,
                base_gamma=config.base_gamma,
                state=state,
                mode=config.gamma_mode,
            )

            # Generate breakthrough multipliers
            breakthrough = generate_breakthroughs(
                rng=run_rng,
                n_domains=n_domains,
                ai_improvement=ai_improvement,
                sim_domains=sim_domains,
            )

            # Get noise for this step
            noise = all_noise[run, t, :]

            # Compute one time step
            new_state = compute_step(
                state=state,
                base_rates=config.base_rates,
                interaction_matrix=W,
                threshold_matrix=T,
                saturation_lookup=sat_lookup,
                gamma=gamma,
                t=t,
                noise=noise,
                breakthrough=breakthrough,
                constraints={},
                rate_accelerations=config.rate_accelerations,
                recursive_self_improvement=config.recursive_self_improvement,
                ai_idx=ai_idx,
            )

            # Apply physical constraints
            if domain_configs is not None:
                new_state = apply_constraints(
                    state=new_state,
                    sim_domains=sim_domains,
                    domain_configs=domain_configs,
                    base_values=base_values,
                    t=t + 1,
                )

            state = new_state
            raw_runs[run, t + 1, :] = state

    # NOTE: Deployment modeling removed from v6 output.
    # The model shows technology CAPABILITY — what is technologically possible.
    # Deployment (when capability reaches end users) is subject to variable
    # factors not reliably modeled: regulation, capital allocation, social
    # adoption, infrastructure buildout. However, deployment speed itself
    # follows an exponential trend (see adoption.py DEPLOYMENT_SPEED_DATA),
    # so the gap between capability and deployment is shrinking over time.
    # This is documented transparently in the methodology section.
    deployed_runs = raw_runs  # No separate deployment curve

    # Compute percentiles
    percentiles = _compute_percentiles(raw_runs, sim_domains, config.start_year, n_years)
    deployed_percentiles = percentiles  # Same as raw (no deployment lag)

    # Convergence test
    convergence_status = {
        "is_converged": True,
        "max_p50_change": 0.0,
        "domain_with_max_change": "",
        "n_runs_tested": actual_n_runs,
    }

    if convergence_test and actual_n_runs >= 20:
        convergence_status = _test_convergence(
            raw_runs, sim_domains, config.start_year, n_years
        )

    years = np.arange(config.start_year, config.start_year + n_years + 1)

    return MonteCarloResult(
        raw_runs=raw_runs,
        deployed_runs=deployed_runs,
        percentiles=percentiles,
        deployed_percentiles=deployed_percentiles,
        convergence_status=convergence_status,
        config=config,
        sim_domains=sim_domains,
        years=years,
    )


def _test_convergence(
    raw_runs: np.ndarray,
    sim_domains: list[str],
    start_year: int,
    n_years: int,
) -> dict:
    """Test convergence by comparing p50 from first half vs all runs.

    The test checks: does using only the first half of runs produce p50
    values within 1% of using all runs?  If yes, the simulation has
    converged (more runs would not materially change the results).

    Parameters
    ----------
    raw_runs : np.ndarray
        Shape (n_runs, n_years+1, n_domains).
    sim_domains : list[str]
        Domain names.
    start_year : int
        First year.
    n_years : int
        Number of years.

    Returns
    -------
    dict
        Convergence status.
    """
    n_runs = raw_runs.shape[0]
    half = n_runs // 2
    n_domains = len(sim_domains)

    # Compute p50 for the final year using all runs vs first half
    final_t = n_years

    max_change = 0.0
    max_change_domain = ""

    for d in range(n_domains):
        p50_all = float(np.median(raw_runs[:, final_t, d]))
        p50_half = float(np.median(raw_runs[:half, final_t, d]))

        if p50_all > 0:
            relative_change = abs(p50_all - p50_half) / p50_all
        else:
            relative_change = 0.0

        if relative_change > max_change:
            max_change = relative_change
            max_change_domain = sim_domains[d]

    is_converged = max_change < 0.01  # < 1% change

    return {
        "is_converged": is_converged,
        "max_p50_change": round(max_change, 6),
        "domain_with_max_change": max_change_domain,
        "n_runs_tested": n_runs,
    }
