"""
Exponential Atlas v6 -- Simulation Package
==========================================
Modular refactor of v5's monolithic simulate() function into proper
components connected to the Phase 1 data/fitting infrastructure.

Architecture:
    config.py       - SimulationConfig dataclass + derive_base_rates()
    dynamics.py     - Core inner loop (compute_step)
    monte_carlo.py  - Full Monte Carlo runner with convergence testing
    adoption.py     - Bass diffusion adoption delay
    breakthroughs.py- Stochastic breakthrough events
    constraints.py  - Physical/regulatory constraints
    gamma.py        - Coupling strength computation
    analyze.py      - Domain analysis (fit + project + acceleration detect)

Quick usage::

    from model.simulation import (
        SimulationConfig,
        run_monte_carlo,
        derive_base_rates,
        fit_all_data_domains,
        analyze_domains,
    )

    # Step 1: Fit all data domains
    domain_fits = fit_all_data_domains()

    # Step 2: Derive base rates from fits
    config = SimulationConfig(scenario='moderate', n_runs=10000)
    config.base_rates = derive_base_rates(domain_fits, config.sim_domains, config.scenario)

    # Step 3: Run Monte Carlo
    result = run_monte_carlo(config)

    # Step 4: Inspect results
    for domain in config.sim_domains:
        p50_2039 = result.percentiles[domain][2039]['p50']
        print(f"{domain}: {p50_2039:.1f}x improvement by 2039")
"""

# Configuration
from .config import (
    SimulationConfig,
    derive_base_rates,
    derive_rate_accelerations,
    fit_all_data_domains,
    SCENARIO_MULTIPLIERS,
    ACCELERATION_SCENARIO_MULTIPLIERS,
)

# Core simulation
from .dynamics import compute_step
from .monte_carlo import run_monte_carlo, MonteCarloResult
from .gamma import compute_gamma

# Stochastic components
from .breakthroughs import generate_breakthroughs, BREAKTHROUGH_PROBS
from .adoption import (
    bass_diffusion_weight,
    apply_adoption_delay,
    apply_fixed_lag,
    get_bass_params_for_year,
    deployment_trend_summary,
    DOMAIN_FRICTION,
    DOMAIN_P,
    DEPLOYMENT_SPEED_DATA,
)

# Constraints
from .constraints import apply_constraints

# Analysis
from .analyze import analyze_domains, analyze_domain

__all__ = [
    # Config
    "SimulationConfig",
    "derive_base_rates",
    "derive_rate_accelerations",
    "fit_all_data_domains",
    "SCENARIO_MULTIPLIERS",
    "ACCELERATION_SCENARIO_MULTIPLIERS",
    # Simulation
    "compute_step",
    "run_monte_carlo",
    "MonteCarloResult",
    "compute_gamma",
    # Stochastic
    "generate_breakthroughs",
    "BREAKTHROUGH_PROBS",
    "bass_diffusion_weight",
    "apply_adoption_delay",
    "apply_fixed_lag",
    "get_bass_params_for_year",
    "deployment_trend_summary",
    "DOMAIN_FRICTION",
    "DOMAIN_P",
    "DEPLOYMENT_SPEED_DATA",
    # Constraints
    "apply_constraints",
    # Analysis
    "analyze_domains",
    "analyze_domain",
]
