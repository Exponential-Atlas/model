"""
Exponential Atlas v6 -- Physical Constraints
=============================================
Applies physical, regulatory, and sanity constraints to the simulation state.

No technology trend continues forever without limit.  This module encodes:
1. Physical floor/ceiling from domain JSON configs
2. Production growth caps (no domain grows > 100x per year)
3. Regulatory friction for domains subject to clinical/safety approval
"""

from __future__ import annotations

import numpy as np

from model.data.domain_registry import SIMULATION_DOMAINS


# ---------------------------------------------------------------------------
# Regulatory friction domains and their multipliers
# ---------------------------------------------------------------------------

REGULATORY_FRICTION: dict[str, float] = {
    "drug":    0.90,   # Clinical trial requirements, FDA/EMA approval
    "bci":     0.90,   # Medical device regulation, safety reviews
    "quantum": 0.90,   # Export controls, dual-use technology restrictions
}

# Maximum per-year growth multiplier (sanity check)
MAX_ANNUAL_GROWTH: float = 100.0

# Minimum improvement factor (never below 1.0 -- cannot un-improve)
MIN_IMPROVEMENT: float = 1.0


def apply_constraints(
    state: np.ndarray,
    sim_domains: list[str],
    domain_configs: dict[str, dict],
    base_values: dict[str, float],
    t: int,
) -> np.ndarray:
    """Apply physical and regulatory constraints to the simulation state.

    Parameters
    ----------
    state : np.ndarray
        Current improvement factors for all domains (N,).
        These are cumulative improvement multipliers from the baseline.
    sim_domains : list[str]
        Ordered list of simulation domain IDs.
    domain_configs : dict[str, dict]
        Mapping of data_domain_id -> loaded domain JSON.  Used to look up
        physical_floor and physical_ceiling values.
    base_values : dict[str, float]
        Starting metric values for each simulation domain, used to
        convert improvement factors to absolute values for constraint checks.
    t : int
        Current time step (for growth rate capping).

    Returns
    -------
    np.ndarray
        Constrained improvement factors (same shape as input).
    """
    constrained = state.copy()
    n = len(sim_domains)

    for i in range(n):
        domain = sim_domains[i]

        # 1. Floor: improvement cannot push value below physical floor
        #    improvement_factor * base_value >= floor
        #    => improvement_factor >= floor / base_value
        base_val = base_values.get(domain)
        if base_val is not None and base_val > 0:
            # Look up physical floor/ceiling from any data domain configs
            floor = _get_physical_bound(domain, domain_configs, "physical_floor")
            ceiling = _get_physical_bound(domain, domain_configs, "physical_ceiling")

            if floor is not None:
                # For decreasing domains (costs), improvement = base/current
                # so max improvement = base / floor
                max_from_floor = base_val / floor
                constrained[i] = min(constrained[i], max_from_floor)

            if ceiling is not None and ceiling > 0:
                # For growing domains, max improvement = ceiling / base
                max_from_ceiling = ceiling / base_val
                constrained[i] = min(constrained[i], max_from_ceiling)

        # 2. Production growth cap: no domain grows > 100x per year
        if t > 0:
            # state represents cumulative improvement, so per-year growth
            # is bounded by ensuring state doesn't jump too much in one step.
            # This is a post-hoc sanity check.
            constrained[i] = min(constrained[i], MAX_ANNUAL_GROWTH ** (t + 1))

        # 3. Regulatory friction: certain domains get a friction multiplier
        if domain in REGULATORY_FRICTION:
            friction = REGULATORY_FRICTION[domain]
            # Apply friction to the excess improvement above 1.0
            excess = constrained[i] - 1.0
            if excess > 0:
                constrained[i] = 1.0 + excess * friction

        # 4. Absolute floor: can never go below 1.0 (no un-improvement)
        constrained[i] = max(constrained[i], MIN_IMPROVEMENT)

    return constrained


def _get_physical_bound(
    sim_domain: str,
    domain_configs: dict[str, dict],
    bound_key: str,
) -> float | None:
    """Look up a physical bound for a simulation domain.

    Since sim domains may aggregate multiple data domains, we take the
    most conservative bound:
    - For 'physical_floor': the maximum floor across data domains
      (hardest constraint)
    - For 'physical_ceiling': the minimum ceiling across data domains
      (hardest constraint)

    Parameters
    ----------
    sim_domain : str
        Simulation domain ID.
    domain_configs : dict[str, dict]
        Mapping of data_domain_id -> domain JSON.
    bound_key : str
        'physical_floor' or 'physical_ceiling'.

    Returns
    -------
    float or None
        The most conservative bound, or None if no bound is defined.
    """
    from model.data.domain_registry import SIM_TO_DATA_MAP

    data_domains = SIM_TO_DATA_MAP.get(sim_domain, [])
    bounds = []

    for dd in data_domains:
        config = domain_configs.get(dd, {})
        bound = config.get(bound_key)
        if bound is not None and isinstance(bound, (int, float)):
            bounds.append(bound)

    if not bounds:
        return None

    if bound_key == "physical_floor":
        return max(bounds)  # Most restrictive floor
    else:
        return min(bounds)  # Most restrictive ceiling
