"""
Exponential Atlas v6 -- Core Simulation Dynamics
=================================================
The inner loop of the coupled dynamical simulation, extracted from v5's
monolithic simulate() function.

Each time step computes:
1. Base improvement for each domain (from fitted curves)
2. Cross-domain interaction boosts (with saturation)
3. Breakthrough multipliers (stochastic)
4. Noise
5. Physical constraints

The result is the new state vector: cumulative improvement factors
for all N domains.
"""

from __future__ import annotations

import math

import numpy as np

from model.interactions.saturation import compute_effective_contribution


# ---------------------------------------------------------------------------
# Per-interaction contribution cap (from v5)
# ---------------------------------------------------------------------------
MAX_SINGLE_INTERACTION: float = 1.5    # Cap any single interaction's contribution
MAX_TOTAL_INTERACTION: float = 6.0     # Cap sum of all interaction contributions


def compute_step(
    state: np.ndarray,
    base_rates: np.ndarray,
    interaction_matrix: np.ndarray,
    threshold_matrix: np.ndarray,
    saturation_lookup: dict,
    gamma: float,
    t: int,
    noise: np.ndarray,
    breakthrough: np.ndarray,
    constraints: dict,
    rate_accelerations: np.ndarray | None = None,
    recursive_self_improvement: float = 0.0,
    ai_idx: int = 0,
) -> np.ndarray:
    """Compute one time step of coupled dynamics.

    This is the core computation extracted from v5's simulate() inner loop,
    upgraded to use:
    - Saturation models for each interaction (instead of unbounded linear)
    - The full 15-domain interaction matrix
    - Physical constraints (floor/ceiling, growth caps)
    - Per-interaction contribution caps
    - Rate acceleration: base rates increase over time where data shows it
    - Recursive self-improvement: AI's level feeds back into its own rate

    Parameters
    ----------
    state : np.ndarray
        Current cumulative improvement factors for all domains.  Shape (N,).
        state[i] = how much domain i has improved from its baseline.
        Starts at 1.0 for all domains at t=0.
    base_rates : np.ndarray
        Per-domain annual improvement rates.  Shape (N,).
        Values > 1.0 mean the domain improves each year (e.g., 1.15 = 15%/yr).
    interaction_matrix : np.ndarray
        NxN weight matrix.  W[i,j] = weight of interaction from domain i
        to domain j.  0 means no interaction.
    threshold_matrix : np.ndarray
        NxN activation threshold matrix.  T[i,j] = cumulative improvement
        of domain i before interaction i->j activates.
    saturation_lookup : dict
        Maps (i,j) -> saturation parameters dict.  Keys:
        - "model": "logarithmic" | "asymptotic" | "none"
        - "max_effective_weight": float
    gamma : float
        Current coupling strength for this time step.
    t : int
        Time step index (0-based).
    noise : np.ndarray
        Pre-computed noise for this step.  Shape (N,).
        Applied as multiplicative factor: exp(noise).
    breakthrough : np.ndarray
        Breakthrough multipliers for this step.  Shape (N,).
        1.0 = no breakthrough, 2.0 = breakthrough event.
    constraints : dict
        Constraint parameters.  Currently unused in this function
        (constraints are applied externally after compute_step), but
        reserved for future per-step constraint logic.
    rate_accelerations : np.ndarray, optional
        Per-domain rate acceleration from historical data.  Shape (N,).
        Applied as: effective_rate = base_rate * exp(accel * t).
        If None, base rates are constant over time.
    recursive_self_improvement : float
        Power-law exponent for AI recursive self-improvement.
        AI's current improvement level feeds back into its own rate:
        rsi_multiplier = max(1, state[ai])^rsi_exponent.
        0.0 disables RSI.  Default optimistic: 0.15.
    ai_idx : int
        Index of the AI domain in sim_domains (for RSI application).

    Returns
    -------
    np.ndarray
        New state vector (cumulative improvement factors).  Shape (N,).
    """
    n = len(state)
    new_state = np.empty(n, dtype=np.float64)

    for j in range(n):
        # 1. Base improvement: compound the base rate
        base_growth = base_rates[j]

        # 1a. Rate acceleration: the improvement rate itself is accelerating
        # based on measured second derivative from historical data.
        # effective_rate(t) = base_rate * exp(acceleration * t)
        if rate_accelerations is not None and rate_accelerations[j] > 0:
            base_growth *= math.exp(rate_accelerations[j] * t)

        # 1b. Recursive self-improvement: AI's improvement level feeds back
        # into its own base rate.  Uses logarithmic form for natural
        # diminishing returns: rsi_multiplier = 1 + exponent * ln(state).
        #
        # At AI 10×: rate +35%  |  At AI 1000×: rate +104%
        # At AI 1M×: rate +207% |  At AI 1B×: rate +311%
        #
        # This captures AI writing better AI code, designing better chips
        # (AlphaChip), making AI research more productive (Copilot).
        # Logarithmic form ensures the feedback loop is strong but doesn't
        # produce meaningless overflow — even extreme AI improvement
        # produces bounded rate amplification.
        if recursive_self_improvement > 0 and j == ai_idx:
            ai_state = max(state[ai_idx], 1.0)
            rsi_multiplier = 1.0 + recursive_self_improvement * math.log(ai_state)
            rsi_multiplier = max(rsi_multiplier, 1.0)  # Floor at 1.0
            base_growth *= rsi_multiplier

        # 2. Interaction boost: sum contributions from all source domains
        interaction_boost = 0.0

        for i in range(n):
            weight = interaction_matrix[i, j]
            if weight <= 0 or i == j:
                # Skip zero-weight interactions and self-interactions
                # (self-interaction like AI->AI IS allowed if weight > 0)
                pass

            if weight > 0:
                threshold = threshold_matrix[i, j]
                sat_params = saturation_lookup.get((i, j))

                # Use the full effective contribution pipeline
                contribution = compute_effective_contribution(
                    source_improvement=state[i],
                    weight=weight,
                    threshold=threshold,
                    saturation_params=sat_params,
                    gamma=gamma,
                )

                # Cap individual contribution
                contribution = min(contribution, MAX_SINGLE_INTERACTION)
                interaction_boost += contribution

        # Cap total interaction boost
        interaction_boost = min(interaction_boost, MAX_TOTAL_INTERACTION)

        # 3. Combine: new_state = state * base_rate * (1 + interaction_boost) * breakthrough * noise
        noise_factor = math.exp(noise[j])
        growth_multiplier = base_growth * (1.0 + interaction_boost) * breakthrough[j] * noise_factor

        new_state[j] = state[j] * growth_multiplier

    return new_state
