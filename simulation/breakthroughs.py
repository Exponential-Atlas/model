"""
Exponential Atlas v6 -- Breakthrough Event Generator
=====================================================
Models domain-specific breakthrough events (rare, high-impact advances)
as stochastic multipliers in the Monte Carlo simulation.

Key improvements over v5:
- Per-domain breakthrough probabilities (v5 only had AI-linked breakthroughs)
- AI improvement increases breakthrough probability across all domains
  (AI as a meta-accelerator for scientific discovery)
- Probabilities are capped to prevent unrealistic breakthrough cascades
"""

from __future__ import annotations

import math

import numpy as np

from model.data.domain_registry import SIMULATION_DOMAINS


# ---------------------------------------------------------------------------
# Per-domain breakthrough probabilities
# ---------------------------------------------------------------------------

BREAKTHROUGH_PROBS: dict[str, float] = {
    "ai":             0.03,   # New architecture, major scaling insight
    "compute":        0.02,   # New chip paradigm, major efficiency gain
    "energy":         0.02,   # Perovskite tandem, next-gen solar
    "batteries":      0.02,   # Solid-state, new chemistry
    "genomics":       0.03,   # CRISPR-class advance, new editing tool
    "drug":           0.02,   # AI-designed drug approval, new modality
    "robotics":       0.02,   # Dexterous manipulation breakthrough
    "space":          0.03,   # Starship-class disruption, reusability advance
    "manufacturing":  0.01,   # Process revolution (lower probability)
    "materials":      0.02,   # New material class discovery
    "bci":            0.02,   # New electrode tech, wireless breakthrough
    "quantum":        0.04,   # Higher probability -- field is young, error correction
    "environment":    0.01,   # Carbon capture / desal breakthrough
    "vr":             0.02,   # Display / optics breakthrough
    "sensors":        0.01,   # New sensing modality
}

# Breakthrough multiplier: how much a breakthrough amplifies the domain
BREAKTHROUGH_MULTIPLIER: float = 2.0

# Maximum probability cap (prevents unrealistic cascades)
MAX_BREAKTHROUGH_PROB: float = 0.15

# AI conditional boost: how much AI improvement increases other domains'
# breakthrough probability
AI_BOOST_COEFFICIENT: float = 0.1


def generate_breakthroughs(
    rng: np.random.Generator,
    n_domains: int,
    ai_improvement: float,
    base_probs: dict[str, float] | None = None,
    sim_domains: list[str] | None = None,
) -> np.ndarray:
    """Generate breakthrough multipliers for one time step.

    For each domain, a breakthrough occurs with probability base_prob,
    boosted by AI improvement for non-AI domains:

        effective_prob = min(MAX_PROB, base_prob * (1 + 0.1 * log10(ai_improvement)))

    When a breakthrough occurs, the multiplier for that domain is 2.0.
    When no breakthrough occurs, the multiplier is 1.0.

    Parameters
    ----------
    rng : np.random.Generator
        Modern numpy random generator (NOT legacy np.random).
    n_domains : int
        Number of simulation domains.
    ai_improvement : float
        Current cumulative AI improvement factor.  Used to boost
        breakthrough probabilities for non-AI domains.
    base_probs : dict[str, float], optional
        Per-domain base probabilities.  Defaults to BREAKTHROUGH_PROBS.
    sim_domains : list[str], optional
        Ordered list of domain IDs.  Defaults to SIMULATION_DOMAINS.

    Returns
    -------
    np.ndarray
        Shape (n_domains,) array of multipliers (1.0 or BREAKTHROUGH_MULTIPLIER).
    """
    domains = sim_domains or list(SIMULATION_DOMAINS)
    probs = base_probs or BREAKTHROUGH_PROBS

    multipliers = np.ones(n_domains, dtype=np.float64)

    # Compute AI boost factor (only relevant if AI has improved significantly)
    ai_boost = 0.0
    if ai_improvement > 1.0:
        ai_boost = AI_BOOST_COEFFICIENT * math.log10(max(ai_improvement, 1.01))

    # Generate uniform random draws for all domains at once
    draws = rng.uniform(0.0, 1.0, size=n_domains)

    for i, domain in enumerate(domains[:n_domains]):
        base_p = probs.get(domain, 0.01)

        # AI conditional boost (non-AI domains only)
        if domain != "ai":
            effective_p = base_p * (1.0 + ai_boost)
        else:
            effective_p = base_p

        # Cap probability
        effective_p = min(effective_p, MAX_BREAKTHROUGH_PROB)

        if draws[i] < effective_p:
            multipliers[i] = BREAKTHROUGH_MULTIPLIER

    return multipliers
