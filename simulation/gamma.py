"""
Exponential Atlas v6 -- Coupling Strength (Gamma) Computation
=============================================================
Controls how strongly domains influence each other over time.

Two modes:
- 'fixed_decay': gamma = base_gamma * 0.95^t  (v5 behavior)
- 'adaptive': gamma responds to median system state, decaying faster
  when improvement is large (diminishing returns at scale).
"""

from __future__ import annotations

import numpy as np


def compute_gamma(
    t: int,
    base_gamma: float,
    state: np.ndarray,
    mode: str = "adaptive",
) -> float:
    """Compute coupling strength for time step t.

    Parameters
    ----------
    t : int
        Current time step (0-indexed year offset from start).
    base_gamma : float
        Starting coupling strength (0.06 in v5).
    state : np.ndarray
        Current improvement factors for all domains (N,).
        Values represent cumulative improvement from baseline.
    mode : str
        'fixed_decay' for v5 behavior, 'adaptive' for state-responsive decay.

    Returns
    -------
    float
        Coupling strength gamma for this time step.
        Always in [0, base_gamma].
    """
    if mode == "fixed_decay":
        # v5 behavior: simple exponential decay
        return base_gamma * (0.95 ** t)

    elif mode == "adaptive":
        # Adaptive: decay rate depends on median system improvement
        median_improvement = float(np.median(state))

        if median_improvement < 10.0:
            # Early phase: slow decay -- interactions are still ramping up
            decay_rate = 0.97
        elif median_improvement < 100.0:
            # Mid phase: moderate decay
            decay_rate = 0.95
        elif median_improvement < 1000.0:
            # Late phase: faster decay -- saturation effects
            decay_rate = 0.92
        else:
            # Extreme phase: strong diminishing returns
            decay_rate = 0.90

        return base_gamma * (decay_rate ** t)

    else:
        raise ValueError(
            f"Unknown gamma mode: '{mode}'. Must be 'fixed_decay' or 'adaptive'."
        )
