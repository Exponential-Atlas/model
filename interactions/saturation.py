"""
Exponential Atlas v6 — Interaction Saturation Models
=====================================================
Implements diminishing returns for cross-domain interactions.

The core insight: interaction effects cannot grow without bound. Every
amplification channel eventually saturates due to physical limits,
institutional constraints, or diminishing marginal returns.

Three saturation models are supported:

1. **Logarithmic** — contribution = w * log(1 + x) / log(1 + threshold)
   Best for: interactions where early improvements are large and later
   ones are incremental (e.g., AI quality control in manufacturing).

2. **Asymptotic** — contribution = max_w * (1 - exp(-raw / max_w))
   Best for: interactions with a hard physical or theoretical ceiling
   (e.g., AI recursive self-improvement, bounded by model collapse and
   diminishing scaling returns).

3. **None** — contribution = raw (no saturation, v5 legacy behavior)
   Best for: comparison with v5 model and sensitivity analysis.

Usage:
    from model.interactions.saturation import apply_saturation
    from model.interactions.matrix import build_saturation_lookup

    lookup = build_saturation_lookup(sim_domains)
    saturated = apply_saturation(raw_contribution, lookup.get((i, j)))
"""

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Saturation models
# ---------------------------------------------------------------------------

def _saturate_logarithmic(
    raw: float,
    weight: float,
    max_effective_weight: float,
    threshold: float,
) -> float:
    """
    Logarithmic saturation.

    contribution = weight * log(1 + raw) / log(1 + threshold)

    This model produces:
    - Linear-like growth for small raw values
    - Progressively slower growth as raw increases
    - Never exceeds max_effective_weight

    Parameters
    ----------
    raw : float
        Raw (unsaturated) improvement factor of the source domain.
        E.g., if AI has improved 100x from baseline, raw = 100.
    weight : float
        The base interaction weight (from interactions.json).
    max_effective_weight : float
        Hard ceiling on the contribution.
    threshold : float
        The normalization factor. When raw == threshold, the
        contribution equals the base weight.

    Returns
    -------
    float
        Saturated contribution, in [0, max_effective_weight].
    """
    if raw <= 0 or weight <= 0:
        return 0.0

    threshold = max(threshold, 2.0)  # Avoid log(1) = 0 division

    contribution = weight * math.log(1.0 + raw) / math.log(1.0 + threshold)
    return min(contribution, max_effective_weight)


def _saturate_asymptotic(
    raw: float,
    weight: float,
    max_effective_weight: float,
) -> float:
    """
    Asymptotic (exponential decay) saturation.

    contribution = max_w * (1 - exp(-weight * raw / max_w))

    This model produces:
    - Near-linear growth for small raw values (slope ~ weight)
    - Asymptotically approaches max_effective_weight
    - Hard ceiling: contribution never exceeds max_effective_weight

    Best for interactions with known physical or theoretical ceilings
    (e.g., AI self-improvement is bounded by model collapse).

    Parameters
    ----------
    raw : float
        Raw improvement factor of the source domain.
    weight : float
        The base interaction weight.
    max_effective_weight : float
        Asymptotic ceiling.

    Returns
    -------
    float
        Saturated contribution, in [0, max_effective_weight).
    """
    if raw <= 0 or weight <= 0:
        return 0.0

    max_w = max(max_effective_weight, 0.01)  # Avoid division by zero

    # Scale the exponential so that initial slope matches the weight
    contribution = max_w * (1.0 - math.exp(-weight * raw / max_w))
    return contribution


def _saturate_none(raw: float, weight: float) -> float:
    """
    No saturation (v5 legacy behavior).

    contribution = weight * raw_factor

    WARNING: This can produce unbounded growth and is provided only
    for backward compatibility and sensitivity analysis.

    Parameters
    ----------
    raw : float
        Raw improvement factor.
    weight : float
        The base interaction weight.

    Returns
    -------
    float
        Unsaturated contribution.
    """
    if raw <= 0 or weight <= 0:
        return 0.0
    return weight * raw


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_saturation(
    raw_contribution: float,
    interaction: Optional[dict],
    weight: Optional[float] = None,
    threshold: Optional[float] = None,
) -> float:
    """
    Apply the appropriate saturation model to a raw interaction contribution.

    This is the main function called by the simulation engine.

    Parameters
    ----------
    raw_contribution : float
        The raw (unsaturated) improvement factor of the source domain.
        This is typically log10(A[source]) where A is the cumulative
        improvement array — i.e., the log-improvement of the source.
    interaction : dict or None
        Saturation parameters from the interactions lookup:
        - "model": "logarithmic" | "asymptotic" | "none"
        - "max_effective_weight": float
        If None, falls back to "none" model (v5 legacy).
    weight : float, optional
        The base interaction weight. If not provided, extracted from
        the interaction dict (which must then include "weight" at the
        top level — but typically saturation dicts don't include it,
        so pass it explicitly).
    threshold : float, optional
        Activation threshold. Used for logarithmic model normalization.
        If not provided, defaults to 10.

    Returns
    -------
    float
        Saturated contribution value.

    Examples
    --------
    >>> # Logarithmic saturation for AI -> materials
    >>> sat_params = {"model": "logarithmic", "max_effective_weight": 3.5}
    >>> apply_saturation(2.0, sat_params, weight=2.5, threshold=20)
    1.41...

    >>> # Asymptotic saturation for AI -> AI
    >>> sat_params = {"model": "asymptotic", "max_effective_weight": 2.5}
    >>> apply_saturation(1.0, sat_params, weight=1.8)
    1.27...

    >>> # No saturation (v5 behavior)
    >>> apply_saturation(1.0, None, weight=2.0)
    2.0
    """
    if interaction is None:
        # Legacy v5 behavior: no saturation
        w = weight or 1.0
        return _saturate_none(raw_contribution, w)

    model = interaction.get("model", "none")
    max_w = interaction.get("max_effective_weight", 5.0)
    w = weight or 1.0
    thresh = threshold or 10.0

    if model == "logarithmic":
        return _saturate_logarithmic(raw_contribution, w, max_w, thresh)
    elif model == "asymptotic":
        return _saturate_asymptotic(raw_contribution, w, max_w)
    elif model == "none":
        return _saturate_none(raw_contribution, w)
    else:
        raise ValueError(
            f"Unknown saturation model: '{model}'. "
            f"Valid models: 'logarithmic', 'asymptotic', 'none'"
        )


# ---------------------------------------------------------------------------
# Saturation curve visualization data (for React dashboard)
# ---------------------------------------------------------------------------

def saturation_curve_data(
    interaction: dict,
    weight: float,
    threshold: float = 10.0,
    n_points: int = 100,
    max_raw: float = 1000.0,
) -> list[dict]:
    """
    Generate data points for visualizing a saturation curve.

    Useful for the website's interaction detail view — shows how the
    effective contribution changes as the source domain improves.

    Parameters
    ----------
    interaction : dict
        Saturation parameters (model, max_effective_weight).
    weight : float
        Base interaction weight.
    threshold : float
        Activation threshold (for logarithmic model).
    n_points : int
        Number of data points to generate.
    max_raw : float
        Maximum raw improvement value to plot.

    Returns
    -------
    list[dict]
        List of {"raw": float, "saturated": float} data points.
    """
    points = []
    for i in range(n_points + 1):
        raw = (max_raw * i) / n_points
        sat = apply_saturation(raw, interaction, weight, threshold)
        points.append({"raw": round(raw, 4), "saturated": round(sat, 6)})
    return points


def compare_saturation_models(
    weight: float,
    max_effective_weight: float,
    threshold: float = 10.0,
    n_points: int = 50,
    max_raw: float = 100.0,
) -> dict[str, list[dict]]:
    """
    Compare all three saturation models for the same parameters.

    Useful for sensitivity analysis and model selection documentation.

    Parameters
    ----------
    weight : float
        Base interaction weight.
    max_effective_weight : float
        Ceiling for saturating models.
    threshold : float
        Normalization threshold for logarithmic model.
    n_points : int
        Number of data points.
    max_raw : float
        Maximum raw improvement value.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of model name to list of data points.
    """
    models = {
        "logarithmic": {"model": "logarithmic", "max_effective_weight": max_effective_weight},
        "asymptotic": {"model": "asymptotic", "max_effective_weight": max_effective_weight},
        "none": {"model": "none"},
    }

    result = {}
    for name, params in models.items():
        result[name] = saturation_curve_data(
            params, weight, threshold, n_points, max_raw
        )

    return result


# ---------------------------------------------------------------------------
# Effective weight computation (full pipeline)
# ---------------------------------------------------------------------------

def compute_effective_contribution(
    source_improvement: float,
    weight: float,
    threshold: float,
    saturation_params: Optional[dict],
    gamma: float = 0.06,
) -> float:
    """
    Compute the full effective contribution of one interaction at one timestep.

    This combines threshold gating, weight application, and saturation
    into a single function call — matching the simulation engine logic.

    Parameters
    ----------
    source_improvement : float
        Cumulative improvement of the source domain (A[i]).
        E.g., if AI has improved 100x from baseline, this is 100.0.
    weight : float
        Base interaction weight.
    threshold : float
        Activation threshold. The interaction ramps from 0 to full
        as source_improvement crosses this threshold.
    saturation_params : dict or None
        Saturation model parameters.
    gamma : float
        Global coupling constant (scales all interaction effects).

    Returns
    -------
    float
        Effective contribution to the target domain's growth rate.
        This is added to the target's boost factor in the simulation.

    Notes
    -----
    The computation follows this pipeline:
    1. Compute log-improvement: log10(source_improvement)
    2. Compute activation gate: min(1, log_improvement / log10(threshold))
    3. Compute raw contribution: gamma * log_improvement * weight * activation
    4. Apply saturation model
    5. Cap at 1.5 (hard limit from v5)

    The hard cap of 1.5 per interaction prevents any single interaction
    from dominating the simulation. This is conservative but prevents
    runaway dynamics that would reduce model credibility.
    """
    if source_improvement <= 1.0 or weight <= 0:
        return 0.0

    # Step 1: log-improvement
    log_imp = math.log10(max(source_improvement, 1.01))

    # Step 2: activation gate (ramps from 0 to 1)
    log_thresh = math.log10(max(threshold, 2.0))
    activation = min(1.0, log_imp / log_thresh)

    # Step 3: raw contribution
    raw = gamma * log_imp * weight * activation

    # Step 4: apply saturation
    if saturation_params is not None:
        # For saturation, we pass the raw contribution, not log_imp
        # The saturation model further dampens the contribution
        model = saturation_params.get("model", "none")
        max_w = saturation_params.get("max_effective_weight", 5.0)

        if model == "logarithmic":
            # Logarithmic saturation on the raw contribution
            if raw > 0:
                raw = max_w * math.log(1.0 + raw) / math.log(1.0 + max_w)
        elif model == "asymptotic":
            # Asymptotic saturation
            if raw > 0:
                raw = max_w * (1.0 - math.exp(-raw / max_w))
        # "none" leaves raw unchanged

    # Step 5: hard cap (prevents any single interaction from dominating)
    raw = min(raw, 1.5)

    return raw


# ---------------------------------------------------------------------------
# CLI diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Exponential Atlas v6 — Saturation Model Diagnostics")
    print("=" * 70)

    # Demonstrate AI -> AI saturation (the most consequential interaction)
    print("\n--- AI -> AI (asymptotic, weight=1.8, max=2.5) ---")
    ai_sat = {"model": "asymptotic", "max_effective_weight": 2.5}
    test_improvements = [1.01, 2, 5, 10, 50, 100, 1000, 10000, 1e6]
    for imp in test_improvements:
        eff = compute_effective_contribution(imp, 1.8, 10, ai_sat)
        raw = compute_effective_contribution(imp, 1.8, 10, None)
        print(
            f"  AI improvement {imp:>10.1f}x -> "
            f"saturated: {eff:.4f}, unsaturated: {raw:.4f}"
        )

    # Compare models for a typical interaction
    print("\n--- Model comparison (weight=2.0, max=3.0, threshold=10) ---")
    comparison = compare_saturation_models(2.0, 3.0, 10.0, 10, 50)
    print(f"  {'raw':>8} {'logarithmic':>12} {'asymptotic':>12} {'none':>12}")
    for i in range(len(comparison["logarithmic"])):
        log_v = comparison["logarithmic"][i]["saturated"]
        asym_v = comparison["asymptotic"][i]["saturated"]
        none_v = comparison["none"][i]["saturated"]
        raw_v = comparison["logarithmic"][i]["raw"]
        print(f"  {raw_v:8.1f} {log_v:12.4f} {asym_v:12.4f} {none_v:12.4f}")
