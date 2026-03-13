# Exponential Atlas v6 — Interactions Package
# =============================================
# Cross-domain interaction weights, evidence, and saturation modeling.
#
# Public API:
#   load_interactions()          — Load all interactions from JSON
#   build_interaction_matrix()   — Build N×N weight matrix for simulation
#   build_threshold_matrix()     — Build N×N activation threshold matrix
#   build_saturation_lookup()    — Build saturation parameter lookup table
#   get_interaction_evidence()   — Get evidence for a specific interaction
#   get_domain_interactions()    — Get all interactions for a domain
#   apply_saturation()           — Apply diminishing returns model
#   compute_effective_contribution() — Full pipeline: gating + weight + saturation
#   validate_interactions()      — Validate interactions.json integrity

from .matrix import (
    DEFAULT_DOMAIN_ORDER,
    load_interactions,
    load_interactions_meta,
    load_key_decisions,
    build_interaction_matrix,
    build_threshold_matrix,
    build_saturation_lookup,
    get_interaction_evidence,
    get_domain_interactions,
    interaction_summary,
    validate_interactions,
)

from .saturation import (
    apply_saturation,
    compute_effective_contribution,
    saturation_curve_data,
    compare_saturation_models,
)

__all__ = [
    "DEFAULT_DOMAIN_ORDER",
    "load_interactions",
    "load_interactions_meta",
    "load_key_decisions",
    "build_interaction_matrix",
    "build_threshold_matrix",
    "build_saturation_lookup",
    "get_interaction_evidence",
    "get_domain_interactions",
    "interaction_summary",
    "validate_interactions",
    "apply_saturation",
    "compute_effective_contribution",
    "saturation_curve_data",
    "compare_saturation_models",
]
