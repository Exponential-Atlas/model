"""
Exponential Atlas v6 — Interaction Matrix Builder
===================================================
Loads interaction definitions from interactions.json and builds the N×N
weight matrix used by the simulation engine.

Key design decisions:
- Interactions are stored as a flat list in JSON for transparency and auditability.
- The matrix builder converts this to a numpy array for simulation performance.
- Every interaction has full evidence provenance — no magic numbers.
- Saturation modeling is delegated to saturation.py.

Usage:
    from model.interactions.matrix import (
        load_interactions,
        build_interaction_matrix,
        get_interaction_evidence,
        get_domain_interactions,
    )

    interactions = load_interactions()
    sim_domains = ["ai", "compute", "energy", ...]
    W = build_interaction_matrix(sim_domains)
    evidence = get_interaction_evidence("ai", "drug")
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTERACTIONS_FILE = Path(__file__).parent / "interactions.json"

# Default simulation domain ordering — must match the simulation engine
DEFAULT_DOMAIN_ORDER = [
    "ai", "compute", "energy", "batteries", "genomics", "drug",
    "robotics", "space", "manufacturing", "materials", "bci", "quantum",
    "environment", "vr", "sensors",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_interactions(filepath: Optional[Path] = None) -> list[dict]:
    """
    Load all interactions from interactions.json.

    Parameters
    ----------
    filepath : Path, optional
        Override the default interactions.json path.

    Returns
    -------
    list[dict]
        List of interaction dictionaries, each containing:
        - from_domain, to_domain: str
        - weight: float
        - v5_weight: float or None
        - activation_threshold: float
        - evidence: list[dict]
        - counter_evidence: list[dict]
        - saturation: dict
        - weight_justification: str

    Raises
    ------
    FileNotFoundError
        If interactions.json does not exist.
    ValueError
        If the JSON structure is invalid.
    """
    path = filepath or _INTERACTIONS_FILE

    if not path.exists():
        raise FileNotFoundError(f"Interactions file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    interactions = data.get("interactions", [])

    if not isinstance(interactions, list):
        raise ValueError("'interactions' must be a list in interactions.json")

    # Validate required fields
    required_fields = [
        "from_domain", "to_domain", "weight",
        "activation_threshold", "evidence",
    ]
    for i, interaction in enumerate(interactions):
        missing = [f for f in required_fields if f not in interaction]
        if missing:
            raise ValueError(
                f"Interaction [{i}] ({interaction.get('id', 'unknown')}): "
                f"missing required fields: {missing}"
            )
        if not isinstance(interaction["weight"], (int, float)):
            raise ValueError(
                f"Interaction [{i}]: 'weight' must be a number, "
                f"got {type(interaction['weight'])}"
            )
        if interaction["weight"] < 0 or interaction["weight"] > 5.0:
            raise ValueError(
                f"Interaction [{i}]: weight {interaction['weight']} "
                f"outside valid range [0, 5.0]"
            )

    return interactions


def load_interactions_meta(filepath: Optional[Path] = None) -> dict:
    """
    Load the metadata section from interactions.json.

    Returns
    -------
    dict
        The 'meta' section containing version, domain_index,
        weight_scale, and changelog_from_v5.
    """
    path = filepath or _INTERACTIONS_FILE

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("meta", {})


def load_key_decisions(filepath: Optional[Path] = None) -> dict:
    """
    Load the key_decisions section from interactions.json.

    This section documents the most important modeling choices and their
    justifications — essential for transparency and peer review.

    Returns
    -------
    dict
        The 'key_decisions' section.
    """
    path = filepath or _INTERACTIONS_FILE

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("key_decisions", {})


# ---------------------------------------------------------------------------
# Matrix Construction
# ---------------------------------------------------------------------------

def build_interaction_matrix(
    sim_domains: Optional[list[str]] = None,
    filepath: Optional[Path] = None,
) -> np.ndarray:
    """
    Build the N×N weight matrix for simulation.

    The matrix W[i, j] represents the amplification weight from domain i
    (source) to domain j (target). A value of 0 means no interaction.

    Parameters
    ----------
    sim_domains : list[str], optional
        Ordered list of domain IDs to include in the matrix.
        Defaults to DEFAULT_DOMAIN_ORDER.
    filepath : Path, optional
        Override the interactions.json path.

    Returns
    -------
    np.ndarray
        Shape (N, N) float64 matrix where N = len(sim_domains).
        W[i, j] > 0 means domain i amplifies domain j with that weight.

    Notes
    -----
    - Interactions referencing domains not in sim_domains are silently skipped.
    - This allows the simulation to run with a subset of domains.
    - The matrix is NOT symmetric: AI→drug and drug→AI can have different weights.
    """
    domains = sim_domains or DEFAULT_DOMAIN_ORDER
    n = len(domains)
    domain_to_idx = {d: i for i, d in enumerate(domains)}

    W = np.zeros((n, n), dtype=np.float64)

    interactions = load_interactions(filepath)

    for interaction in interactions:
        src = interaction["from_domain"]
        tgt = interaction["to_domain"]

        if src in domain_to_idx and tgt in domain_to_idx:
            i = domain_to_idx[src]
            j = domain_to_idx[tgt]
            W[i, j] = interaction["weight"]

    return W


def build_threshold_matrix(
    sim_domains: Optional[list[str]] = None,
    filepath: Optional[Path] = None,
) -> np.ndarray:
    """
    Build the N×N activation threshold matrix for simulation.

    T[i, j] is the cumulative improvement threshold of domain i before
    the interaction i→j activates. The interaction ramps from 0 to full
    weight as the source domain crosses this threshold.

    Parameters
    ----------
    sim_domains : list[str], optional
        Ordered list of domain IDs. Defaults to DEFAULT_DOMAIN_ORDER.
    filepath : Path, optional
        Override the interactions.json path.

    Returns
    -------
    np.ndarray
        Shape (N, N) float64 matrix. Default threshold is 1.0 (always active)
        for interactions not explicitly defined.
    """
    domains = sim_domains or DEFAULT_DOMAIN_ORDER
    n = len(domains)
    domain_to_idx = {d: i for i, d in enumerate(domains)}

    # Default threshold of 1.0 means the interaction is always fully active
    T = np.ones((n, n), dtype=np.float64)

    interactions = load_interactions(filepath)

    for interaction in interactions:
        src = interaction["from_domain"]
        tgt = interaction["to_domain"]

        if src in domain_to_idx and tgt in domain_to_idx:
            i = domain_to_idx[src]
            j = domain_to_idx[tgt]
            T[i, j] = interaction["activation_threshold"]

    return T


def build_saturation_lookup(
    sim_domains: Optional[list[str]] = None,
    filepath: Optional[Path] = None,
) -> dict[tuple[int, int], dict]:
    """
    Build a lookup table for saturation parameters keyed by (i, j) indices.

    Parameters
    ----------
    sim_domains : list[str], optional
        Ordered list of domain IDs. Defaults to DEFAULT_DOMAIN_ORDER.
    filepath : Path, optional
        Override the interactions.json path.

    Returns
    -------
    dict[tuple[int, int], dict]
        Maps (source_idx, target_idx) to saturation parameters:
        - "model": str — "logarithmic", "asymptotic", or "none"
        - "max_effective_weight": float
    """
    domains = sim_domains or DEFAULT_DOMAIN_ORDER
    domain_to_idx = {d: i for i, d in enumerate(domains)}

    lookup: dict[tuple[int, int], dict] = {}

    interactions = load_interactions(filepath)

    for interaction in interactions:
        src = interaction["from_domain"]
        tgt = interaction["to_domain"]

        if src in domain_to_idx and tgt in domain_to_idx:
            i = domain_to_idx[src]
            j = domain_to_idx[tgt]
            sat = interaction.get("saturation", {"model": "none"})
            lookup[(i, j)] = {
                "model": sat.get("model", "none"),
                "max_effective_weight": sat.get("max_effective_weight", 5.0),
                "notes": sat.get("notes", ""),
            }

    return lookup


# ---------------------------------------------------------------------------
# Evidence Lookup
# ---------------------------------------------------------------------------

def get_interaction_evidence(
    from_domain: str,
    to_domain: str,
    filepath: Optional[Path] = None,
) -> Optional[dict]:
    """
    Get the full evidence record for a specific interaction.

    Parameters
    ----------
    from_domain : str
        Source domain ID (e.g., "ai").
    to_domain : str
        Target domain ID (e.g., "drug").
    filepath : Path, optional
        Override the interactions.json path.

    Returns
    -------
    dict or None
        The full interaction record including evidence, counter_evidence,
        weight_justification, and saturation. Returns None if the
        interaction does not exist.
    """
    interactions = load_interactions(filepath)

    for interaction in interactions:
        if (interaction["from_domain"] == from_domain
                and interaction["to_domain"] == to_domain):
            return interaction

    return None


def get_domain_interactions(
    domain: str,
    filepath: Optional[Path] = None,
) -> dict:
    """
    Get all interactions involving a domain, as source or target.

    Parameters
    ----------
    domain : str
        Domain ID (e.g., "ai").
    filepath : Path, optional
        Override the interactions.json path.

    Returns
    -------
    dict
        {
            "as_source": [interactions where domain is the source],
            "as_target": [interactions where domain is the target],
            "total_outgoing_weight": float,
            "total_incoming_weight": float,
            "strongest_outgoing": dict or None,
            "strongest_incoming": dict or None,
        }
    """
    interactions = load_interactions(filepath)

    as_source = []
    as_target = []

    for interaction in interactions:
        if interaction["from_domain"] == domain:
            as_source.append(interaction)
        if interaction["to_domain"] == domain:
            as_target.append(interaction)

    # Sort by weight descending
    as_source.sort(key=lambda x: x["weight"], reverse=True)
    as_target.sort(key=lambda x: x["weight"], reverse=True)

    total_out = sum(i["weight"] for i in as_source)
    total_in = sum(i["weight"] for i in as_target)

    return {
        "domain": domain,
        "as_source": as_source,
        "as_target": as_target,
        "total_outgoing_weight": total_out,
        "total_incoming_weight": total_in,
        "num_outgoing": len(as_source),
        "num_incoming": len(as_target),
        "strongest_outgoing": as_source[0] if as_source else None,
        "strongest_incoming": as_target[0] if as_target else None,
    }


# ---------------------------------------------------------------------------
# Summary & Diagnostics
# ---------------------------------------------------------------------------

def interaction_summary(
    sim_domains: Optional[list[str]] = None,
    filepath: Optional[Path] = None,
) -> dict:
    """
    Generate a summary of the interaction matrix for diagnostic purposes.

    Returns
    -------
    dict
        {
            "n_domains": int,
            "n_interactions": int,
            "density": float (fraction of possible interactions that are nonzero),
            "total_weight": float,
            "mean_weight": float,
            "max_weight": float,
            "max_interaction": str,
            "top_sources": list of (domain, total_outgoing_weight),
            "top_targets": list of (domain, total_incoming_weight),
        }
    """
    domains = sim_domains or DEFAULT_DOMAIN_ORDER
    interactions = load_interactions(filepath)

    # Filter to simulation domains
    domain_set = set(domains)
    active = [
        i for i in interactions
        if i["from_domain"] in domain_set and i["to_domain"] in domain_set
    ]

    n = len(domains)
    n_possible = n * n  # includes self-interactions
    n_active = len(active)

    weights = [i["weight"] for i in active]
    total = sum(weights)
    max_w = max(weights) if weights else 0

    max_interaction = None
    if active:
        best = max(active, key=lambda x: x["weight"])
        max_interaction = f"{best['from_domain']} -> {best['to_domain']} ({best['weight']})"

    # Top sources and targets
    source_weights: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    for i in active:
        source_weights[i["from_domain"]] = (
            source_weights.get(i["from_domain"], 0) + i["weight"]
        )
        target_weights[i["to_domain"]] = (
            target_weights.get(i["to_domain"], 0) + i["weight"]
        )

    top_sources = sorted(
        source_weights.items(), key=lambda x: x[1], reverse=True
    )[:5]
    top_targets = sorted(
        target_weights.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "n_domains": n,
        "n_interactions": n_active,
        "density": n_active / n_possible if n_possible > 0 else 0,
        "total_weight": round(total, 2),
        "mean_weight": round(total / n_active, 2) if n_active > 0 else 0,
        "max_weight": max_w,
        "max_interaction": max_interaction,
        "top_sources": top_sources,
        "top_targets": top_targets,
    }


def validate_interactions(filepath: Optional[Path] = None) -> list[str]:
    """
    Validate all interactions for consistency and completeness.

    Returns
    -------
    list[str]
        List of warning/error strings. Empty list means all checks pass.
    """
    warnings: list[str] = []

    try:
        interactions = load_interactions(filepath)
    except (FileNotFoundError, ValueError) as e:
        return [str(e)]

    meta = load_interactions_meta(filepath)
    valid_domains = set(meta.get("domain_index", DEFAULT_DOMAIN_ORDER))

    seen_pairs: set[tuple[str, str]] = set()

    for i, interaction in enumerate(interactions):
        iid = interaction.get("id", f"[{i}]")

        # Check domains exist
        src = interaction["from_domain"]
        tgt = interaction["to_domain"]
        if src not in valid_domains:
            warnings.append(f"{iid}: from_domain '{src}' not in domain index")
        if tgt not in valid_domains:
            warnings.append(f"{iid}: to_domain '{tgt}' not in domain index")

        # Check for duplicates
        pair = (src, tgt)
        if pair in seen_pairs:
            warnings.append(f"{iid}: duplicate interaction {src} -> {tgt}")
        seen_pairs.add(pair)

        # Check evidence exists
        evidence = interaction.get("evidence", [])
        if not evidence:
            warnings.append(f"{iid}: no evidence provided")

        # Check each evidence item has required fields
        for j, ev in enumerate(evidence):
            if "description" not in ev:
                warnings.append(f"{iid}: evidence[{j}] missing 'description'")
            if "source" not in ev:
                warnings.append(f"{iid}: evidence[{j}] missing 'source'")
            if "source_url" not in ev:
                warnings.append(f"{iid}: evidence[{j}] missing 'source_url'")

        # Check counter-evidence exists for high-weight interactions
        if interaction["weight"] >= 2.0:
            counter = interaction.get("counter_evidence", [])
            if not counter:
                warnings.append(
                    f"{iid}: weight >= 2.0 but no counter_evidence provided"
                )

        # Check saturation defined
        if "saturation" not in interaction:
            warnings.append(f"{iid}: no saturation model defined")

        # Check weight justification
        if not interaction.get("weight_justification"):
            warnings.append(f"{iid}: no weight_justification provided")

        # Check weight scale
        weight_scale = meta.get("weight_scale", {})
        min_w = weight_scale.get("min", 0.5)
        max_w = weight_scale.get("max", 3.0)
        if interaction["weight"] < min_w or interaction["weight"] > max_w:
            warnings.append(
                f"{iid}: weight {interaction['weight']} outside "
                f"declared scale [{min_w}, {max_w}]"
            )

    return warnings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Exponential Atlas v6 — Interaction Matrix Diagnostics")
    print("=" * 70)

    # Validate
    warnings = validate_interactions()
    if warnings:
        print(f"\nValidation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nAll interactions pass validation.")

    # Summary
    summary = interaction_summary()
    print(f"\nDomains: {summary['n_domains']}")
    print(f"Interactions: {summary['n_interactions']}")
    print(f"Matrix density: {summary['density']:.1%}")
    print(f"Total weight: {summary['total_weight']}")
    print(f"Mean weight: {summary['mean_weight']}")
    print(f"Max interaction: {summary['max_interaction']}")

    print("\nTop sources (most outgoing influence):")
    for domain, weight in summary["top_sources"]:
        print(f"  {domain:<15} -> total weight: {weight:.1f}")

    print("\nTop targets (most incoming influence):")
    for domain, weight in summary["top_targets"]:
        print(f"  {domain:<15} <- total weight: {weight:.1f}")

    # Build and display matrix
    W = build_interaction_matrix()
    domains = DEFAULT_DOMAIN_ORDER

    print(f"\nWeight matrix ({len(domains)}x{len(domains)}):")
    print("          ", end="")
    for d in domains:
        print(f"{d[:5]:>6}", end="")
    print()
    for i, src in enumerate(domains):
        print(f"{src:<10}", end="")
        for j in range(len(domains)):
            v = W[i, j]
            if v > 0:
                print(f"{v:6.1f}", end="")
            else:
                print("     .", end="")
        print()
