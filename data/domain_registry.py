"""
Exponential Atlas v6 — Domain Registry
=======================================
Makes the mapping between data domains and simulation domains EXPLICIT.

In v5 this mapping was completely implicit:
- 24 data domains were defined in one place
- 12 simulation domains appeared in a separate list
- No code connected them

v6 makes this a first-class, inspectable, testable data structure.
"""

# ---------------------------------------------------------------------------
# Simulation domains — the expanded set (v5 had 12, v6 has 15)
# ---------------------------------------------------------------------------

SIMULATION_DOMAINS: list[str] = [
    "ai",
    "compute",
    "energy",
    "batteries",
    "genomics",
    "drug",
    "robotics",
    "space",
    "manufacturing",
    "materials",       # derived from interactions, no direct data domain
    "bci",
    "quantum",
    "environment",     # NEW in v6 — was not a v5 sim domain
    "vr",              # NEW in v6 — was not a v5 sim domain
    "sensors",         # NEW in v6 — was lumped into compute in v5
]

# ---------------------------------------------------------------------------
# Data domain -> Simulation domain mapping
# ---------------------------------------------------------------------------

DATA_TO_SIM_MAP: dict[str, str] = {
    # AI cluster
    "ai_inference":     "ai",
    "ai_training":      "ai",
    "ai_context":       "ai",
    # Compute cluster
    "compute_flops":    "compute",
    "storage_cost":     "compute",
    "bandwidth":        "compute",
    # Energy cluster
    "solar_module":     "energy",
    "solar_lcoe":       "energy",
    "wind_lcoe":        "energy",
    "storage_lcoe":     "energy",
    # Batteries (single data domain)
    "battery_pack":     "batteries",
    # Genomics cluster
    "genome":           "genomics",
    "dna_synthesis":    "genomics",
    # Drug discovery (single data domain)
    "drug_time":        "drug",
    # Robotics cluster
    "humanoid":         "robotics",
    "industrial_robot": "robotics",
    # Space (single data domain)
    "launch":           "space",
    # Manufacturing (single data domain)
    "3d_metal":         "manufacturing",
    # BCI (single data domain)
    "bci":              "bci",
    # Quantum (single data domain)
    "quantum":          "quantum",
    # Environment cluster (NEW — these existed as data but had no sim domain)
    "desal":            "environment",
    "carbon_capture":   "environment",
    # VR (NEW — existed as data, no sim domain in v5)
    "vr_res":           "vr",
    # Sensors (NEW — was lumped into compute in v5)
    "sensor":           "sensors",
}

# ---------------------------------------------------------------------------
# Reverse mapping: Simulation domain -> list of data domains
# ---------------------------------------------------------------------------

SIM_TO_DATA_MAP: dict[str, list[str]] = {}
for _data_dom, _sim_dom in DATA_TO_SIM_MAP.items():
    SIM_TO_DATA_MAP.setdefault(_sim_dom, []).append(_data_dom)

# Materials has no data domains — it is derived from interaction effects
SIM_TO_DATA_MAP.setdefault("materials", [])

# Sort each list for deterministic ordering
for _k in SIM_TO_DATA_MAP:
    SIM_TO_DATA_MAP[_k] = sorted(SIM_TO_DATA_MAP[_k])

# ---------------------------------------------------------------------------
# Aggregation method for multi-data-domain simulation domains
# ---------------------------------------------------------------------------

AGGREGATION_METHOD: dict[str, str] = {
    # Geometric mean of improvement rates — appropriate when sub-domains
    # measure fundamentally different things (cost, capability, context)
    "ai":             "geometric_mean",
    # Geometric mean — cost per FLOP, cost per GB storage, cost per GB bandwidth
    "compute":        "geometric_mean",
    # Geometric mean — module cost, LCOE solar, LCOE wind, LCOE storage
    "energy":         "geometric_mean",
    # Primary — single data domain
    "batteries":      "primary",
    # Geometric mean — sequencing cost, synthesis cost
    "genomics":       "geometric_mean",
    # Primary — single data domain
    "drug":           "primary",
    # Min (conservative) — humanoid and industrial measure different markets
    "robotics":       "min",
    # Primary — single data domain
    "space":          "primary",
    # Primary — single data domain
    "manufacturing":  "primary",
    # Derived — no data domains, driven by interaction effects
    "materials":      "derived",
    # Primary — single data domain
    "bci":            "primary",
    # Primary — single data domain
    "quantum":        "primary",
    # Geometric mean — desalination + carbon capture
    "environment":    "geometric_mean",
    # Primary — single data domain
    "vr":             "primary",
    # Primary — single data domain
    "sensors":        "primary",
}


# ---------------------------------------------------------------------------
# ALL_DATA_DOMAINS — canonical ordered list of every data domain id
# ---------------------------------------------------------------------------

ALL_DATA_DOMAINS: list[str] = sorted(DATA_TO_SIM_MAP.keys())


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_sim_domain(data_domain_id: str) -> str:
    """Return the simulation domain for a given data domain id."""
    if data_domain_id not in DATA_TO_SIM_MAP:
        raise KeyError(
            f"Unknown data domain '{data_domain_id}'. "
            f"Known: {ALL_DATA_DOMAINS}"
        )
    return DATA_TO_SIM_MAP[data_domain_id]


def get_data_domains(sim_domain: str) -> list[str]:
    """Return the list of data domains feeding a simulation domain."""
    if sim_domain not in SIM_TO_DATA_MAP:
        raise KeyError(
            f"Unknown simulation domain '{sim_domain}'. "
            f"Known: {SIMULATION_DOMAINS}"
        )
    return SIM_TO_DATA_MAP[sim_domain]


def get_aggregation(sim_domain: str) -> str:
    """Return the aggregation method for a simulation domain."""
    if sim_domain not in AGGREGATION_METHOD:
        raise KeyError(
            f"No aggregation method defined for '{sim_domain}'. "
            f"Known: {list(AGGREGATION_METHOD.keys())}"
        )
    return AGGREGATION_METHOD[sim_domain]
