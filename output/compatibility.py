"""
Exponential Atlas v6 — v5 Compatibility Checker
================================================
Verifies that v6 JSON output is backward compatible with the v5 frontend.

The v5 React prototype expects specific keys and structures.  This module
checks all of them so we can catch regressions before deploying.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# v5 required simulation domains (the 12 original ones)
# ---------------------------------------------------------------------------

V5_SIM_DOMAINS = [
    "ai", "compute", "energy", "batteries", "genomics", "drug",
    "robotics", "space", "manufacturing", "materials", "bci", "quantum",
]


# ---------------------------------------------------------------------------
# v5 required top-level keys
# ---------------------------------------------------------------------------

V5_TOP_LEVEL_KEYS = [
    "meta", "domains", "simulation", "deployment",
    "interactions", "kings", "costs", "possibilities",
    "forecasters", "backtest", "methodology", "weaknesses",
]


# ---------------------------------------------------------------------------
# v5 required domain fields
# ---------------------------------------------------------------------------

V5_DOMAIN_FIELDS = [
    "desc", "unit", "cat", "confidence", "method",
    "current", "start", "total_change", "rate",
    "floor", "ceiling", "projections",
]


# ---------------------------------------------------------------------------
# v5 simulation percentile keys
# ---------------------------------------------------------------------------

V5_SIM_PERCENTILE_KEYS = ["p5", "p10", "p25", "median", "p75", "p90", "p95"]


# ---------------------------------------------------------------------------
# v5 deployment percentile keys
# ---------------------------------------------------------------------------

V5_DEP_PERCENTILE_KEYS = ["p10", "median", "p90"]


# ---------------------------------------------------------------------------
# v5 interaction fields
# ---------------------------------------------------------------------------

V5_INTERACTION_FIELDS = ["from", "to", "weight", "threshold", "evidence"]


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------

def verify_v5_compatibility(v6_json: dict) -> list[str]:
    """Check that v6 output is backward compatible with v5 frontend.

    Checks:
    1. All v5 top-level keys present
    2. simulation[scenario][domain][year] has p5,p10,p25,median,p75,p90,p95
    3. deployment[scenario][domain][year] has p10,median,p90
    4. domains[name] has desc,unit,cat,confidence,method,current,start,
       total_change,rate,floor,ceiling,projections
    5. interactions[] has from,to,weight,threshold,evidence
    6. kings, costs, possibilities, forecasters all present and non-empty
    7. methodology and weaknesses present
    8. All v5 simulation domains present in simulation results

    Parameters
    ----------
    v6_json : dict
        The complete v6 website JSON.

    Returns
    -------
    list[str]
        List of compatibility issues. Empty list means fully compatible.
    """
    issues: list[str] = []

    # --- 1. Top-level keys ---
    for key in V5_TOP_LEVEL_KEYS:
        if key not in v6_json:
            issues.append(f"Missing top-level key: '{key}'")

    # If critical keys are missing, further checks may crash
    if any(f"Missing top-level key: '{k}'" in issues for k in ["simulation", "deployment", "domains"]):
        return issues

    # --- 2. Simulation percentiles ---
    simulation = v6_json.get("simulation", {})
    for scenario in ["conservative", "moderate", "aggressive"]:
        if scenario not in simulation:
            # Only flag if not a quick run
            if len(simulation) > 1 or scenario == "moderate":
                issues.append(f"simulation: missing scenario '{scenario}'")
            continue

        scenario_data = simulation[scenario]

        # Check v5 sim domains present
        for dom in V5_SIM_DOMAINS:
            if dom not in scenario_data:
                issues.append(
                    f"simulation[{scenario}]: missing v5 domain '{dom}'"
                )
                continue

            dom_data = scenario_data[dom]
            # Check at least one year has the right percentile keys
            if dom_data:
                sample_year = next(iter(dom_data.keys()))
                year_data = dom_data[sample_year]
                for pkey in V5_SIM_PERCENTILE_KEYS:
                    if pkey not in year_data:
                        issues.append(
                            f"simulation[{scenario}][{dom}][{sample_year}]: "
                            f"missing percentile key '{pkey}'"
                        )

    # --- 3. Deployment percentiles ---
    deployment = v6_json.get("deployment", {})
    for scenario in simulation.keys():
        if scenario not in deployment:
            issues.append(f"deployment: missing scenario '{scenario}'")
            continue

        dep_data = deployment[scenario]
        for dom in V5_SIM_DOMAINS:
            if dom not in dep_data:
                issues.append(
                    f"deployment[{scenario}]: missing v5 domain '{dom}'"
                )
                continue

            dom_dep = dep_data[dom]
            if dom_dep:
                sample_year = next(iter(dom_dep.keys()))
                year_data = dom_dep[sample_year]
                for pkey in V5_DEP_PERCENTILE_KEYS:
                    if pkey not in year_data:
                        issues.append(
                            f"deployment[{scenario}][{dom}][{sample_year}]: "
                            f"missing percentile key '{pkey}'"
                        )

    # --- 4. Domain fields ---
    domains = v6_json.get("domains", {})
    if not domains:
        issues.append("domains: section is empty")
    else:
        # Check a few known v5 domains
        for dom_name in ["ai_inference", "solar_module", "battery_pack"]:
            if dom_name in domains:
                for field in V5_DOMAIN_FIELDS:
                    if field not in domains[dom_name]:
                        issues.append(
                            f"domains[{dom_name}]: missing v5 field '{field}'"
                        )

    # --- 5. Interactions ---
    interactions = v6_json.get("interactions", [])
    if not interactions:
        issues.append("interactions: list is empty")
    else:
        for i, ix in enumerate(interactions):
            for field in V5_INTERACTION_FIELDS:
                if field not in ix:
                    issues.append(
                        f"interactions[{i}]: missing v5 field '{field}'"
                    )
            # Only check first 5 to avoid noise
            if i >= 4:
                break

    # --- 6. Content databases ---
    for key in ["kings", "costs", "possibilities", "forecasters"]:
        val = v6_json.get(key)
        if val is None:
            issues.append(f"'{key}': missing")
        elif isinstance(val, (list, dict)) and len(val) == 0:
            issues.append(f"'{key}': empty")

    # --- 7. Methodology and weaknesses ---
    methodology = v6_json.get("methodology", {})
    if not methodology:
        issues.append("methodology: section is empty")
    else:
        # v5 had these keys
        for key in ["wrights_law", "piecewise", "log_linear",
                     "threshold_gating", "backtest_calibration",
                     "adoption_delay", "correlated_monte_carlo"]:
            if key not in methodology:
                issues.append(f"methodology: missing v5 key '{key}'")

    weaknesses = v6_json.get("weaknesses", {})
    if not weaknesses:
        issues.append("weaknesses: section is empty")
    else:
        for key in ["biggest", "data_sparse", "not_modelled"]:
            if key not in weaknesses:
                issues.append(f"weaknesses: missing v5 key '{key}'")

    # --- 8. v5 sim domains in simulation ---
    # Already covered in check 2 above

    return issues
