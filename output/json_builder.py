"""
Exponential Atlas v6 — JSON Builder for Website
================================================
Builds the complete JSON output for the website frontend.

The output is a STRICT SUPERSET of v5 format — every field that v5 provided
is preserved so the existing React prototype continues to work unchanged.
New v6 fields are added alongside the v5 fields.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

from model.data.domain_registry import (
    SIMULATION_DOMAINS,
    DATA_TO_SIM_MAP,
    SIM_TO_DATA_MAP,
    ALL_DATA_DOMAINS,
)
from model.validation.benchmarks import EXTERNAL_FORECASTS


# ============================================================================
# v5 Content Databases (carried forward verbatim)
# ============================================================================

KINGS = [
    {
        "name": "Ramesses II", "era": "Egypt, 1250 BC", "year": -1250,
        "couldnt": [
            "light after dark without smoke",
            "communicate beyond messengers",
            "cure any infection",
            "preserve food more than days",
        ],
        "best": "Built Abu Simbel. Ruled 66 years. Could command millions but couldn't see clearly in a mirror.",
    },
    {
        "name": "Qin Shi Huang", "era": "China, 210 BC", "year": -210,
        "couldnt": [
            "travel faster than a horse",
            "prevent food poisoning",
            "hear music without live performers",
            "know tomorrow's weather",
        ],
        "best": "Unified China, built the Great Wall. Died seeking immortality from mercury potions.",
    },
    {
        "name": "Cleopatra", "era": "Egypt, 40 BC", "year": -40,
        "couldnt": [
            "see her own face clearly (polished metal mirrors)",
            "treat a tooth infection",
            "communicate in real-time beyond eyesight",
            "regulate temperature in summer",
        ],
        "best": "Spoke 9 languages, ruled an empire. Never saw a clear photograph of her own face.",
    },
    {
        "name": "Marcus Aurelius", "era": "Rome, 170 AD", "year": 170,
        "couldnt": [
            "drink clean water reliably",
            "cure the Antonine Plague",
            "illuminate a room brightly",
            "travel faster than a sailing ship",
        ],
        "best": "Philosopher-emperor of Rome at its height. Likely drank contaminated water daily.",
    },
    {
        "name": "Mansa Musa", "era": "Mali, 1324", "year": 1324,
        "couldnt": [
            "access any book not physically carried across the Sahara",
            "preserve food for ocean voyages",
            "cure malaria",
            "communicate across his empire in less than weeks",
        ],
        "best": "Richest human who ever lived. Transported books by camel across deserts.",
    },
    {
        "name": "Henry VIII", "era": "England, 1530", "year": 1530,
        "couldnt": [
            "stop leg ulcer pain",
            "prevent 2 of 6 wives dying in childbirth-related causes",
            "refrigerate food",
            "hear any music not performed live",
        ],
        "best": "King of England, head of Church. Spent decades in untreatable agony from leg wounds.",
    },
    {
        "name": "Akbar the Great", "era": "Mughal Empire, 1580", "year": 1580,
        "couldnt": [
            "have ice without relay runners from Himalayas (most melted)",
            "treat smallpox",
            "record any moment visually",
            "move information faster than a horse",
        ],
        "best": "Built the Mughal golden age. A cold drink required a human relay chain across hundreds of miles.",
    },
    {
        "name": "Louis XIV", "era": "France, 1700", "year": 1700,
        "couldnt": [
            "eat chocolate as we know it (bitter gritty drink)",
            "bathe regularly (feared water)",
            "cure his gangrenous leg",
            "illuminate Versailles without 20,000 candles",
        ],
        "best": "The Sun King, most powerful man in Europe. Versailles took 20,000 candles to partially light.",
    },
    {
        "name": "Frederick the Great", "era": "Prussia, 1750", "year": 1750,
        "couldnt": [
            "replay a piece of music",
            "operate on a patient without screaming",
            "travel Berlin to Paris in less than days",
            "photograph anything",
        ],
        "best": "Employed C.P.E. Bach personally. Could never hear the same performance twice.",
    },
    {
        "name": "Queen Victoria", "era": "Britain, 1870", "year": 1870,
        "couldnt": [
            "fly anywhere",
            "cure tuberculosis",
            "phone anyone (until very late reign)",
            "air condition any room",
        ],
        "best": "Ruled the largest empire in history. Lost a daughter to diphtheria — no treatment existed.",
    },
    {
        "name": "A 2025 billionaire", "era": "Earth, 2025", "year": 2025,
        "couldnt": [
            "have a drug designed for their genome",
            "reverse ageing",
            "live on the Moon",
            "experience full-immersion virtual reality",
            "have infinite patient genius AI companion 24/7",
        ],
        "best": "Commands billions. Still ages, still gets cancer, still dies. Constrained to one body, one reality, one mind.",
    },
]

COSTS = [
    {"item": "Human genome sequenced", "base_cost": 100, "base_year": 2025, "domain": "genomics", "factor_key": "genomics"},
    {"item": "Humanoid robot", "base_cost": 5900, "base_year": 2025, "domain": "robotics", "factor_key": "robotics"},
    {"item": "1 kg to orbit", "base_cost": 94, "base_year": 2026, "domain": "space", "factor_key": "space"},
    {"item": "Solar panel (1 watt)", "base_cost": 0.08, "base_year": 2025, "domain": "energy", "factor_key": "energy"},
    {"item": "Battery pack (1 kWh)", "base_cost": 108, "base_year": 2025, "domain": "batteries", "factor_key": "batteries"},
    {"item": "MRI scan equivalent (AI diagnostic)", "base_cost": 2000, "base_year": 2025, "domain": "ai", "factor_key": "ai"},
    {"item": "Personalised drug design", "base_cost": 2600000000, "base_year": 2025, "domain": "drug", "factor_key": "drug"},
    {"item": "1 TB storage", "base_cost": 10, "base_year": 2025, "domain": "compute", "factor_key": "compute"},
    {"item": "Desalinated water (1000 litres)", "base_cost": 0.35, "base_year": 2025, "domain": "environment", "factor_key": "environment"},
    {"item": "Carbon capture (1 tonne CO2)", "base_cost": 150, "base_year": 2025, "domain": "environment", "factor_key": "environment"},
    {"item": "Cancer treatment (immunotherapy)", "base_cost": 150000, "base_year": 2025, "domain": "drug", "factor_key": "drug"},
    {"item": "University degree equivalent", "base_cost": 40000, "base_year": 2025, "domain": "ai", "factor_key": "ai"},
    {"item": "DNA synthesis (1000 bp gene)", "base_cost": 20, "base_year": 2025, "domain": "genomics", "factor_key": "genomics"},
]

POSSIBILITIES = [
    {
        "query": "cure for most cancers", "year_range": [2033, 2038],
        "depends": ["genomics", "drug", "ai"],
        "threshold": {"genomics": 500, "drug": 500, "ai": 1e6},
        "explanation": "Requires continuous biosensor monitoring + AI pattern recognition + personalised immunotherapy design. All three curves converging by mid-2030s.",
    },
    {
        "query": "humanoid robot cheaper than a phone", "year_range": [2030, 2035],
        "depends": ["robotics", "manufacturing"],
        "threshold": {"robotics": 50, "manufacturing": 10},
        "explanation": "Robotics costs declining ~50%/yr. At current rate, $5,900 -> sub-$1000 by ~2030, sub-$100 by ~2033.",
    },
    {
        "query": "visit the Moon", "year_range": [2034, 2038],
        "depends": ["space", "robotics", "energy"],
        "threshold": {"space": 100, "robotics": 100, "energy": 5},
        "explanation": "Requires launch <$10/kg + autonomous habitat construction + cheap life support energy.",
    },
    {
        "query": "live on Mars", "year_range": [2037, 2042],
        "depends": ["space", "robotics", "energy", "materials", "ai"],
        "threshold": {"space": 500, "robotics": 1000, "energy": 10, "materials": 50, "ai": 1e9},
        "explanation": "Most demanding convergence. Needs all curves at advanced levels plus autonomous infrastructure.",
    },
    {
        "query": "never age", "year_range": [2036, 2045],
        "depends": ["genomics", "drug", "ai"],
        "threshold": {"genomics": 1000, "drug": 1000, "ai": 1e10},
        "explanation": "Requires complete molecular understanding of ageing + ability to design interventions for every pathway.",
    },
    {
        "query": "full immersion virtual reality", "year_range": [2033, 2037],
        "depends": ["compute", "ai", "bci"],
        "threshold": {"compute": 100, "ai": 1e8, "bci": 50},
        "explanation": "Real-time photorealistic world generation + AI NPCs + advanced haptics or BCI.",
    },
    {
        "query": "free energy", "year_range": [2032, 2037],
        "depends": ["energy", "batteries"],
        "threshold": {"energy": 8, "batteries": 10},
        "explanation": "Solar <$3/MWh + storage <$10/kWh makes energy cost negligible for most applications.",
    },
    {
        "query": "end of involuntary work", "year_range": [2033, 2039],
        "depends": ["robotics", "ai", "manufacturing"],
        "threshold": {"robotics": 1000, "ai": 1e8, "manufacturing": 20},
        "explanation": "Requires cheap intelligent robots capable of all physical tasks.",
    },
    {
        "query": "personalised medicine", "year_range": [2029, 2033],
        "depends": ["genomics", "drug", "ai"],
        "threshold": {"genomics": 100, "drug": 50, "ai": 1e4},
        "explanation": "Already beginning. Full personalisation requires cheap sequencing + AI interpretation + rapid drug design.",
    },
    {
        "query": "talk to AI with your thoughts", "year_range": [2034, 2039],
        "depends": ["bci", "ai", "compute"],
        "threshold": {"bci": 100, "ai": 1e9, "compute": 200},
        "explanation": "Non-invasive BCI with sufficient resolution + AI signal processing. Early forms may appear sooner.",
    },
]


# ============================================================================
# Forecasters — merge v5 originals with v6 benchmarks data
# ============================================================================

def _build_forecasters_section() -> dict:
    """Merge the v5 FORECASTERS with the expanded v6 benchmark data."""
    # Start with the v5 format entries
    forecasters = {
        "ark_2026": {
            "name": "ARK Invest Big Ideas 2026",
            "note": "Annual research report. Models each domain independently. Does NOT model cross-domain amplification.",
            "predictions": {
                "ai": {"claim": "AI could trigger high single-digit real GDP growth by end of decade", "year": 2030},
                "robotics": {"claim": "Robots moving from narrow tasks to open-ended capability", "year": 2030},
                "energy": {"claim": "Massive expansion of low-cost electricity", "year": 2030},
                "batteries": {"claim": "Falling power and storage costs enabling new growth wave", "year": 2030},
            },
        },
        "kurzweil": {
            "name": "Ray Kurzweil (The Singularity Is Nearer, 2024)",
            "note": "Predicts AGI by 2029, singularity by 2045. Models compute mainly.",
            "predictions": {
                "ai": {"claim": "AGI achieved", "year": 2029},
                "longevity": {"claim": "Longevity escape velocity", "year": 2030},
            },
        },
        "iea_weo": {
            "name": "IEA World Energy Outlook 2025",
            "note": "Most comprehensive energy model. Historically too pessimistic on solar/wind.",
            "predictions": {
                "energy": {"claim": "Solar becomes largest source of electricity globally", "year": 2033},
            },
        },
        "epoch_ai": {
            "name": "Epoch AI (2025)",
            "note": "Most rigorous AI-specific trend analysis. Does NOT model cross-domain effects.",
            "predictions": {
                "ai": {"claim": "Inference cost declining 10-200x per year", "year": 2025},
                "compute": {"claim": "FLOP/s per $ doubling every ~2.5 years", "year": 2025},
            },
        },
    }

    # Merge expanded v6 benchmark data
    for key, fc_data in EXTERNAL_FORECASTS.items():
        if key not in forecasters:
            # Add new forecasters from v6 benchmarks
            preds = {}
            for pred in fc_data["predictions"]:
                domain = pred["domain"]
                if domain not in preds:
                    preds[domain] = {
                        "claim": pred["value"],
                        "year": pred["year"],
                        "source_url": pred["source_url"],
                    }
            forecasters[key] = {
                "name": fc_data["name"],
                "note": fc_data.get("methodology", ""),
                "credibility_note": fc_data.get("credibility_note", ""),
                "predictions": preds,
            }
        else:
            # Augment existing forecasters with source_url info from v6
            for pred in fc_data.get("predictions", []):
                domain = pred["domain"]
                if domain in forecasters[key]["predictions"]:
                    forecasters[key]["predictions"][domain]["source_url"] = pred.get("source_url", "")

    return forecasters


# ============================================================================
# Domain section builder
# ============================================================================

def _build_domains_section(
    domain_analyses: dict,
    raw_domains: dict | None = None,
) -> dict:
    """Build the 'domains' section of the JSON.

    Produces a superset of v5 fields for each domain.
    """
    analyses = domain_analyses.get("domains", {})
    domains_out = {}

    for domain_id, analysis in analyses.items():
        if "error" in analysis:
            # Still include it, but with minimal info
            domains_out[domain_id] = {
                "desc": analysis.get("name", domain_id),
                "unit": "unknown",
                "cat": analysis.get("category", "Unknown"),
                "confidence": "low",
                "method": "none",
                "current": None,
                "start": None,
                "total_change": None,
                "rate": None,
                "early_rate": None,
                "late_rate": None,
                "accelerating": False,
                "floor": None,
                "ceiling": None,
                "projections": {},
                "obs_learning_rate": None,
                "error": analysis["error"],
            }
            continue

        # Get raw domain data for additional fields
        raw = {}
        if raw_domains and domain_id in raw_domains:
            raw = raw_domains[domain_id]

        # Compute v5-compatible fields
        value_range = analysis.get("value_range", [0, 0])
        start_val = value_range[0] if value_range else None
        current_val = value_range[1] if value_range and len(value_range) > 1 else None
        total_change = (current_val / start_val) if (start_val and current_val and start_val != 0) else None

        # Rate from the best fit annual improvement factor
        rate = analysis.get("annual_improvement_factor")

        # Acceleration info
        accel = analysis.get("acceleration", {})
        early_rate = accel.get("early_slope", None)
        late_rate = accel.get("late_slope", None)
        accelerating = accel.get("status") == "accelerating"

        # Wright's Law observed learning rate
        obs_lr = None
        fit_params = analysis.get("fit_params", {})
        if analysis.get("best_fit_method") == "wrights_law":
            alpha = fit_params.get("alpha")
            if alpha is not None:
                obs_lr = 1 - 2 ** (-alpha)

        # Projections
        projections = analysis.get("projections", {})
        # Convert to string keys for v5 compatibility
        proj_str = {str(k): v for k, v in projections.items() if v is not None}

        # Data points with sources
        data_points_out = []
        if raw:
            for pt in raw.get("data_points", []):
                data_points_out.append({
                    "year": pt.get("year"),
                    "value": pt.get("value"),
                    "source": pt.get("source", ""),
                    "source_url": pt.get("source_url", ""),
                })

        # All fits compared
        all_fits_compared = analysis.get("all_fits_summary", [])

        entry = {
            # -- v5 fields (all present) --
            "desc": raw.get("description", analysis.get("name", domain_id)),
            "unit": raw.get("unit", "unknown"),
            "cat": raw.get("category", analysis.get("category", "Unknown")),
            "confidence": raw.get("confidence", analysis.get("confidence", "low")),
            "method": analysis.get("best_fit_method", "unknown"),
            "current": current_val,
            "start": start_val,
            "total_change": total_change,
            "rate": rate,
            "early_rate": early_rate,
            "late_rate": late_rate,
            "accelerating": accelerating,
            "floor": raw.get("physical_floor"),
            "ceiling": raw.get("physical_ceiling"),
            "projections": proj_str,
            "obs_learning_rate": obs_lr,
            # -- v6 new fields --
            "r_squared": analysis.get("r_squared"),
            "data_points": data_points_out,
            "fit_details": {
                "method": analysis.get("best_fit_method"),
                "params": {
                    k: v for k, v in fit_params.items()
                    if not callable(v)
                },
                "aic": analysis.get("aic"),
                "bic": analysis.get("bic"),
            },
            "all_fits_compared": all_fits_compared,
        }

        domains_out[domain_id] = entry

    return domains_out


# ============================================================================
# Simulation section builder
# ============================================================================

def _build_simulation_section(sim_results: dict) -> dict:
    """Build the 'simulation' and 'deployment' sections.

    sim_results: dict[scenario_name -> MonteCarloResult]

    For v5 compatibility:
    - simulation[scenario][domain][year_str] = {p5, p10, p25, median, p75, p90, p95}
    - deployment[scenario][domain][year_str] = {p10, median, p90}
    """
    simulation = {}
    deployment = {}

    for scenario_name, mc_result in sim_results.items():
        sim_scenario = {}
        dep_scenario = {}

        # Raw percentiles for simulation
        for domain, yearly_data in mc_result.percentiles.items():
            dom_yearly = {}
            for year, pcts in yearly_data.items():
                # Ensure v5 key names: p5, p10, p25, median, p75, p90, p95
                entry = {
                    "p5": pcts.get("p5", pcts.get("p05", 0)),
                    "p10": pcts.get("p10", 0),
                    "p25": pcts.get("p25", 0),
                    "median": pcts.get("p50", pcts.get("median", 0)),
                    "p75": pcts.get("p75", 0),
                    "p90": pcts.get("p90", 0),
                    "p95": pcts.get("p95", 0),
                }
                dom_yearly[str(year)] = entry
            sim_scenario[domain] = dom_yearly

        # Deployed percentiles for deployment
        for domain, yearly_data in mc_result.deployed_percentiles.items():
            dep_yearly = {}
            for year, pcts in yearly_data.items():
                entry = {
                    "p10": pcts.get("p10", 0),
                    "median": pcts.get("p50", pcts.get("median", 0)),
                    "p90": pcts.get("p90", 0),
                }
                dep_yearly[str(year)] = entry
            dep_scenario[domain] = dep_yearly

        simulation[scenario_name] = sim_scenario
        deployment[scenario_name] = dep_scenario

    return simulation, deployment


# ============================================================================
# Interactions section builder
# ============================================================================

def _build_interactions_section(interaction_data: list) -> list:
    """Build the 'interactions' list with v5 + v6 fields.

    interaction_data: list of interaction dicts from load_interactions()
    """
    interactions_out = []

    for ix in interaction_data:
        entry = {
            # v5 fields
            "from": ix["from_domain"],
            "to": ix["to_domain"],
            "weight": ix["weight"],
            "threshold": ix["activation_threshold"],
            "evidence": _summarize_evidence(ix.get("evidence", [])),
            # v6 fields
            "v5_weight": ix.get("v5_weight"),
            "weight_justification": ix.get("weight_justification", ""),
            "counter_evidence": [
                ce.get("description", str(ce)) if isinstance(ce, dict) else str(ce)
                for ce in ix.get("counter_evidence", [])
            ],
            "saturation_model": ix.get("saturation", {}).get("model", "none") if isinstance(ix.get("saturation"), dict) else "none",
        }
        interactions_out.append(entry)

    return interactions_out


def _summarize_evidence(evidence_list: list) -> str:
    """Summarize evidence into a single string (v5 compatibility)."""
    if not evidence_list:
        return ""
    parts = []
    for ev in evidence_list:
        if isinstance(ev, dict):
            desc = ev.get("description", "")
            source = ev.get("source", "")
            if desc:
                parts.append(f"{desc} ({source})" if source else desc)
        elif isinstance(ev, str):
            parts.append(ev)
    return "; ".join(parts[:3])  # Limit to first 3 for readability


# ============================================================================
# Backtest section builder
# ============================================================================

def _build_backtest_section(backtest_results) -> dict:
    """Build the 'backtest' section from FullBacktestResult."""
    if backtest_results is None:
        return {
            "cutoff_years": [2005, 2010, 2015, 2020],
            "overall_mape": None,
            "calibration_factor": 1.0,
            "bias_direction": "unknown",
            "per_domain": {},
            "per_method": {},
        }

    summary = backtest_results.summary
    overall_mape = summary.get("overall_mape")
    bias_direction = summary.get("bias_direction", "unknown")
    calibration_factor = backtest_results.calibration_factor

    # Per-domain results
    per_domain = {}
    for did, stats in backtest_results.by_domain.items():
        per_domain[did] = {
            "avg_mape": stats.get("avg_mape"),
            "avg_log_error": stats.get("avg_log_error"),
            "bias": stats.get("bias", "neutral"),
            "n_cutoffs_tested": stats.get("n_cutoffs_tested", 0),
            "methods_used": stats.get("methods_used", []),
        }

    # Per-method results
    per_method = {}
    for method, stats in backtest_results.by_method.items():
        per_method[method] = {
            "avg_mape": stats.get("avg_mape"),
            "avg_log_error": stats.get("avg_log_error"),
            "n_domains": stats.get("n_domains", 0),
            "bias": stats.get("bias", "neutral"),
        }

    return {
        "cutoff_years": sorted(backtest_results.by_year.keys()) if backtest_results.by_year else [2005, 2010, 2015, 2020],
        "overall_mape": overall_mape,
        "calibration_factor": calibration_factor,
        "bias_direction": bias_direction,
        "n_comparisons": summary.get("n_comparisons", 0),
        "per_domain": per_domain,
        "per_method": per_method,
    }


# ============================================================================
# Methodology and Weaknesses
# ============================================================================

def _build_deployment_trend() -> dict:
    """Build the deployment trend data for website display.

    This provides the data for the 'deployment speed is itself exponential'
    chart — from electricity (46 years) to ChatGPT (2 months).
    """
    from model.simulation.adoption import deployment_trend_summary
    return deployment_trend_summary()


def _build_methodology() -> dict:
    """Build the 'methodology' section describing all methods used."""
    return {
        "wrights_law": (
            "Used for solar, batteries, sensors — domains where cost is driven by "
            "cumulative production. More accurate than time-based regression because "
            "it captures the causal mechanism."
        ),
        "piecewise": (
            "Used for genomics, drug discovery, space launch — domains with technology "
            "discontinuities. Fitting a single line across a regime change is misleading."
        ),
        "log_linear": (
            "Used for steady-state domains without production data or discontinuities."
        ),
        "logistic": (
            "NEW — S-curve for domains approaching physical limits. Used when a domain "
            "has a defined ceiling or floor and data shows deceleration. Captures the "
            "real-world constraint that exponential trends eventually saturate."
        ),
        "model_selection": (
            "NEW — BIC-based automatic selection of best fit per domain. All applicable "
            "methods are tried (log-linear, piecewise, Wright's Law, logistic) and the "
            "one with the lowest BIC is selected. This removes human bias from method "
            "choice and is fully reproducible."
        ),
        "threshold_gating": (
            "Interaction weights ramp from 0 to full as the source domain crosses "
            "capability thresholds. AI's amplification of drug discovery was zero in "
            "2015 and transformative in 2025 — the model captures this."
        ),
        "backtest_calibration": (
            "Model's historical predictions are compared to actuals at 4 cutoff years "
            "(2005, 2010, 2015, 2020). Systematic bias is measured and partially "
            "corrected in forward projections."
        ),
        "deployment_note": (
            "This model shows TECHNOLOGY CAPABILITY — what is technologically possible. "
            "Deployment (when capability reaches end users) is NOT modeled as a separate curve, "
            "because deployment speed is subject to variable factors (regulation, capital, social "
            "adoption, infrastructure) that cannot be reliably projected. However, we note that "
            "deployment speed itself follows an exponential trend: the time from 'technology "
            "available' to '25% adoption' has fallen from 46 years (electricity, 1882) to 2 months "
            "(ChatGPT, 2022), halving approximately every 30 years. This trend suggests the gap "
            "between capability and deployment is shrinking, not growing. We chose not to model "
            "this explicitly because the uncertainty is too high and a poorly-calibrated deployment "
            "model would mislead more than inform."
        ),
        "correlated_monte_carlo": (
            "Global economic conditions affect all domains simultaneously. A recession "
            "slows everything; a boom accelerates everything. Implemented via shared "
            "noise component across domains per timestep."
        ),
        "saturation": (
            "NEW — Diminishing returns on cross-domain interactions. As a source domain "
            "improves enormously, its marginal amplification of targets decreases. "
            "Implemented as logarithmic or asymptotic saturation per interaction."
        ),
        "sobol_analysis": (
            "NEW — Global sensitivity analysis identifying key drivers. Saltelli (2010) "
            "method with Sobol quasi-random sequences, computing first-order and total-order "
            "indices for 30 parameters. Reveals which parameters matter most and where "
            "better data would reduce uncertainty."
        ),
        "adaptive_gamma": (
            "NEW — Coupling strength responds to system state. As total system improvement "
            "grows, the coupling parameter (gamma) can increase (positive feedback) or "
            "decrease (decay), depending on configuration. Default mode is adaptive."
        ),
        "rate_acceleration": (
            "NEW — Second derivative from data. For each domain, we fit a quadratic in "
            "log-space to measure whether the improvement rate itself is accelerating. "
            "Domains where historical data shows the rate speeding up (e.g., AI inference "
            "cost declining faster each year) have their base rates increase over time in "
            "the simulation: effective_rate(t) = base_rate × exp(acceleration × t). "
            "Domains without measured acceleration keep constant base rates. This is "
            "entirely data-driven — no arbitrary assumptions about which domains accelerate."
        ),
        "recursive_self_improvement": (
            "NEW — AI's improvement level feeds back into its own rate of progress. "
            "Uses logarithmic form: rate_multiplier = 1 + RSI × ln(AI_improvement). "
            "Default RSI exponent: 0.15 (optimistic). At 10× AI improvement, rate "
            "increases by 35%; at 1000× improvement, rate increases by 104%. "
            "Logarithmic form provides natural diminishing returns — the feedback loop "
            "is strong but bounded. This captures AI writing better AI code, designing "
            "better chips (AlphaChip), and making AI research more productive. "
            "The RSI parameter is adjustable (slider: 0 to 0.30) and its impact "
            "propagates to ALL other domains through the interaction matrix. "
            "This is the core mechanism behind the projected singularity trajectory."
        ),
    }


def _build_weaknesses(domain_analyses: dict, backtest_results) -> dict:
    """Build the 'weaknesses' section — honest disclosure."""
    # Find data-sparse domains
    analyses = domain_analyses.get("domains", {})
    data_sparse = []
    for did, a in analyses.items():
        n = a.get("n_points", 0)
        if n <= 5 and "error" not in a:
            data_sparse.append(did)

    # Backtest bias info
    bias_str = "Backtest not yet run"
    if backtest_results is not None:
        summary = backtest_results.summary
        mape = summary.get("overall_mape")
        direction = summary.get("bias_direction", "unknown")
        if mape is not None:
            bias_str = (
                f"Overall MAPE: {mape:.1f}%. Bias direction: {direction}. "
                f"Forward projections should be treated as estimates with "
                f"significant uncertainty."
            )

    return {
        "biggest": (
            "The AI recursive self-improvement parameter (RSI exponent, default 0.15) and the "
            "AI→AI interaction weight (1.8 in v6, reduced from 2.5 in v5) together form the "
            "most consequential driver of the model's long-term projections. RSI creates a "
            "compounding feedback loop: better AI → faster AI improvement → even better AI. "
            "Evidence supports this in narrow domains (NAS, AlphaChip, Copilot coding gains) "
            "but the rate at which this generalizes is the model's biggest uncertainty. "
            "Sensitivity analysis shows these parameters dominate all 2039 outcomes. "
            "Adjust the RSI slider to see how different assumptions change the trajectory."
        ),
        "data_sparse": data_sparse,
        "not_modelled": [
            "Regulatory constraints",
            "Geopolitical disruption",
            "Resource bottlenecks (rare earth minerals, skilled labor)",
            "Social resistance to technology",
            "Black swan events",
            "Supply chain disruptions",
        ],
        "backtest_bias": bias_str,
    }


# ============================================================================
# Model card builder
# ============================================================================

def _build_model_card(
    model_card: dict | None,
    domain_analyses: dict,
    sim_results: dict,
    backtest_results,
    interactions: list,
) -> dict:
    """Build the 'model_card' section for transparency."""
    if model_card is not None:
        return model_card

    # Auto-generate a model card from the available data
    summary = domain_analyses.get("summary", {})
    n_domains = summary.get("total_domains", 0)
    mean_r2 = summary.get("mean_r_squared", 0)

    n_interactions = len(interactions)

    scenarios_run = list(sim_results.keys())
    n_runs = 0
    if sim_results:
        first_result = next(iter(sim_results.values()))
        n_runs = first_result.config.n_runs

    return {
        "title": "Exponential Atlas v6",
        "version": "6.0",
        "purpose": (
            "The first open computational model of cross-domain technology acceleration. "
            "Treats technology domains as a connected system with recursive feedback loops."
        ),
        "domains_fitted": n_domains,
        "mean_r_squared": mean_r2,
        "interactions": n_interactions,
        "scenarios": scenarios_run,
        "mc_runs_per_scenario": n_runs,
        "limitations": [
            "Exponential extrapolation is inherently uncertain beyond 3-5 years",
            "Cross-domain interaction weights have limited empirical grounding",
            "Does not model regulatory, geopolitical, or social constraints",
            "Backtest only covers domains with sufficient historical data",
            "Breakthrough probabilities are speculative",
        ],
        "intended_use": (
            "Educational and research tool for exploring technology convergence scenarios. "
            "NOT investment advice. NOT policy guidance without additional context."
        ),
    }


# ============================================================================
# Sensitivity section builder
# ============================================================================

def _build_sensitivity_section(sensitivity_data) -> dict:
    """Build the 'sensitivity' section from Sobol analysis results."""
    if sensitivity_data is None:
        return {
            "top_drivers": [],
            "tornado_data": {},
            "sobol_indices": {},
            "note": "Sobol analysis not run. Use --sensitivity flag for full analysis.",
        }

    # Extract top drivers across all outputs
    top_drivers = []
    if hasattr(sensitivity_data, 'total_order'):
        for output_key, indices in sensitivity_data.total_order.items():
            sorted_params = sorted(indices.items(), key=lambda kv: kv[1], reverse=True)
            for param_name, idx_val in sorted_params[:5]:
                top_drivers.append({
                    "parameter": param_name,
                    "output": output_key,
                    "total_order_index": round(idx_val, 4),
                })

    # Sobol indices
    sobol_indices = {}
    if hasattr(sensitivity_data, 'first_order') and hasattr(sensitivity_data, 'total_order'):
        sobol_indices = {
            "first_order": sensitivity_data.first_order,
            "total_order": sensitivity_data.total_order,
        }

    # Tornado data
    tornado_data = {}
    if hasattr(sensitivity_data, 'tornado_data'):
        tornado_data = sensitivity_data.tornado_data

    return {
        "top_drivers": top_drivers,
        "tornado_data": tornado_data,
        "sobol_indices": sobol_indices,
    }


# ============================================================================
# RSI variants section builder
# ============================================================================

def _build_rsi_variants_section(rsi_variants: dict | None) -> dict:
    """Build the 'rsi_variants' section for the RSI toggle on the website.

    rsi_variants: dict mapping RSI float value -> dict[scenario -> MonteCarloResult]
        Each RSI value has a full set of scenario results, formatted identically
        to the main 'simulation' section.

    Returns
    -------
    dict
        Maps RSI value string (e.g. "0.0", "0.15", "0.3") to simulation data
        in the same format as the main simulation section:
        { scenario: { domain: { year_str: {p5, p10, p25, median, p75, p90, p95} } } }
    """
    if not rsi_variants:
        return {}

    variants_out = {}
    for rsi_val, scenario_results in rsi_variants.items():
        rsi_key = str(rsi_val)
        # Reuse _build_simulation_section to format each variant identically
        sim_section, _ = _build_simulation_section(scenario_results)
        variants_out[rsi_key] = sim_section

    return variants_out


# ============================================================================
# MAIN BUILDER
# ============================================================================

def build_website_json(
    domain_analyses: dict,
    simulation_results: dict,
    backtest_results,
    interaction_data: list,
    model_card: dict | None,
    benchmark_comparisons: dict | None,
    sensitivity_data=None,
    raw_domains: dict | None = None,
    rsi_variants: dict | None = None,
) -> dict:
    """Build the complete website JSON.

    This is the single function that produces the final JSON output.
    The output is a strict superset of v5 format.

    Parameters
    ----------
    domain_analyses : dict
        Output from analyze_domains().
    simulation_results : dict
        Mapping scenario_name -> MonteCarloResult.
    backtest_results
        Output from run_full_backtest(). Can be a FullBacktestResult or None.
    interaction_data : list
        Output from load_interactions().
    model_card : dict or None
        Pre-built model card, or None to auto-generate.
    benchmark_comparisons : dict or None
        Output from compare_to_benchmarks(), or None.
    sensitivity_data
        Output from run_sobol_analysis(), or None.
    raw_domains : dict or None
        The raw loaded domain dicts (for source URLs, descriptions, etc.).
    rsi_variants : dict or None
        Mapping RSI float value -> dict[scenario -> MonteCarloResult].
        Used to build the rsi_variants section for the website RSI toggle.

    Returns
    -------
    dict
        Complete website JSON ready for json.dump().
    """
    # Count total data points across all raw domains
    total_data_points = 0
    if raw_domains:
        for dom_data in raw_domains.values():
            total_data_points += len(dom_data.get("data_points", []))
    else:
        summary = domain_analyses.get("summary", {})
        total_data_points = sum(
            a.get("n_points", 0)
            for a in domain_analyses.get("domains", {}).values()
        )

    n_domains = domain_analyses.get("summary", {}).get("total_domains", 0)

    # Determine MC run count and config from results
    n_runs = 0
    rsi_exponent = 0.15
    n_accel_domains = 0
    if simulation_results:
        first_result = next(iter(simulation_results.values()))
        n_runs = first_result.config.n_runs
        rsi_exponent = first_result.config.recursive_self_improvement
        if first_result.config.rate_accelerations is not None:
            import numpy as _np
            n_accel_domains = int(_np.sum(first_result.config.rate_accelerations > 0))

    scenarios = sorted(simulation_results.keys()) if simulation_results else []

    # Build sections
    domains_section = _build_domains_section(domain_analyses, raw_domains)
    simulation_section, deployment_section = _build_simulation_section(simulation_results)
    interactions_section = _build_interactions_section(interaction_data)
    backtest_section = _build_backtest_section(backtest_results)
    methodology_section = _build_methodology()
    weaknesses_section = _build_weaknesses(domain_analyses, backtest_results)
    sensitivity_section = _build_sensitivity_section(sensitivity_data)
    rsi_variants_section = _build_rsi_variants_section(rsi_variants)
    model_card_section = _build_model_card(
        model_card, domain_analyses, simulation_results, backtest_results, interaction_data,
    )
    forecasters_section = _build_forecasters_section()

    website = {
        "meta": {
            "title": "The Exponential Atlas",
            "subtitle": "The first open model of cross-domain technology acceleration",
            "version": "v6",
            "domains": n_domains,
            "simulation_domains": len(SIMULATION_DOMAINS),
            "data_points": total_data_points,
            "monte_carlo_runs": n_runs,
            "scenarios": scenarios,
            "generated": date.today().isoformat(),
            "unique_contribution": (
                "This is the first publicly available model that treats technology "
                "domains as a connected system with recursive feedback loops, rather "
                "than isolated trends. The key insight: AI improvement amplifies every "
                "other domain, which feeds back into faster AI improvement. No other "
                "open model captures this cross-domain recursive amplification."
            ),
            "what_this_shows": (
                "Technology CAPABILITY — what is technologically possible at each point in time. "
                "Deployment to end users is not modeled separately because deployment speed is "
                "itself on an exponential trajectory (halving every ~30 years) and subject to "
                "variable factors that cannot be reliably projected. See methodology.deployment_note."
            ),
            "recursive_self_improvement": {
                "exponent": rsi_exponent,
                "formula": "rate_multiplier = 1 + RSI × ln(AI_improvement)",
                "description": (
                    "AI's improvement level feeds back into its own rate. "
                    "Logarithmic: At 10x AI: rate +35%. At 1000x AI: rate +104%. "
                    "Slider range: 0 (disabled) to 0.30 (very aggressive)."
                ),
                "slider_range": [0.0, 0.30],
                "default": 0.15,
            },
            "rate_acceleration": {
                "domains_with_acceleration": n_accel_domains,
                "description": (
                    "Second derivative from data. Domains where the historical "
                    "improvement rate is itself accelerating have time-varying "
                    "base rates in the simulation."
                ),
            },
            "v6_improvements": [
                f"{n_domains} domains (up from 24)",
                "100% source citations",
                "BIC-selected best fit per domain",
                "Sobol sensitivity analysis",
                "Evidence-grounded interaction weights with counter-evidence",
                "AI->AI weight reduced from 2.5 to 1.8 based on evidence",
                "Saturation modeling on all interactions",
                "Base rates derived from fitted data (not hardcoded)",
                "Rate acceleration: improvement rates increase over time where data supports it",
                "Recursive self-improvement: AI capability feeds back into AI growth rate",
            ],
        },
        "domains": domains_section,
        "simulation": simulation_section,
        "deployment": deployment_section,
        "interactions": interactions_section,
        "kings": KINGS,
        "costs": COSTS,
        "possibilities": POSSIBILITIES,
        "forecasters": forecasters_section,
        "backtest": backtest_section,
        "deployment_trend": _build_deployment_trend(),
        "methodology": methodology_section,
        "weaknesses": weaknesses_section,
        "sensitivity": sensitivity_section,
        "rsi_variants": rsi_variants_section,
        "model_card": model_card_section,
    }

    return website
