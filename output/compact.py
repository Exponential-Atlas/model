"""
Exponential Atlas v6 — Compact JSON Builder
=============================================
Strips the full JSON down to ~30-50KB for frontend embedding.

What is stripped:
- Individual data points (keep count only)
- Full Sobol indices (keep top 10 only)
- Tornado sweep data (keep summary only)
- Full evidence text (keep first 100 chars)
- Counter-evidence details
- Fit details beyond method + R-squared
- Selection notes
- All fits compared details

What is kept:
- All simulation percentiles
- All deployment percentiles
- Domain summaries
- Interaction weights and brief evidence
- Kings, costs, possibilities, forecasters
- Model card summary
- Methodology and weaknesses
"""

from __future__ import annotations

import copy


def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate text to max_len chars, adding ellipsis if needed."""
    if not isinstance(text, str):
        return str(text)[:max_len]
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _compact_domains(domains: dict) -> dict:
    """Strip data points and fit details from domain entries."""
    out = {}
    for did, d in domains.items():
        entry = {
            # Keep v5 fields
            "desc": d.get("desc"),
            "unit": d.get("unit"),
            "cat": d.get("cat"),
            "confidence": d.get("confidence"),
            "method": d.get("method"),
            "current": d.get("current"),
            "start": d.get("start"),
            "total_change": d.get("total_change"),
            "rate": d.get("rate"),
            "early_rate": d.get("early_rate"),
            "late_rate": d.get("late_rate"),
            "accelerating": d.get("accelerating"),
            "floor": d.get("floor"),
            "ceiling": d.get("ceiling"),
            "projections": d.get("projections", {}),
            "obs_learning_rate": d.get("obs_learning_rate"),
            # v6 summary fields
            "r_squared": d.get("r_squared"),
            "n_data_points": len(d.get("data_points", [])),
            # Strip data_points, fit_details, all_fits_compared
        }
        out[did] = entry
    return out


def _compact_interactions(interactions: list) -> list:
    """Truncate evidence and remove counter-evidence details."""
    out = []
    for ix in interactions:
        entry = {
            "from": ix.get("from"),
            "to": ix.get("to"),
            "weight": ix.get("weight"),
            "threshold": ix.get("threshold"),
            "evidence": _truncate(ix.get("evidence", ""), 100),
            "v5_weight": ix.get("v5_weight"),
            "saturation_model": ix.get("saturation_model"),
            # Strip: weight_justification, counter_evidence
        }
        out.append(entry)
    return out


def _compact_sensitivity(sensitivity: dict) -> dict:
    """Keep top 10 drivers only, strip sweep data."""
    if not sensitivity:
        return sensitivity

    out = {
        "note": sensitivity.get("note", ""),
    }

    # Top drivers: keep at most 10
    top_drivers = sensitivity.get("top_drivers", [])
    out["top_drivers"] = top_drivers[:10]

    # Sobol indices: keep top 10 per output
    sobol = sensitivity.get("sobol_indices", {})
    if sobol:
        compact_sobol = {}
        for order_type in ["first_order", "total_order"]:
            order_data = sobol.get(order_type, {})
            compact_order = {}
            for output_key, params in order_data.items():
                if isinstance(params, dict):
                    sorted_params = sorted(
                        params.items(), key=lambda kv: kv[1], reverse=True
                    )
                    compact_order[output_key] = dict(sorted_params[:10])
            compact_sobol[order_type] = compact_order
        out["sobol_indices"] = compact_sobol

    # Tornado: strip sweep data, keep summary only
    tornado = sensitivity.get("tornado_data", {})
    if tornado:
        compact_tornado = {}
        for output_key, tornado_entries in tornado.items():
            if isinstance(tornado_entries, list):
                compact_tornado[output_key] = [
                    {
                        "parameter": t.get("parameter"),
                        "impact": t.get("impact"),
                        "relative_impact": t.get("relative_impact"),
                        "output_at_low": t.get("output_at_low"),
                        "output_at_high": t.get("output_at_high"),
                        "output_at_default": t.get("output_at_default"),
                        # Strip: sweep, description, low_value, high_value, default_value
                    }
                    for t in tornado_entries[:10]  # top 10 only
                ]
        out["tornado_data"] = compact_tornado

    return out


def _compact_backtest(backtest: dict) -> dict:
    """Keep summary, strip per-prediction details."""
    if not backtest:
        return backtest

    return {
        "cutoff_years": backtest.get("cutoff_years"),
        "overall_mape": backtest.get("overall_mape"),
        "calibration_factor": backtest.get("calibration_factor"),
        "bias_direction": backtest.get("bias_direction"),
        "n_comparisons": backtest.get("n_comparisons"),
        # Strip per_domain and per_method details
    }


def _compact_model_card(model_card: dict) -> dict:
    """Keep summary fields only."""
    if not model_card:
        return model_card

    return {
        "title": model_card.get("title"),
        "version": model_card.get("version"),
        "purpose": _truncate(model_card.get("purpose", ""), 200),
        "domains_fitted": model_card.get("domains_fitted"),
        "mean_r_squared": model_card.get("mean_r_squared"),
        "interactions": model_card.get("interactions"),
        "limitations": model_card.get("limitations"),
    }


def _compact_forecasters(forecasters: dict) -> dict:
    """Keep structure but truncate notes."""
    if not forecasters:
        return forecasters

    out = {}
    for key, fc in forecasters.items():
        out[key] = {
            "name": fc.get("name"),
            "note": _truncate(fc.get("note", ""), 150),
            "predictions": fc.get("predictions", {}),
        }
    return out


def build_compact_json(full_json: dict) -> dict:
    """Build compact JSON for frontend embedding (~30-50KB target).

    Parameters
    ----------
    full_json : dict
        The full website JSON from build_website_json().

    Returns
    -------
    dict
        Compact version suitable for frontend embedding.
    """
    compact = {
        # Meta stays as-is (small)
        "meta": full_json.get("meta", {}),
        # Domains: stripped
        "domains": _compact_domains(full_json.get("domains", {})),
        # Simulation: keep all percentiles (this is the core data)
        "simulation": full_json.get("simulation", {}),
        # Deployment: keep all percentiles
        "deployment": full_json.get("deployment", {}),
        # Interactions: truncated evidence
        "interactions": _compact_interactions(full_json.get("interactions", [])),
        # Content databases: keep as-is (small)
        "kings": full_json.get("kings", []),
        "costs": full_json.get("costs", []),
        "possibilities": full_json.get("possibilities", []),
        # Forecasters: truncated
        "forecasters": _compact_forecasters(full_json.get("forecasters", {})),
        # Backtest: summary only
        "backtest": _compact_backtest(full_json.get("backtest", {})),
        # Methodology: keep as-is (small text)
        "methodology": full_json.get("methodology", {}),
        # Weaknesses: keep as-is (small text)
        "weaknesses": full_json.get("weaknesses", {}),
        # Sensitivity: compacted
        "sensitivity": _compact_sensitivity(full_json.get("sensitivity", {})),
        # Model card: summary
        "model_card": _compact_model_card(full_json.get("model_card", {})),
    }

    return compact
