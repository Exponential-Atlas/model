"""
Exponential Atlas v6 — External Benchmark Comparisons
======================================================

Structured comparison of Atlas projections against major external
forecasters.  Each prediction includes a specific quantitative claim,
a year, and a source URL.

This module serves two purposes:

1. **Calibration** — If the Atlas consistently disagrees with all
   external forecasters in the same direction, that's a signal
   to investigate.

2. **Transparency** — Users can see exactly how the Atlas compares
   to ARK Invest, IEA, Epoch AI, BloombergNEF, Metaculus, and IPCC.
   Agreement or disagreement is documented with context for why.

Note on source URLs: These point to the best publicly available
version of each forecast as of early 2026.  Some may require
institutional access or may have been updated.  The URLs are provided
for traceability, not guaranteed permanence.

How to read the comparisons:
    - "agrees" means Atlas projection is within 30% of the benchmark.
    - "more_aggressive" means Atlas projects faster progress.
    - "more_conservative" means Atlas projects slower progress.
    - "different_metric" means direct comparison is not possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# External forecasts database
# ---------------------------------------------------------------------------

EXTERNAL_FORECASTS: dict[str, dict] = {
    # =======================================================================
    # ARK Invest
    # =======================================================================
    "ark_invest": {
        "name": "ARK Invest Big Ideas 2024-2026",
        "methodology": (
            "Bottom-up sector analysis with independent domain models. "
            "Known for aggressive technology adoption assumptions. "
            "Track record: correctly predicted Tesla and Bitcoin growth, "
            "missed on Zoom and genomics timelines."
        ),
        "credibility_note": (
            "ARK's forecasts are consistently among the most optimistic "
            "from any institutional forecaster.  Their methodology is "
            "bottom-up and well-documented but their error bars are "
            "extremely wide."
        ),
        "predictions": [
            {
                "domain": "ai",
                "metric": "AI software market revenue",
                "year": 2030,
                "value": "$13T cumulative AI software revenue",
                "value_numeric": 13e12,
                "unit": "USD",
                "source_url": "https://ark-invest.com/big-ideas-2025",
                "context": (
                    "ARK's most headline number.  Assumes AI productivity "
                    "gains across all knowledge work.  Upper bound of their "
                    "range."
                ),
            },
            {
                "domain": "ai",
                "metric": "Cost per AI training run equivalent to GPT-4",
                "year": 2030,
                "value": "~$300 (vs ~$100M in 2023)",
                "value_numeric": 300,
                "unit": "USD per GPT-4-equivalent training run",
                "source_url": "https://ark-invest.com/big-ideas-2025",
                "context": (
                    "Based on ~10x annual cost decline in training compute.  "
                    "Very aggressive assumption."
                ),
            },
            {
                "domain": "robotics",
                "metric": "Humanoid robot market",
                "year": 2030,
                "value": "$24B humanoid robot market",
                "value_numeric": 24e9,
                "unit": "USD",
                "source_url": "https://ark-invest.com/big-ideas-2025",
                "context": (
                    "Assumes Tesla Optimus and competitors reach volume "
                    "production.  Consensus is much smaller (~$5B)."
                ),
            },
            {
                "domain": "batteries",
                "metric": "Battery pack cost",
                "year": 2030,
                "value": "$55/kWh battery pack cost",
                "value_numeric": 55,
                "unit": "USD/kWh",
                "source_url": "https://ark-invest.com/big-ideas-2024",
                "context": (
                    "Extrapolation of Wright's Law at historical learning "
                    "rates.  BloombergNEF 2024 data shows $115/kWh in 2024 "
                    "(pack level)."
                ),
            },
        ],
    },

    # =======================================================================
    # IEA (International Energy Agency)
    # =======================================================================
    "iea": {
        "name": "IEA World Energy Outlook 2024",
        "methodology": (
            "Integrated energy system modeling with three scenarios: "
            "Stated Policies (STEPS), Announced Pledges (APS), and "
            "Net Zero Emissions (NZE).  IEA has historically "
            "under-predicted renewable energy growth — their forecasts "
            "for solar capacity have been revised upward every year for "
            "over a decade."
        ),
        "credibility_note": (
            "The IEA is the authoritative institutional voice on energy "
            "but has a documented track record of underestimating solar "
            "and wind deployment.  Their 2010 forecast for solar in 2020 "
            "was off by ~10x."
        ),
        "predictions": [
            {
                "domain": "energy",
                "metric": "Solar becomes largest electricity source globally",
                "year": 2033,
                "value": "Solar overtakes coal as #1 electricity source (STEPS scenario)",
                "value_numeric": None,
                "unit": "milestone",
                "source_url": "https://www.iea.org/reports/world-energy-outlook-2024",
                "context": (
                    "In NZE scenario, this happens by 2030.  Given IEA's "
                    "track record of underestimation, the STEPS date of "
                    "2033 may be conservative."
                ),
            },
            {
                "domain": "energy",
                "metric": "Solar PV LCOE",
                "year": 2030,
                "value": "$20-30/MWh global average solar LCOE (utility scale)",
                "value_numeric": 25,
                "unit": "USD/MWh",
                "source_url": "https://www.iea.org/reports/world-energy-outlook-2024",
                "context": (
                    "Best-resource regions already at $15/MWh in 2024 "
                    "(IRENA).  Global average includes less optimal sites."
                ),
            },
            {
                "domain": "energy",
                "metric": "Global renewable electricity share",
                "year": 2030,
                "value": "~50% of global electricity from renewables (STEPS)",
                "value_numeric": 50,
                "unit": "percent",
                "source_url": "https://www.iea.org/reports/world-energy-outlook-2024",
                "context": (
                    "Up from ~30% in 2023.  IEA STEPS scenario.  NZE "
                    "scenario projects higher."
                ),
            },
            {
                "domain": "environment",
                "metric": "Clean energy investment",
                "year": 2030,
                "value": "$2.7T annual clean energy investment (NZE pathway)",
                "value_numeric": 2.7e12,
                "unit": "USD/year",
                "source_url": "https://www.iea.org/reports/world-energy-outlook-2024",
                "context": (
                    "2024 clean energy investment was ~$2T.  NZE requires "
                    "~35% increase."
                ),
            },
        ],
    },

    # =======================================================================
    # Epoch AI
    # =======================================================================
    "epoch_ai": {
        "name": "Epoch AI (2024-2025)",
        "methodology": (
            "Empirical trend analysis of ML systems, training compute, "
            "hardware efficiency, and algorithmic progress.  Epoch is the "
            "most rigorous quantitative tracker of AI progress."
        ),
        "credibility_note": (
            "Epoch AI is widely regarded as the gold standard for empirical "
            "AI trend analysis.  Their data is used by OECD, US government, "
            "and most major AI labs."
        ),
        "predictions": [
            {
                "domain": "ai",
                "metric": "Training compute growth rate",
                "year": 2030,
                "value": "~4.1x/year growth in compute used by notable ML systems",
                "value_numeric": 4.1,
                "unit": "x/year growth rate",
                "source_url": "https://epochai.org/trends-in-machine-learning",
                "context": (
                    "This is the observed trend, not a forecast.  Whether "
                    "it sustains depends on capital investment and hardware "
                    "availability."
                ),
            },
            {
                "domain": "ai",
                "metric": "Algorithmic efficiency improvement",
                "year": 2030,
                "value": "~2x efficiency gain every 8-9 months for vision models",
                "value_numeric": 2.0,
                "unit": "x improvement per 8-9 months",
                "source_url": "https://epochai.org/blog/algorithmic-progress-in-language-models",
                "context": (
                    "Algorithmic progress is as important as compute scaling "
                    "but harder to measure.  Epoch's estimates are the "
                    "most careful available."
                ),
            },
            {
                "domain": "compute",
                "metric": "GPU price-performance trend",
                "year": 2030,
                "value": "~2.3x/year improvement in GPU FLOP/$ for ML workloads",
                "value_numeric": 2.3,
                "unit": "x/year improvement",
                "source_url": "https://epochai.org/blog/trends-in-gpu-price-performance",
                "context": (
                    "Faster than Moore's Law (~1.4x/year) due to "
                    "architecture specialisation (tensor cores, sparsity)."
                ),
            },
            {
                "domain": "ai",
                "metric": "Training data stock exhaustion",
                "year": 2028,
                "value": "High-quality text data for LLM training may be exhausted by 2028",
                "value_numeric": None,
                "unit": "milestone",
                "source_url": "https://epochai.org/blog/will-we-run-out-of-data",
                "context": (
                    "A potential hard limit on scaling.  Mitigations: "
                    "synthetic data, multimodal data, more efficient "
                    "learning."
                ),
            },
        ],
    },

    # =======================================================================
    # BloombergNEF
    # =======================================================================
    "bloombergnef": {
        "name": "BloombergNEF (BNEF) Energy Outlook 2024-2025",
        "methodology": (
            "Bottom-up technology cost and deployment modeling.  BNEF's "
            "battery and solar cost tracking is the industry standard.  "
            "Their forecasts are more aggressive than IEA but less "
            "aggressive than ARK."
        ),
        "credibility_note": (
            "BNEF's technology cost data (especially batteries) is the "
            "most widely cited in the energy industry.  Their annual "
            "battery price survey is considered definitive."
        ),
        "predictions": [
            {
                "domain": "batteries",
                "metric": "Lithium-ion battery pack price",
                "year": 2030,
                "value": "$72/kWh average pack price (cell: ~$50/kWh)",
                "value_numeric": 72,
                "unit": "USD/kWh (pack level)",
                "source_url": "https://about.bnef.com/blog/lithium-ion-battery-pack-prices-hit-record-low-of-115-per-kilowatt-hour/",
                "context": (
                    "2024 average was ~$115/kWh (pack).  BNEF projects "
                    "continued Wright's Law decline.  China already below "
                    "$100/kWh for LFP."
                ),
            },
            {
                "domain": "energy",
                "metric": "Global EV share of new car sales",
                "year": 2030,
                "value": "~45% of new passenger cars are electric",
                "value_numeric": 45,
                "unit": "percent of new sales",
                "source_url": "https://about.bnef.com/electric-vehicle-outlook/",
                "context": (
                    "2024 was ~20% globally.  China already at ~40%.  "
                    "Europe and US lagging."
                ),
            },
            {
                "domain": "energy",
                "metric": "Annual solar installations",
                "year": 2030,
                "value": "~800 GW annual solar PV installations",
                "value_numeric": 800,
                "unit": "GW/year",
                "source_url": "https://about.bnef.com/blog/global-solar-capacity-additions/",
                "context": (
                    "2024 was ~500 GW.  China drives most growth.  "
                    "Supply chain constraints may limit upside."
                ),
            },
            {
                "domain": "environment",
                "metric": "Green hydrogen cost",
                "year": 2030,
                "value": "$2.0-3.0/kg green hydrogen (best locations: $1.5/kg)",
                "value_numeric": 2.5,
                "unit": "USD/kg",
                "source_url": "https://about.bnef.com/blog/hydrogen-economy-outlook-key-messages/",
                "context": (
                    "Requires cheap electricity ($10-20/MWh) and "
                    "electrolyzer cost reduction to ~$200/kW."
                ),
            },
        ],
    },

    # =======================================================================
    # Metaculus
    # =======================================================================
    "metaculus": {
        "name": "Metaculus Community & AI Forecasts (2024-2025)",
        "methodology": (
            "Crowd-sourced probabilistic forecasting platform with "
            "calibrated forecasters.  Metaculus forecasters have strong "
            "track records on technology predictions.  Questions are "
            "resolved based on specific, well-defined criteria."
        ),
        "credibility_note": (
            "Metaculus crowd forecasts have been shown to outperform "
            "individual experts on many technology questions.  Their "
            "AI-specific forecasts have been particularly well-calibrated."
        ),
        "predictions": [
            {
                "domain": "ai",
                "metric": "Artificial General Intelligence (AGI) arrival",
                "year": 2032,
                "value": "50th percentile estimate for AGI (weakly defined) by 2032",
                "value_numeric": 2032,
                "unit": "year",
                "source_url": "https://www.metaculus.com/questions/5121/date-of-artificial-general-intelligence/",
                "context": (
                    "Definition-dependent.  Metaculus uses 'when AI can "
                    "perform any cognitive task a human can.'  25th "
                    "percentile: ~2028.  75th percentile: ~2040."
                ),
            },
            {
                "domain": "ai",
                "metric": "AI passes Turing Test",
                "year": 2029,
                "value": "Median forecast: AI convincingly passes Turing Test by 2029",
                "value_numeric": 2029,
                "unit": "year",
                "source_url": "https://www.metaculus.com/questions/274/ai-turing-test/",
                "context": (
                    "Some argue GPT-4 already passes weak versions.  "
                    "The Metaculus question specifies a rigorous protocol."
                ),
            },
            {
                "domain": "space",
                "metric": "SpaceX Starship cost to LEO",
                "year": 2030,
                "value": "<$100/kg to LEO (median forecast)",
                "value_numeric": 100,
                "unit": "USD/kg to LEO",
                "source_url": "https://www.metaculus.com/questions/9550/spacex-starship-launch-cost/",
                "context": (
                    "Current Falcon 9 is ~$2,700/kg.  SpaceX targets "
                    "$10/kg with Starship full reusability.  Metaculus "
                    "median is more conservative."
                ),
            },
            {
                "domain": "quantum",
                "metric": "Quantum advantage for practical problem",
                "year": 2031,
                "value": "Median forecast: quantum advantage for a commercially useful problem by 2031",
                "value_numeric": 2031,
                "unit": "year",
                "source_url": "https://www.metaculus.com/questions/5733/quantum-advantage-for-practical-problem/",
                "context": (
                    "Not arbitrary benchmarks (Sycamore/Willow) but "
                    "actual commercial utility.  Drug discovery or "
                    "materials simulation most likely first applications."
                ),
            },
        ],
    },

    # =======================================================================
    # IPCC
    # =======================================================================
    "ipcc": {
        "name": "IPCC AR6 & Special Reports (2021-2023)",
        "methodology": (
            "Comprehensive literature synthesis by hundreds of climate "
            "scientists.  IPCC does not forecast technology but projects "
            "technology deployment in climate scenarios.  Their estimates "
            "of technology needs imply specific cost trajectories."
        ),
        "credibility_note": (
            "IPCC is the most authoritative voice on climate science.  "
            "Their technology deployment estimates in mitigation scenarios "
            "are conservative relative to observed deployment rates — "
            "e.g., their 2018 projection for 2030 solar was exceeded by "
            "2023 actuals."
        ),
        "predictions": [
            {
                "domain": "energy",
                "metric": "Solar PV capacity needed for 1.5C pathway",
                "year": 2030,
                "value": "~5,000 GW cumulative solar PV (up from ~1,600 GW in 2023)",
                "value_numeric": 5000,
                "unit": "GW cumulative",
                "source_url": "https://www.ipcc.ch/report/ar6/wg3/",
                "context": (
                    "Requires ~500-600 GW annual additions.  2024 was ~500 "
                    "GW — already on track for this milestone."
                ),
            },
            {
                "domain": "environment",
                "metric": "Carbon capture needed for 1.5C",
                "year": 2030,
                "value": "~1 GtCO2/year carbon removal needed by 2030 (1.5C scenario)",
                "value_numeric": 1e9,
                "unit": "tonnes CO2/year",
                "source_url": "https://www.ipcc.ch/report/ar6/wg3/",
                "context": (
                    "Current DAC capacity is ~0.01 MtCO2/year — 100,000x "
                    "gap.  BECCS and nature-based solutions assumed to "
                    "provide most of this."
                ),
            },
            {
                "domain": "environment",
                "metric": "Global emissions pathway for 1.5C",
                "year": 2030,
                "value": "Global CO2 emissions must fall 43% from 2019 levels by 2030",
                "value_numeric": -43,
                "unit": "percent change from 2019",
                "source_url": "https://www.ipcc.ch/report/ar6/syr/",
                "context": (
                    "Current trajectory: emissions roughly flat or slowly "
                    "rising.  The gap between needed and actual is the "
                    "defining challenge."
                ),
            },
            {
                "domain": "energy",
                "metric": "Electricity share of final energy",
                "year": 2050,
                "value": "~50-65% of final energy is electricity (up from ~20% in 2023)",
                "value_numeric": 57,
                "unit": "percent",
                "source_url": "https://www.ipcc.ch/report/ar6/wg3/",
                "context": (
                    "Electrification of transport, heating, and industry "
                    "is the core decarbonisation pathway."
                ),
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Comparison structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkComparison:
    """One comparison between Atlas and an external forecast.

    Attributes
    ----------
    forecaster : str
        Key into EXTERNAL_FORECASTS, e.g. ``"ark_invest"``.
    forecaster_name : str
        Human-readable name.
    domain : str
        Atlas simulation domain.
    metric : str
        What is being compared.
    year : int
        Comparison year.
    external_value : str
        The external forecast's prediction (human-readable).
    external_numeric : float or None
        Numeric value if comparable.
    atlas_value : float or None
        Atlas prediction (numeric).
    atlas_description : str
        Human-readable Atlas prediction.
    agreement : str
        One of: ``"agrees"``, ``"more_aggressive"``,
        ``"more_conservative"``, ``"different_metric"``.
    notes : str
        Explanation of agreement or disagreement.
    source_url : str
        URL for the external forecast.
    """

    forecaster: str
    forecaster_name: str
    domain: str
    metric: str
    year: int
    external_value: str
    external_numeric: Optional[float]
    atlas_value: Optional[float]
    atlas_description: str
    agreement: str
    notes: str
    source_url: str


@dataclass
class BenchmarkReport:
    """Full benchmark comparison report.

    Attributes
    ----------
    year : int
        Comparison year.
    comparisons : list[BenchmarkComparison]
        All individual comparisons.
    summary : dict
        ``{"agrees": int, "more_aggressive": int, "more_conservative": int,
        "different_metric": int}``.
    by_domain : dict[str, list[BenchmarkComparison]]
        Comparisons grouped by domain.
    by_forecaster : dict[str, list[BenchmarkComparison]]
        Comparisons grouped by forecaster.
    """

    year: int = 2030
    comparisons: list[BenchmarkComparison] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    by_domain: dict[str, list[BenchmarkComparison]] = field(
        default_factory=dict
    )
    by_forecaster: dict[str, list[BenchmarkComparison]] = field(
        default_factory=dict
    )

    def summary_table(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Exponential Atlas v6 — Benchmark Comparison @ {self.year}",
            "=" * 65,
            "",
            f"{'Forecaster':<20} {'#':>3} {'Agree':>6} {'More Agg':>9} "
            f"{'More Con':>9} {'Diff':>6}",
            "-" * 55,
        ]
        for fc_key, fc_comps in sorted(self.by_forecaster.items()):
            n = len(fc_comps)
            agree = sum(1 for c in fc_comps if c.agreement == "agrees")
            agg = sum(1 for c in fc_comps if c.agreement == "more_aggressive")
            con = sum(
                1 for c in fc_comps if c.agreement == "more_conservative"
            )
            diff = sum(
                1 for c in fc_comps if c.agreement == "different_metric"
            )
            name = EXTERNAL_FORECASTS[fc_key]["name"][:18]
            lines.append(
                f"{name:<20} {n:>3} {agree:>6} {agg:>9} {con:>9} {diff:>6}"
            )

        lines.append("")
        lines.append(f"Total comparisons: {len(self.comparisons)}")
        s = self.summary
        lines.append(
            f"  Agrees: {s.get('agrees', 0)}  |  "
            f"More aggressive: {s.get('more_aggressive', 0)}  |  "
            f"More conservative: {s.get('more_conservative', 0)}  |  "
            f"Different metric: {s.get('different_metric', 0)}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------

def compare_to_benchmarks(
    atlas_predictions: dict[str, float],
    year: int = 2030,
    comparison_threshold: float = 0.30,
) -> BenchmarkReport:
    """Compare Atlas predictions to external forecasters at a specific year.

    Parameters
    ----------
    atlas_predictions : dict[str, float]
        Atlas predictions keyed by strings that include domain names,
        e.g. ``{"ai_2030_median": 1e6, "energy_2030_median": 15.0}``.
        The function will attempt to match these to external forecasts by
        domain name and year.
    year : int
        Year to compare at (default 2030).
    comparison_threshold : float
        Fraction difference threshold for "agrees" classification.
        Default 0.30 (30%).

    Returns
    -------
    BenchmarkReport
        Structured comparison results.
    """
    report = BenchmarkReport(year=year)

    # Build lookup of Atlas predictions by domain
    # We try to find predictions matching the pattern: {domain}_{year}_*
    atlas_by_domain: dict[str, dict[str, float]] = {}
    for key, val in atlas_predictions.items():
        for domain in [
            "ai", "compute", "energy", "batteries", "genomics", "drug",
            "robotics", "space", "manufacturing", "materials", "bci",
            "quantum", "environment", "vr", "sensors",
        ]:
            if key.startswith(f"{domain}_"):
                if domain not in atlas_by_domain:
                    atlas_by_domain[domain] = {}
                atlas_by_domain[domain][key] = val
                break

    for fc_key, fc_data in EXTERNAL_FORECASTS.items():
        fc_name = fc_data["name"]

        for pred in fc_data["predictions"]:
            # Only compare predictions at the requested year
            pred_year = pred.get("year", 0)
            if abs(pred_year - year) > 3:
                # Skip predictions too far from requested year
                continue

            domain = pred["domain"]
            metric = pred["metric"]
            ext_value = pred["value"]
            ext_numeric = pred.get("value_numeric")
            source_url = pred["source_url"]

            # Try to find matching Atlas prediction
            atlas_val = None
            atlas_desc = "No matching Atlas prediction"
            agreement = "different_metric"
            notes = ""

            if domain in atlas_by_domain:
                # Look for median or main prediction
                domain_preds = atlas_by_domain[domain]
                # Try common key patterns
                for suffix in [
                    f"_{year}_median",
                    f"_{year}",
                    f"_{year}_mean",
                    f"_{year}_p50",
                ]:
                    candidate_key = f"{domain}{suffix}"
                    if candidate_key in domain_preds:
                        atlas_val = domain_preds[candidate_key]
                        atlas_desc = f"Atlas {domain} @ {year}: {atlas_val:.4g}"
                        break

                # If we found a numeric Atlas value and external is numeric,
                # compute agreement
                if atlas_val is not None and ext_numeric is not None:
                    if ext_numeric == 0:
                        agreement = "different_metric"
                        notes = "External value is zero; cannot compute ratio."
                    else:
                        ratio = atlas_val / ext_numeric
                        if abs(ratio - 1.0) <= comparison_threshold:
                            agreement = "agrees"
                            notes = (
                                f"Atlas/External ratio: {ratio:.2f} "
                                f"(within {comparison_threshold*100:.0f}% "
                                f"threshold)"
                            )
                        elif ratio > 1.0 + comparison_threshold:
                            agreement = "more_aggressive"
                            notes = (
                                f"Atlas/External ratio: {ratio:.2f} "
                                f"(Atlas projects {(ratio-1)*100:.0f}% more "
                                f"than {fc_name})"
                            )
                        else:
                            agreement = "more_conservative"
                            notes = (
                                f"Atlas/External ratio: {ratio:.2f} "
                                f"(Atlas projects {(1-ratio)*100:.0f}% less "
                                f"than {fc_name})"
                            )
                elif atlas_val is not None and ext_numeric is None:
                    agreement = "different_metric"
                    notes = (
                        "External prediction is qualitative; "
                        "direct numeric comparison not possible."
                    )
                else:
                    notes = "No numeric Atlas prediction available for this domain."
            else:
                notes = f"No Atlas prediction available for domain '{domain}'."

            comp = BenchmarkComparison(
                forecaster=fc_key,
                forecaster_name=fc_name,
                domain=domain,
                metric=metric,
                year=pred_year,
                external_value=ext_value,
                external_numeric=ext_numeric,
                atlas_value=atlas_val,
                atlas_description=atlas_desc,
                agreement=agreement,
                notes=notes,
                source_url=source_url,
            )
            report.comparisons.append(comp)

            # Group by domain
            report.by_domain.setdefault(domain, []).append(comp)

            # Group by forecaster
            report.by_forecaster.setdefault(fc_key, []).append(comp)

    # Summary counts
    report.summary = {
        "agrees": sum(
            1 for c in report.comparisons if c.agreement == "agrees"
        ),
        "more_aggressive": sum(
            1 for c in report.comparisons
            if c.agreement == "more_aggressive"
        ),
        "more_conservative": sum(
            1 for c in report.comparisons
            if c.agreement == "more_conservative"
        ),
        "different_metric": sum(
            1 for c in report.comparisons
            if c.agreement == "different_metric"
        ),
        "total": len(report.comparisons),
    }

    return report


# ---------------------------------------------------------------------------
# Utility: list all external predictions
# ---------------------------------------------------------------------------

def list_all_predictions() -> list[dict]:
    """Return a flat list of all external predictions for inspection.

    Returns
    -------
    list[dict]
        Each dict contains ``forecaster``, ``domain``, ``metric``,
        ``year``, ``value``, ``source_url``.
    """
    result = []
    for fc_key, fc_data in EXTERNAL_FORECASTS.items():
        for pred in fc_data["predictions"]:
            result.append({
                "forecaster": fc_key,
                "forecaster_name": fc_data["name"],
                "domain": pred["domain"],
                "metric": pred["metric"],
                "year": pred["year"],
                "value": pred["value"],
                "value_numeric": pred.get("value_numeric"),
                "unit": pred.get("unit"),
                "source_url": pred["source_url"],
            })
    return result


def count_predictions() -> dict[str, int]:
    """Count predictions by forecaster and domain.

    Returns
    -------
    dict[str, int]
        ``{"total": int, "by_forecaster": {...}, "by_domain": {...}}``.
    """
    by_fc: dict[str, int] = {}
    by_domain: dict[str, int] = {}

    for fc_key, fc_data in EXTERNAL_FORECASTS.items():
        fc_count = len(fc_data["predictions"])
        by_fc[fc_key] = fc_count
        for pred in fc_data["predictions"]:
            d = pred["domain"]
            by_domain[d] = by_domain.get(d, 0) + 1

    return {
        "total": sum(by_fc.values()),
        "by_forecaster": by_fc,
        "by_domain": by_domain,
    }
