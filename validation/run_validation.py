#!/usr/bin/env python3
"""
Exponential Atlas v6 — Full Validation Suite
=============================================

Runs the complete validation pipeline:

1. Load all 42 domain data files
2. Run domain analysis (fit + project + acceleration detect)
3. Run full backtest at cutoff years 2005, 2010, 2015, 2020
4. Run Monte Carlo simulation (moderate scenario, 10K runs)
5. Generate model card (without Sobol — that takes too long)
6. Print summary table and save model card JSON

Usage::

    .venv/bin/python3 model/validation/run_validation.py

    # Quick mode (fewer MC runs, faster):
    .venv/bin/python3 model/validation/run_validation.py --quick

    # Save model card to specific path:
    .venv/bin/python3 model/validation/run_validation.py --output model/output/model_card.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so imports work
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _timer():
    """Simple timer context manager."""
    class Timer:
        def __init__(self):
            self.start = time.time()
            self.elapsed = 0.0
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start
    return Timer()


def _print_header(text: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _print_subheader(text: str):
    """Print a formatted subsection header."""
    print(f"\n--- {text} ---")


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def run_validation(
    quick: bool = False,
    output_path: str = None,
    skip_simulation: bool = False,
) -> dict:
    """Run the full validation suite.

    Parameters
    ----------
    quick : bool
        If True, use 1000 MC runs instead of 10000 for faster execution.
    output_path : str, optional
        Path to save the model card JSON.  If None, saves to
        ``model/output/model_card.json``.
    skip_simulation : bool
        If True, skip Monte Carlo simulation (useful for testing
        just backtest + model card generation).

    Returns
    -------
    dict
        The generated model card.
    """
    total_start = time.time()

    _print_header("EXPONENTIAL ATLAS v6 -- VALIDATION SUITE")
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print(f"MC runs: {1000 if quick else 10000}")
    print(f"Sobol: SKIPPED (run separately for full analysis)")

    # ===================================================================
    # Step 1: Load all domains
    # ===================================================================
    _print_subheader("Step 1: Loading domain data")

    with _timer() as t1:
        from model.data.loader import load_all_domains
        domains = load_all_domains(validate=False)

    print(f"Loaded {len(domains)} domains in {t1.elapsed:.1f}s")

    # Quick summary
    total_points = sum(
        len(d.get("data_points", []))
        for d in domains.values()
    )
    print(f"Total data points: {total_points}")

    # ===================================================================
    # Step 2: Run domain analysis
    # ===================================================================
    _print_subheader("Step 2: Running domain analysis (fit + project)")

    with _timer() as t2:
        from model.simulation.analyze import analyze_domains
        domain_analyses = analyze_domains(domains=domains)

    summary = domain_analyses.get("summary", {})
    print(f"Completed in {t2.elapsed:.1f}s")
    print(f"  Successful fits: {summary.get('successful_fits', 0)}")
    print(f"  Failed fits:     {summary.get('failed_fits', 0)}")
    print(f"  Mean R-squared:  {summary.get('mean_r_squared', 0):.4f}")
    print(f"  Accelerating:    {summary.get('accelerating_count', 0)}")

    # Print fit method distribution
    from collections import Counter
    method_counts = Counter()
    for did, analysis in domain_analyses.get("domains", {}).items():
        m = analysis.get("best_fit_method", "failed")
        if "error" in analysis:
            m = "failed"
        method_counts[m] += 1

    print(f"\n  Fit method distribution:")
    for method, count in method_counts.most_common():
        print(f"    {method:<15} {count:>3}")

    # Print R-squared distribution
    r2_values = []
    for did, analysis in domain_analyses.get("domains", {}).items():
        r2 = analysis.get("r_squared")
        if r2 is not None and "error" not in analysis:
            r2_values.append((did, r2))

    r2_values.sort(key=lambda x: x[1])
    below_085 = [(d, r) for d, r in r2_values if r < 0.85]

    if below_085:
        print(f"\n  Domains below R-squared 0.85 ({len(below_085)}):")
        for did, r2 in below_085:
            method = domain_analyses["domains"][did].get("best_fit_method", "?")
            n_pts = domain_analyses["domains"][did].get("n_points", 0)
            print(f"    {did:<25} R2={r2:.4f}  method={method:<15} pts={n_pts}")

    # ===================================================================
    # Step 3: Load interactions
    # ===================================================================
    _print_subheader("Step 3: Loading interactions")

    with _timer() as t3:
        from model.interactions.matrix import (
            load_interactions,
            interaction_summary,
            validate_interactions,
        )
        interaction_data = load_interactions()
        int_summary = interaction_summary()
        int_warnings = validate_interactions()

    print(f"Loaded {len(interaction_data)} interactions in {t3.elapsed:.1f}s")
    print(f"  Matrix density: {int_summary['density']:.1%}")
    print(f"  Total weight:   {int_summary['total_weight']}")
    print(f"  Max interaction: {int_summary['max_interaction']}")

    if int_warnings:
        print(f"\n  Validation warnings ({len(int_warnings)}):")
        for w in int_warnings[:10]:
            print(f"    - {w}")
        if len(int_warnings) > 10:
            print(f"    ... and {len(int_warnings) - 10} more")
    else:
        print("  All interactions pass validation.")

    # ===================================================================
    # Step 4: Run full backtest
    # ===================================================================
    _print_subheader("Step 4: Running backtest (2005, 2010, 2015, 2020)")

    with _timer() as t4:
        from model.validation.backtest import run_full_backtest
        from model import fits as fits_module

        backtest_results = run_full_backtest(
            domains=domains,
            cutoff_years=[2005, 2010, 2015, 2020],
            fit_module=fits_module,
        )

    print(f"Completed in {t4.elapsed:.1f}s")
    print()
    print(backtest_results.summary_table())

    # Print by-method breakdown
    if backtest_results.by_method:
        print(f"\n  Accuracy by fit method:")
        print(f"    {'Method':<15} {'MAPE':>8} {'N domains':>10} {'Bias':<10}")
        print(f"    {'-'*45}")
        for method, stats in sorted(
            backtest_results.by_method.items(),
            key=lambda x: x[1].get("avg_mape", 999),
        ):
            print(
                f"    {method:<15} "
                f"{stats['avg_mape']:>7.1f}% "
                f"{stats['n_domains']:>10} "
                f"{stats['bias']:<10}"
            )

    # ===================================================================
    # Step 5: Run Monte Carlo simulation (optional)
    # ===================================================================
    simulation_results = None
    if not skip_simulation:
        n_mc_runs = 1000 if quick else 10000
        _print_subheader(
            f"Step 5: Running Monte Carlo simulation ({n_mc_runs:,} runs)"
        )

        with _timer() as t5:
            from model.simulation.config import (
                SimulationConfig,
                derive_base_rates,
                fit_all_data_domains,
            )
            from model.simulation.monte_carlo import run_monte_carlo

            # Fit all data domains to derive base rates
            print("  Fitting all data domains...")
            domain_fits = fit_all_data_domains()
            print(f"  Fitted {len(domain_fits)} data domains")

            # Create config
            config = SimulationConfig(
                scenario="moderate",
                n_runs=n_mc_runs,
                seed=42,
            )

            # Derive base rates from fits
            config.base_rates = derive_base_rates(
                domain_fits, config.sim_domains, config.scenario
            )
            print(f"  Derived base rates for {len(config.sim_domains)} sim domains")

            # Print base rates
            for i, dom in enumerate(config.sim_domains):
                rate = config.base_rates[i]
                print(f"    {dom:<15} {rate:.4f}x/year ({(rate-1)*100:+.1f}%)")

            # Run simulation
            print(f"\n  Running {n_mc_runs:,} Monte Carlo runs...")
            simulation_results = run_monte_carlo(
                config,
                domain_configs=domains,
            )

        print(f"Completed in {t5.elapsed:.1f}s")

        # Print convergence
        conv = simulation_results.convergence_status
        print(
            f"  Convergence: {'YES' if conv['is_converged'] else 'NO'} "
            f"(max p50 change: {conv['max_p50_change']:.6f} "
            f"in {conv['domain_with_max_change']})"
        )

        # Print 2039 p50 values
        print(f"\n  2039 median improvement factors (moderate scenario):")
        end_year = config.start_year + config.n_years
        for dom in config.sim_domains:
            pcts = simulation_results.percentiles.get(dom, {}).get(end_year, {})
            p10 = pcts.get("p10", 0)
            p50 = pcts.get("p50", 0)
            p90 = pcts.get("p90", 0)
            print(
                f"    {dom:<15} p10={p10:>8.1f}x  "
                f"p50={p50:>8.1f}x  p90={p90:>8.1f}x"
            )
    else:
        print("\n  [Skipping Monte Carlo simulation]")

    # ===================================================================
    # Step 6: Generate model card
    # ===================================================================
    _print_subheader("Step 6: Generating model card")

    with _timer() as t6:
        from model.validation.model_card import (
            generate_model_card,
            print_model_card_summary,
        )

        model_card = generate_model_card(
            domain_analyses=domain_analyses,
            simulation_results=simulation_results,
            backtest_results=backtest_results,
            interaction_data=interaction_data,
            sensitivity_data=None,  # Sobol skipped for quick validation
        )

    print(f"Generated in {t6.elapsed:.1f}s")
    print()

    # Print the full model card summary
    card_summary = print_model_card_summary(model_card)
    print(card_summary)

    # ===================================================================
    # Step 7: Save model card
    # ===================================================================
    if output_path is None:
        output_dir = _project_root / "model" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "model_card.json")

    _print_subheader(f"Step 7: Saving model card to {output_path}")

    # Ensure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, default=_json_serializer)

    file_size = os.path.getsize(output_path)
    print(f"Saved: {output_path} ({file_size:,} bytes)")

    # ===================================================================
    # Final summary
    # ===================================================================
    total_elapsed = time.time() - total_start

    _print_header("VALIDATION COMPLETE")

    se = model_card.get("self_evaluation", {})
    print(f"  Total runtime:    {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Scorecard:        {se.get('overall_score', '?')}")
    print(f"  Domains:          {len(domains)}")
    print(f"  Data points:      {total_points}")
    print(f"  Interactions:     {len(interaction_data)}")

    bt_summary = model_card.get("performance_metrics", {}).get("backtest", {})
    print(f"  Backtest MAPE:    {bt_summary.get('overall_mape', 'N/A')}%")
    print(f"  Backtest bias:    {bt_summary.get('bias_direction', 'N/A')}")

    if simulation_results is not None:
        conv = model_card.get("performance_metrics", {}).get("convergence", {})
        print(f"  MC converged:     {'Yes' if conv.get('converged') else 'No'}")

    print(f"  Model card:       {output_path}")
    print()

    return model_card


def _json_serializer(obj):
    """Custom JSON serializer for numpy types and other non-standard types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "__dict__"):
        # Dataclass or similar — convert to dict
        return {
            k: v for k, v in obj.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exponential Atlas v6 — Full Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  .venv/bin/python3 model/validation/run_validation.py
  .venv/bin/python3 model/validation/run_validation.py --quick
  .venv/bin/python3 model/validation/run_validation.py --output results/card.json
  .venv/bin/python3 model/validation/run_validation.py --skip-simulation
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1000 MC runs instead of 10000",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for model card JSON",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip Monte Carlo simulation (faster, backtest + card only)",
    )

    args = parser.parse_args()

    try:
        run_validation(
            quick=args.quick,
            output_path=args.output,
            skip_simulation=args.skip_simulation,
        )
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nValidation failed with error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
