#!/usr/bin/env python3
"""
EXPONENTIAL ATLAS v6 -- Main runner
===================================
Generates complete website JSON from raw domain data.

This is the single entry point that:
1. Loads all 42 domain data files
2. Fits curves to all domains (BIC model selection)
3. Runs Monte Carlo simulations for all scenarios
4. Runs historical backtesting
5. Loads interaction data
6. Generates model card
7. Builds complete website JSON (v5-compatible superset)

Usage:
    python model/run.py                    # Full run (10K MC, all scenarios)
    python model/run.py --quick            # Quick run (500 MC, moderate only)
    python model/run.py --runs 5000        # Custom MC runs
    python model/run.py --sensitivity      # Include Sobol analysis (slow)
    python model/run.py --rsi 0.0 0.15 0.30  # RSI variants for website toggle
    python model/run.py --output out.json  # Custom output path
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Add project root to path so 'model' package is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Exponential Atlas v6 -- Generate website JSON"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run (500 MC runs, moderate scenario only)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20000,
        help="Number of Monte Carlo runs (default: 20000)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Include Sobol sensitivity analysis (slow)",
    )
    parser.add_argument(
        "--rsi",
        type=float,
        nargs="+",
        default=None,
        help="RSI exponent values to simulate (e.g., --rsi 0.0 0.15 0.30). "
             "If not set, only default 0.15 is used.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: atlas_v6_website_data.json)",
    )
    args = parser.parse_args()

    start = time.time()

    # ======================================================================
    # Step 1: Load domains
    # ======================================================================
    print("Step 1/7: Loading domain data...")
    from model.data.loader import load_all_domains

    domains = load_all_domains(validate=False)
    total_data_points = sum(
        len(d.get("data_points", [])) for d in domains.values()
    )
    print(f"  Loaded {len(domains)} domains, {total_data_points} data points")

    # ======================================================================
    # Step 2: Analyze all domains (fit curves)
    # ======================================================================
    print("Step 2/7: Fitting curves to all domains...")
    from model.simulation.analyze import analyze_domains

    analyses = analyze_domains(domains)
    summary = analyses.get("summary", {})
    print(
        f"  Fitted {summary.get('successful_fits', 0)} domains, "
        f"mean R2={summary.get('mean_r_squared', 0):.4f}"
    )
    if summary.get("failed_fits", 0) > 0:
        print(f"  WARNING: {summary['failed_fits']} domains failed to fit")

    # ======================================================================
    # Step 3: Run Monte Carlo simulation for all scenarios
    # ======================================================================
    print("Step 3/7: Running Monte Carlo simulations...")
    from model.simulation.config import (
        SimulationConfig,
        fit_all_data_domains,
        derive_base_rates,
        derive_rate_accelerations,
    )
    from model.simulation.monte_carlo import run_monte_carlo

    n_runs = 500 if args.quick else args.runs
    scenarios = ["moderate"] if args.quick else ["conservative", "moderate", "aggressive"]

    # Fit all data domains to derive base rates
    domain_fits = fit_all_data_domains()
    print(f"  Derived base rates from {len(domain_fits)} domain fits")

    # Helper to run all scenarios at a given RSI exponent value
    def _run_scenarios(rsi_exponent, label=""):
        results = {}
        for scenario in scenarios:
            tag = f" (RSI={rsi_exponent})" if label else ""
            print(f"  Running {scenario}{tag} ({n_runs} runs)...")
            config = SimulationConfig(
                scenario=scenario,
                n_runs=n_runs,
                recursive_self_improvement=rsi_exponent,
            )
            base_rates = derive_base_rates(domain_fits, config.sim_domains, scenario)
            config.base_rates = base_rates

            # Derive rate accelerations from historical data (second derivative)
            rate_accels = derive_rate_accelerations(config.sim_domains, scenario, domains)
            config.rate_accelerations = rate_accels
            n_accel = int(np.sum(rate_accels > 0))
            if scenario == scenarios[0] and not label:
                print(f"  Rate acceleration: {n_accel} domains show accelerating improvement")
                print(f"  Recursive self-improvement (AI): exponent={config.recursive_self_improvement}")

            result = run_monte_carlo(config, convergence_test=(n_runs >= 100))
            results[scenario] = result

            conv = result.convergence_status
            if conv.get("is_converged"):
                print(f"    Converged (max p50 change: {conv.get('max_p50_change', 0):.4f})")
            else:
                print(
                    f"    NOT converged (max p50 change: {conv.get('max_p50_change', 0):.4f} "
                    f"in {conv.get('domain_with_max_change', '?')})"
                )
        return results

    # Main simulation at default RSI=0.15
    sim_results = _run_scenarios(0.15)

    # RSI variants: run additional simulations at each requested RSI value
    rsi_variant_results = {}  # {rsi_value: {scenario: MonteCarloResult}}
    if args.rsi:
        rsi_values = sorted(set(args.rsi))
        print(f"\n  Running RSI variants: {rsi_values}")
        for rsi_val in rsi_values:
            if rsi_val == 0.15:
                # Default already computed -- reuse it
                rsi_variant_results[rsi_val] = sim_results
                print(f"  RSI={rsi_val}: reusing default results")
            else:
                rsi_variant_results[rsi_val] = _run_scenarios(rsi_val, label=f"RSI={rsi_val}")

    # ======================================================================
    # Step 4: Run backtest
    # ======================================================================
    print("Step 4/7: Running historical backtesting...")
    from model.validation.backtest import run_full_backtest
    from model import fits

    backtest = run_full_backtest(domains, fit_module=fits)
    bt_summary = backtest.summary
    n_comparisons = bt_summary.get("n_comparisons", 0)
    overall_mape = bt_summary.get("overall_mape")
    print(f"  Tested at {len(backtest.by_year)} cutoff years, {n_comparisons} comparisons")
    if overall_mape is not None:
        trimmed = bt_summary.get('overall_mape_trimmed_mean', overall_mape)
        raw_mean = bt_summary.get('overall_mape_raw_mean', overall_mape)
        print(f"  Median MAPE: {overall_mape:.1f}%, Trimmed mean: {trimmed:.1f}%, Bias: {bt_summary.get('bias_direction', '?')}")
        print(f"  Calibration factor: {backtest.calibration_factor:.4f}")

    # ======================================================================
    # Step 5: Load interactions
    # ======================================================================
    print("Step 5/7: Loading interaction data...")
    from model.interactions.matrix import load_interactions

    interactions = load_interactions()
    print(f"  {len(interactions)} interactions loaded")

    # ======================================================================
    # Step 6: Generate model card
    # ======================================================================
    print("Step 6/7: Generating model card...")
    from model.validation.model_card import generate_model_card

    # Pass the first sim result for convergence info (use moderate if available)
    sim_for_card = sim_results.get("moderate", next(iter(sim_results.values())))
    model_card = generate_model_card(
        domain_analyses=analyses,
        simulation_results=sim_for_card,
        backtest_results=backtest,
        interaction_data=interactions,
    )
    se = model_card.get("self_evaluation", {})
    print(f"  Self-evaluation: {se.get('overall_score', 'N/A')}")

    # ======================================================================
    # Step 6b: Sobol sensitivity analysis (if requested)
    # ======================================================================
    sensitivity_data = None
    if args.sensitivity:
        print("Step 6b: Running Sobol sensitivity analysis...")
        from model.validation.sensitivity import (
            run_sobol_analysis,
            define_parameters,
            add_tornado_to_result,
        )

        # Build a simulation wrapper that takes a parameter dict and returns
        # output metrics.  Uses reduced MC runs for performance.
        _sensitivity_mc_runs = min(200, n_runs)
        _sim_domains_list = list(SimulationConfig().sim_domains)
        _ai_idx = _sim_domains_list.index("ai") if "ai" in _sim_domains_list else 0

        def _sensitivity_simulate_fn(params: dict) -> dict:
            """Wrapper for Sobol analysis: takes param dict, returns output metrics."""
            # Extract base rates from params (domain_base_rate keys)
            config = SimulationConfig(
                scenario="moderate",
                n_runs=_sensitivity_mc_runs,
            )

            # Override base rates if provided in params
            base_rates_arr = derive_base_rates(domain_fits, config.sim_domains, "moderate")
            for i, dom in enumerate(config.sim_domains):
                rate_key = f"{dom}_base_rate"
                if rate_key in params:
                    # The sensitivity param is the log-space slope; convert to
                    # improvement factor the same way derive_base_rates does
                    import math as _math
                    base_rates_arr[i] = max(1.001, min(_math.exp(params[rate_key]), 3.0))
            config.base_rates = base_rates_arr

            # Override structural parameters
            if "base_gamma" in params:
                config.base_gamma = params["base_gamma"]
            if "noise_scale" in params:
                config.noise_std = 0.05 * params["noise_scale"]

            # Rate accelerations (use default moderate)
            config.rate_accelerations = derive_rate_accelerations(
                config.sim_domains, "moderate", domains
            )

            # Run a reduced MC simulation
            result = run_monte_carlo(config, convergence_test=False)

            # Extract output metrics: domain_2039_median for key domains
            outputs = {}
            for dom in config.sim_domains:
                pcts = result.percentiles.get(dom, {})
                year_2039 = pcts.get(2039, {})
                outputs[f"{dom}_2039_median"] = year_2039.get("p50", 1.0)
            return outputs

        # Define parameters (use fitted base rates as defaults)
        base_rate_defaults = {}
        for dom in _sim_domains_list:
            # Extract the fitted base rate as log-slope
            import math as _math
            idx = _sim_domains_list.index(dom)
            default_rates = derive_base_rates(domain_fits, _sim_domains_list, "moderate")
            base_rate_defaults[dom] = _math.log(max(default_rates[idx], 1.001))

        sensitivity_params = define_parameters(base_rate_defaults=base_rate_defaults)

        n_sobol_samples = 1024
        total_evals = n_sobol_samples * (2 + len(sensitivity_params))
        print(f"  {len(sensitivity_params)} parameters, N={n_sobol_samples}, "
              f"total evaluations: {total_evals:,}")
        print(f"  MC runs per evaluation: {_sensitivity_mc_runs}")
        print(f"  Estimated time: {total_evals * 0.5:.0f}s ({total_evals * 0.5 / 60:.0f}min)")

        sobol_start = time.time()

        def _progress(completed, total):
            if completed % 500 == 0 or completed == total:
                elapsed_s = time.time() - sobol_start
                rate = completed / elapsed_s if elapsed_s > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                print(f"    {completed}/{total} evaluations "
                      f"({elapsed_s:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        sensitivity_data = run_sobol_analysis(
            simulate_fn=_sensitivity_simulate_fn,
            parameters=sensitivity_params,
            n_samples=n_sobol_samples,
            n_bootstrap=200,
            seed=42,
            progress_callback=_progress,
        )

        # Add tornado data for key outputs
        tornado_outputs = ["ai_2039_median", "compute_2039_median", "energy_2039_median"]
        available_outputs = [o for o in tornado_outputs if o in sensitivity_data.outputs_analyzed]
        if available_outputs:
            print(f"  Computing tornado diagrams for: {available_outputs}")
            sensitivity_data = add_tornado_to_result(
                sensitivity_data, _sensitivity_simulate_fn, outputs=available_outputs,
            )

        sobol_elapsed = time.time() - sobol_start
        print(f"  Sobol analysis complete in {sobol_elapsed:.0f}s")

    # ======================================================================
    # Step 7: Build JSON
    # ======================================================================
    print("Step 7/7: Building website JSON...")
    from model.output.json_builder import build_website_json
    from model.output.compatibility import verify_v5_compatibility
    from model.output.compact import build_compact_json

    website_json = build_website_json(
        domain_analyses=analyses,
        simulation_results=sim_results,
        backtest_results=backtest,
        interaction_data=interactions,
        model_card=model_card,
        benchmark_comparisons=None,
        sensitivity_data=sensitivity_data,
        raw_domains=domains,
        rsi_variants=rsi_variant_results if rsi_variant_results else None,
    )

    # Verify backward compatibility
    issues = verify_v5_compatibility(website_json)
    if issues:
        print(f"  WARNING: {len(issues)} v5 compatibility issues:")
        for issue in issues[:10]:
            print(f"    - {issue}")
    else:
        print("  v5 compatibility: PASS")

    # ======================================================================
    # Save outputs
    # ======================================================================
    output_dir = _project_root
    full_path = args.output or os.path.join(output_dir, "atlas_v6_website_data.json")
    compact_path = full_path.replace("website_data", "compact")
    if compact_path == full_path:
        # Fallback if no 'website_data' in filename
        base, ext = os.path.splitext(full_path)
        compact_path = base + "_compact" + ext

    with open(full_path, "w") as f:
        json.dump(website_json, f, indent=2, cls=NpEncoder)

    compact = build_compact_json(website_json)
    with open(compact_path, "w") as f:
        json.dump(compact, f, indent=2, cls=NpEncoder)

    elapsed = time.time() - start

    # ======================================================================
    # Print summary
    # ======================================================================
    print(f"\n{'=' * 60}")
    print("EXPONENTIAL ATLAS v6 -- BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Domains: {website_json['meta']['domains']}")
    print(f"Data points: {website_json['meta']['data_points']}")
    print(f"Simulation domains: {website_json['meta']['simulation_domains']}")
    print(f"Interactions: {len(website_json['interactions'])}")
    print(f"MC runs: {n_runs} x {len(scenarios)} scenarios")
    print(f"Runtime: {elapsed:.1f}s")

    # Show 2039 moderate scenario medians
    mod_key = "moderate"
    if mod_key in sim_results:
        print(f"\n2039 Moderate Scenario (median improvement):")
        mod_sim = website_json["simulation"][mod_key]
        for dom in sorted(mod_sim.keys()):
            year_data = mod_sim[dom].get("2039", {})
            v = year_data.get("median", year_data.get("p50", 0))
            if isinstance(v, dict):
                v = v.get("median", v.get("p50", 0))
            if v is None or v == 0:
                s = "N/A"
            elif v > 1e9:
                s = f"{v:.1e}x"
            elif v > 1e6:
                s = f"{v / 1e6:.1f}Mx"
            elif v > 1e3:
                s = f"{v / 1e3:.1f}Kx"
            else:
                s = f"{v:.1f}x"
            print(f"  {dom:<20}: {s:>14}")

    full_size_kb = os.path.getsize(full_path) / 1024
    compact_size_kb = os.path.getsize(compact_path) / 1024
    print(f"\nFull JSON:    {full_path} ({full_size_kb:.0f}KB)")
    print(f"Compact JSON: {compact_path} ({compact_size_kb:.0f}KB)")
    print(f"\nReady for website build.")


if __name__ == "__main__":
    main()
