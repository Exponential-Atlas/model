"""
Exponential Atlas v6 — Validation & Sensitivity Analysis Package
=================================================================

This package provides four validation capabilities:

1. **Sobol Sensitivity Analysis** (``sensitivity.py``)
   First-order and total-order Sobol indices using the Saltelli method,
   plus tornado diagram data for parameter impact visualisation.

2. **Backtesting** (``backtest.py``)
   Fit-and-predict at historical cutoff years (2005, 2010, 2015, 2020)
   to quantify model accuracy on data it has never seen.

3. **External Benchmarks** (``benchmarks.py``)
   Structured comparison of Atlas projections against ARK Invest, IEA,
   Epoch AI, BloombergNEF, Metaculus, and IPCC forecasts — each with a
   quantitative prediction and source URL.

4. **Model Card** (``model_card.py``)
   Comprehensive model card generation following Mitchell et al. 2019,
   with all sections populated from actual model outputs.

Quick usage::

    from model.validation import (
        # Sensitivity
        SensitivityParameter,
        SobolResult,
        define_parameters,
        run_sobol_analysis,
        compute_tornado_data,
        # Backtesting
        backtest_at_year,
        run_full_backtest,
        # Benchmarks
        EXTERNAL_FORECASTS,
        compare_to_benchmarks,
        # Model Card
        generate_model_card,
        print_model_card_summary,
    )
"""

# Sensitivity analysis
from .sensitivity import (
    SensitivityParameter,
    SobolResult,
    define_parameters,
    run_sobol_analysis,
    compute_tornado_data,
)

# Backtesting
from .backtest import (
    backtest_at_year,
    run_full_backtest,
)

# External benchmarks
from .benchmarks import (
    EXTERNAL_FORECASTS,
    compare_to_benchmarks,
)

# Model card
from .model_card import (
    generate_model_card,
    print_model_card_summary,
)

__all__ = [
    # Sensitivity
    "SensitivityParameter",
    "SobolResult",
    "define_parameters",
    "run_sobol_analysis",
    "compute_tornado_data",
    # Backtesting
    "backtest_at_year",
    "run_full_backtest",
    # External benchmarks
    "EXTERNAL_FORECASTS",
    "compare_to_benchmarks",
    # Model card
    "generate_model_card",
    "print_model_card_summary",
]
