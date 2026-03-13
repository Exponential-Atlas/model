"""
Exponential Atlas v6 — Sobol Sensitivity Analysis
===================================================

Implements variance-based global sensitivity analysis using the Saltelli
(2002, 2010) method with Sobol quasi-random sequences for sampling.

The key insight: we need to know **which parameters matter** and **how much**.
If the model is dominated by 3 of 30 parameters, that tells us exactly where
better data would reduce uncertainty the most.

Expected findings (documented before running — a sign of intellectual honesty):
    - AI base rate and the AI->AI self-improvement weight are expected to
      dominate total-order indices for most outputs, because AI interacts
      with nearly every other domain.
    - Structural parameters (gamma, breakthrough_scale) should have high
      total-order indices because they affect *all* domains simultaneously.
    - Most individual interaction weights should have low first-order indices
      but higher total-order indices, reflecting their participation in
      cross-domain feedback loops.
    - For energy-related outputs, energy base rate + energy->environment
      weight should dominate locally.

Algorithm
---------
We follow the Saltelli (2010) estimator for Sobol indices:

1. Generate two independent Sobol quasi-random matrices **A** and **B**,
   each of shape ``(N, D)`` where ``N = n_samples`` and ``D = n_params``.

2. For each parameter *i*, construct matrix **AB_i** by taking **A** but
   replacing column *i* with the corresponding column from **B**.

3. Evaluate the model at all ``N * (2 + D)`` points: rows of A, rows of B,
   and rows of each AB_i.

4. Compute first-order indices::

       S_i = (1/N) * sum(f(B) * (f(AB_i) - f(A))) / Var(Y)

5. Compute total-order indices::

       ST_i = (1/(2N)) * sum((f(A) - f(AB_i))^2) / Var(Y)

6. Bootstrap for confidence intervals.

References
----------
- Saltelli, A. (2002). "Making best use of model evaluations to compute
  sensitivity indices." Computer Physics Communications 145(2): 280-297.
- Saltelli, A. et al. (2010). "Variance based sensitivity analysis of model
  output." Computer Physics Communications 181(2): 259-270.
- Sobol, I.M. (2001). "Global sensitivity indices for nonlinear mathematical
  models and their Monte Carlo estimates." Mathematics and Computers in
  Simulation 55(1-3): 271-280.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.stats.qmc import Sobol as SobolEngine


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensitivityParameter:
    """A single parameter to be varied in sensitivity analysis.

    Attributes
    ----------
    name : str
        Machine-readable identifier, e.g. ``"ai_base_rate"``.
    low : float
        Lower bound of the plausible range.
    high : float
        Upper bound of the plausible range.
    default : float
        Nominal / best-estimate value.
    description : str
        Human-readable description for reports and website.
    category : str
        One of ``"base_rate"``, ``"interaction_weight"``, ``"structural"``.
    """

    name: str
    low: float
    high: float
    default: float
    description: str
    category: str  # "base_rate", "interaction_weight", "structural"

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Parameter '{self.name}': low ({self.low}) must be < "
                f"high ({self.high})"
            )
        if not (self.low <= self.default <= self.high):
            raise ValueError(
                f"Parameter '{self.name}': default ({self.default}) must be "
                f"between low ({self.low}) and high ({self.high})"
            )
        valid_categories = {"base_rate", "interaction_weight", "structural"}
        if self.category not in valid_categories:
            raise ValueError(
                f"Parameter '{self.name}': category must be one of "
                f"{valid_categories}, got '{self.category}'"
            )


@dataclass
class SobolResult:
    """Results of a Sobol sensitivity analysis.

    Attributes
    ----------
    parameters : list[SensitivityParameter]
        The parameters that were varied.
    first_order : dict[str, dict[str, float]]
        ``{output_name: {param_name: S1_index}}``.
        First-order (main effect) Sobol index for each parameter on each
        output.  Represents the fraction of output variance explained by
        that parameter alone.
    total_order : dict[str, dict[str, float]]
        ``{output_name: {param_name: ST_index}}``.
        Total-order Sobol index: fraction of output variance that would
        remain if all parameters *except* this one were fixed.  Always
        ``>= S1``.  The gap ``ST - S1`` quantifies interaction effects.
    first_order_ci : dict[str, dict[str, tuple[float, float]]]
        95% bootstrap confidence intervals for first-order indices.
    total_order_ci : dict[str, dict[str, tuple[float, float]]]
        95% bootstrap confidence intervals for total-order indices.
    n_samples : int
        Base number of samples (N).  Total model evaluations =
        ``N * (2 + D)`` where D is the number of parameters.
    total_evaluations : int
        Actual number of model evaluations performed.
    outputs_analyzed : list[str]
        Names of the output variables analyzed.
    tornado_data : dict[str, list[dict]]
        Pre-computed tornado diagram data for each output.
    convergence_status : dict[str, str]
        Per-output convergence assessment: "converged", "marginal", or
        "not_converged".
    """

    parameters: list[SensitivityParameter]
    first_order: dict[str, dict[str, float]]
    total_order: dict[str, dict[str, float]]
    first_order_ci: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )
    total_order_ci: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )
    n_samples: int = 0
    total_evaluations: int = 0
    outputs_analyzed: list[str] = field(default_factory=list)
    tornado_data: dict[str, list[dict]] = field(default_factory=dict)
    convergence_status: dict[str, str] = field(default_factory=dict)

    def top_parameters(
        self,
        output: str,
        order: str = "total",
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-n most influential parameters for a given output.

        Parameters
        ----------
        output : str
            Output variable name, e.g. ``"ai_2039_median"``.
        order : str
            ``"first"`` for first-order or ``"total"`` for total-order.
        n : int
            Number of top parameters to return.

        Returns
        -------
        list[tuple[str, float]]
            ``[(param_name, index_value), ...]`` sorted descending.
        """
        indices = (
            self.first_order if order == "first" else self.total_order
        )
        if output not in indices:
            raise KeyError(
                f"Output '{output}' not found. "
                f"Available: {list(indices.keys())}"
            )
        ranked = sorted(
            indices[output].items(), key=lambda kv: kv[1], reverse=True
        )
        return ranked[:n]

    def summary_table(self, output: str) -> str:
        """Return a human-readable table of Sobol indices for one output."""
        if output not in self.first_order:
            raise KeyError(f"Output '{output}' not found.")

        lines = [
            f"Sobol Sensitivity Analysis — {output}",
            f"{'='*60}",
            f"Base samples: {self.n_samples:,}  |  "
            f"Total evaluations: {self.total_evaluations:,}  |  "
            f"Convergence: {self.convergence_status.get(output, 'unknown')}",
            "",
            f"{'Parameter':<30} {'S1':>8} {'ST':>8} {'ST-S1':>8} "
            f"{'Category':<20}",
            f"{'-'*76}",
        ]

        # Sort by total-order descending
        params_sorted = sorted(
            self.parameters,
            key=lambda p: self.total_order[output].get(p.name, 0),
            reverse=True,
        )
        for p in params_sorted:
            s1 = self.first_order[output].get(p.name, 0.0)
            st = self.total_order[output].get(p.name, 0.0)
            interaction = st - s1
            lines.append(
                f"{p.name:<30} {s1:>8.4f} {st:>8.4f} {interaction:>8.4f} "
                f"{p.category:<20}"
            )

        lines.append("")
        lines.append(
            "S1 = first-order index (main effect only)"
        )
        lines.append(
            "ST = total-order index (main effect + all interactions)"
        )
        lines.append(
            "ST-S1 = contribution from parameter interactions"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 15 simulation domains (matching domain_registry.py)
# ---------------------------------------------------------------------------
SIMULATION_DOMAINS = [
    "ai", "compute", "energy", "batteries", "genomics", "drug",
    "robotics", "space", "manufacturing", "materials", "bci", "quantum",
    "environment", "vr", "sensors",
]


# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------

def define_parameters(
    base_rate_defaults: Optional[dict[str, float]] = None,
) -> list[SensitivityParameter]:
    """Define the ~30 parameters to be varied in sensitivity analysis.

    The parameters fall into three categories:

    1. **Base rates** (15 params) — annual improvement rate per simulation
       domain.  Varied from 0.5x to 2.0x of the fitted rate.  These capture
       uncertainty about whether historical trends continue, accelerate,
       or decelerate.

    2. **Key interaction weights** (10 params) — the top 10 most
       consequential cross-domain amplification weights.  Ranges are set
       based on the evidence bounds documented in ``interactions.json``.

    3. **Structural parameters** (5 params) — parameters that control the
       simulation engine's global behaviour: discount rate for future
       acceleration (gamma), breakthrough probability scaling, noise
       scaling, and adoption speed.

    Parameters
    ----------
    base_rate_defaults : dict[str, float], optional
        Override the default base rates per domain.  Keys are domain names
        (e.g. ``"ai"``), values are annual improvement rates.  If not
        provided, reasonable defaults from v5/v6 fitted rates are used.

    Returns
    -------
    list[SensitivityParameter]
        30 parameters ready for Sobol analysis.
    """
    params: list[SensitivityParameter] = []

    # ----- 1. Base rates (15 parameters) -----
    # Default annual improvement rates (log-linear slope in log-space,
    # representing annual fractional change).  These are approximate central
    # estimates from fitted data; actual values come from the fits module.
    _default_base_rates: dict[str, tuple[float, str]] = {
        "ai":            (0.40, "AI capability annual improvement rate (benchmarks, FLOP efficiency)"),
        "compute":       (0.35, "Compute cost-performance annual improvement rate ($/FLOP)"),
        "energy":        (0.15, "Solar/wind LCOE annual cost reduction rate"),
        "batteries":     (0.18, "Battery pack cost annual reduction rate ($/kWh)"),
        "genomics":      (0.45, "Genome sequencing/synthesis cost annual reduction rate"),
        "drug":          (0.05, "Drug discovery timeline annual improvement rate"),
        "robotics":      (0.12, "Robotics capability/cost annual improvement rate"),
        "space":         (0.20, "Launch cost annual reduction rate ($/kg to LEO)"),
        "manufacturing": (0.08, "Additive manufacturing cost annual reduction rate"),
        "materials":     (0.10, "Materials discovery rate annual improvement"),
        "bci":           (0.25, "BCI channel count / accuracy annual improvement rate"),
        "quantum":       (0.30, "Quantum qubit count / error rate annual improvement"),
        "environment":   (0.12, "Carbon capture / desalination cost annual reduction rate"),
        "vr":            (0.20, "VR resolution/pixel density annual improvement rate"),
        "sensors":       (0.18, "Sensor cost-performance annual improvement rate"),
    }

    for domain in SIMULATION_DOMAINS:
        if base_rate_defaults and domain in base_rate_defaults:
            default_rate = base_rate_defaults[domain]
            desc = f"{domain} base improvement rate (custom)"
        else:
            default_rate, desc = _default_base_rates[domain]

        # Range: 0.5x to 2.0x of the default rate.
        # This captures uncertainty about whether the historical trend
        # continues, decelerates (pessimistic), or accelerates (optimistic).
        params.append(SensitivityParameter(
            name=f"{domain}_base_rate",
            low=round(default_rate * 0.5, 4),
            high=round(default_rate * 2.0, 4),
            default=round(default_rate, 4),
            description=desc,
            category="base_rate",
        ))

    # ----- 2. Key interaction weights (10 parameters) -----
    # Selected as the 10 interactions with the highest total-order impact
    # expectation, based on domain connectivity and weight magnitude.
    # Ranges come from evidence bounds in interactions.json.
    _interaction_params = [
        # (name, low, high, default, description)
        (
            "ai_ai_weight", 1.0, 2.5, 1.8,
            "AI recursive self-improvement weight — the most scrutinised "
            "parameter in the model (v5: 2.5, v6: 1.8)"
        ),
        (
            "compute_ai_weight", 1.5, 3.5, 2.5,
            "Compute -> AI weight: scaling laws show compute is the "
            "primary driver of AI capability"
        ),
        (
            "ai_drug_weight", 1.0, 3.0, 2.2,
            "AI -> Drug discovery weight: AI co-scientist and INS018_055 "
            "evidence (reduced from v5 3.0)"
        ),
        (
            "ai_genomics_weight", 1.5, 3.5, 2.5,
            "AI -> Genomics weight: AlphaFold and popEVE as primary evidence"
        ),
        (
            "ai_materials_weight", 1.5, 3.5, 2.5,
            "AI -> Materials weight: GNoME 10x known materials, A-Lab "
            "autonomous synthesis"
        ),
        (
            "ai_robotics_weight", 1.0, 3.0, 2.0,
            "AI -> Robotics weight: RT-2, Figure 02, but limited deployment"
        ),
        (
            "materials_batteries_weight", 1.5, 3.5, 2.5,
            "Materials -> Batteries weight: LFP revolution, sodium-ion "
            "emergence"
        ),
        (
            "energy_environment_weight", 1.5, 3.5, 2.5,
            "Energy -> Environment weight: energy is 50-70% of desalination "
            "and carbon capture cost"
        ),
        (
            "genomics_drug_weight", 1.0, 3.0, 2.0,
            "Genomics -> Drug weight: GWAS targets have 2x higher clinical "
            "trial success rate"
        ),
        (
            "robotics_manufacturing_weight", 1.0, 3.0, 2.0,
            "Robotics -> Manufacturing weight: Foxconn 100K robots, Tesla "
            "Gigafactory automation"
        ),
    ]

    for name, low, high, default, desc in _interaction_params:
        params.append(SensitivityParameter(
            name=name,
            low=low,
            high=high,
            default=default,
            description=desc,
            category="interaction_weight",
        ))

    # ----- 3. Structural parameters (5 parameters) -----
    _structural_params = [
        (
            "base_gamma", 0.03, 0.12, 0.06,
            "Base discount rate for future acceleration.  Higher gamma means "
            "the model trusts near-term trends more and discounts far-future "
            "acceleration more heavily."
        ),
        (
            "gamma_decay", 0.88, 0.98, 0.95,
            "Annual decay factor for gamma.  At 0.95, gamma decreases by 5% "
            "per year, reflecting increasing confidence in sustained "
            "acceleration as evidence accumulates."
        ),
        (
            "breakthrough_scale", 0.5, 2.0, 1.0,
            "Multiplier on all breakthrough event probabilities.  At 2.0, "
            "breakthroughs are twice as likely as the base case in every "
            "domain."
        ),
        (
            "noise_scale", 0.5, 2.0, 1.0,
            "Multiplier on Monte Carlo noise magnitude.  At 2.0, annual "
            "noise is twice as large, producing wider confidence intervals."
        ),
        (
            "adoption_speed", 0.5, 2.0, 1.0,
            "Multiplier on the Bass diffusion imitation parameter (q).  "
            "At 2.0, technology adoption curves are twice as fast, reflecting "
            "a world where new technologies diffuse faster than historical "
            "norms."
        ),
    ]

    for name, low, high, default, desc in _structural_params:
        params.append(SensitivityParameter(
            name=name,
            low=low,
            high=high,
            default=default,
            description=desc,
            category="structural",
        ))

    return params


# ---------------------------------------------------------------------------
# Sobol quasi-random sampling and matrix construction (Saltelli method)
# ---------------------------------------------------------------------------

def _generate_saltelli_samples(
    n_samples: int,
    n_params: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent Sobol quasi-random matrices in [0,1]^D.

    We use scipy's ``Sobol`` engine to generate a ``(2*N, D)`` matrix
    and split it into two halves.  Using a single engine ensures the two
    halves are from complementary Sobol subsequences (better than two
    independent engines).

    Parameters
    ----------
    n_samples : int
        Base number of samples (N).  Must be a power of 2.
    n_params : int
        Dimensionality (D).
    seed : int
        Seed for scrambling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Matrices A and B, each of shape ``(n_samples, n_params)``.
    """
    if n_samples & (n_samples - 1) != 0 or n_samples < 2:
        raise ValueError(
            f"n_samples must be a power of 2 (got {n_samples}). "
            f"Try {2 ** int(np.ceil(np.log2(max(n_samples, 2))))}."
        )

    engine = SobolEngine(d=n_params, scramble=True, seed=seed)
    # Generate 2*N points and split
    combined = engine.random(2 * n_samples)  # shape (2*N, D)
    A = combined[:n_samples]
    B = combined[n_samples:]
    return A, B


def _scale_samples(
    unit_samples: np.ndarray,
    parameters: list[SensitivityParameter],
) -> np.ndarray:
    """Scale samples from [0,1] to parameter ranges.

    Uses a simple linear mapping: ``value = low + unit * (high - low)``.

    Parameters
    ----------
    unit_samples : np.ndarray
        Samples in [0, 1], shape ``(N, D)``.
    parameters : list[SensitivityParameter]
        Parameter definitions with ``low`` and ``high`` bounds.

    Returns
    -------
    np.ndarray
        Scaled samples, shape ``(N, D)``.
    """
    lows = np.array([p.low for p in parameters])
    highs = np.array([p.high for p in parameters])
    return lows + unit_samples * (highs - lows)


def _samples_to_param_dicts(
    scaled: np.ndarray,
    parameters: list[SensitivityParameter],
) -> list[dict[str, float]]:
    """Convert a matrix of scaled samples into a list of parameter dicts.

    Parameters
    ----------
    scaled : np.ndarray
        Shape ``(N, D)`` of scaled parameter values.
    parameters : list[SensitivityParameter]
        Parameter definitions (used for names).

    Returns
    -------
    list[dict[str, float]]
        One dict per row, mapping parameter name -> value.
    """
    names = [p.name for p in parameters]
    return [
        {name: float(row[j]) for j, name in enumerate(names)}
        for row in scaled
    ]


# ---------------------------------------------------------------------------
# Sobol index computation (Saltelli estimator)
# ---------------------------------------------------------------------------

def _compute_sobol_indices(
    y_A: np.ndarray,
    y_B: np.ndarray,
    y_AB: np.ndarray,
    n_samples: int,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first-order (S1) and total-order (ST) Sobol indices.

    Uses the Saltelli (2010) estimators which are numerically stable
    and have good convergence properties.

    Parameters
    ----------
    y_A : np.ndarray
        Model outputs at matrix A samples, shape ``(N,)``.
    y_B : np.ndarray
        Model outputs at matrix B samples, shape ``(N,)``.
    y_AB : np.ndarray
        Model outputs at AB_i matrices, shape ``(D, N)`` where
        ``y_AB[i, :]`` corresponds to matrix AB_i.
    n_samples : int
        N — base sample count.
    n_params : int
        D — number of parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(S1, ST)`` arrays of shape ``(D,)``.
    """
    # Total variance estimated from combined A and B samples
    all_y = np.concatenate([y_A, y_B])
    var_y = np.var(all_y, ddof=1)

    if var_y < 1e-30:
        # Output is essentially constant — all indices are zero
        return np.zeros(n_params), np.zeros(n_params)

    S1 = np.zeros(n_params)
    ST = np.zeros(n_params)

    for i in range(n_params):
        # First-order: Saltelli (2010) Eq. (b)
        # S_i = (1/N) * sum(f(B) * (f(AB_i) - f(A))) / Var(Y)
        S1[i] = np.mean(y_B * (y_AB[i] - y_A)) / var_y

        # Total-order: Jansen (1999) estimator — more stable than
        # the Saltelli estimator for ST
        # ST_i = (1/(2N)) * sum((f(A) - f(AB_i))^2) / Var(Y)
        ST[i] = 0.5 * np.mean((y_A - y_AB[i]) ** 2) / var_y

    return S1, ST


def _bootstrap_sobol_ci(
    y_A: np.ndarray,
    y_B: np.ndarray,
    y_AB: np.ndarray,
    n_samples: int,
    n_params: int,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for Sobol indices.

    Resamples with replacement and recomputes indices to estimate
    the sampling distribution.

    Returns
    -------
    tuple of 4 arrays, each shape ``(D,)``
        ``(S1_low, S1_high, ST_low, ST_high)``
    """
    if rng is None:
        rng = np.random.default_rng(seed=123)

    alpha = (1 - ci_level) / 2
    S1_boot = np.zeros((n_bootstrap, n_params))
    ST_boot = np.zeros((n_bootstrap, n_params))

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        s1_b, st_b = _compute_sobol_indices(
            y_A[idx], y_B[idx], y_AB[:, idx], n_samples, n_params
        )
        S1_boot[b] = s1_b
        ST_boot[b] = st_b

    S1_low = np.quantile(S1_boot, alpha, axis=0)
    S1_high = np.quantile(S1_boot, 1 - alpha, axis=0)
    ST_low = np.quantile(ST_boot, alpha, axis=0)
    ST_high = np.quantile(ST_boot, 1 - alpha, axis=0)

    return S1_low, S1_high, ST_low, ST_high


def _assess_convergence(
    S1: np.ndarray,
    ST: np.ndarray,
    S1_low: np.ndarray,
    S1_high: np.ndarray,
    ST_low: np.ndarray,
    ST_high: np.ndarray,
) -> str:
    """Assess whether the Sobol analysis has converged.

    Heuristics:
    - "converged" if all total-order CIs have width < 0.10 and
      sum of first-order indices <= 1.1 (allowing small numerical error).
    - "marginal" if CIs are < 0.20 and sum <= 1.3.
    - "not_converged" otherwise.
    """
    max_st_width = np.max(ST_high - ST_low)
    s1_sum = np.sum(np.clip(S1, 0, None))  # clip negatives

    if max_st_width < 0.10 and s1_sum <= 1.1:
        return "converged"
    elif max_st_width < 0.20 and s1_sum <= 1.3:
        return "marginal"
    else:
        return "not_converged"


# ---------------------------------------------------------------------------
# Main Sobol analysis function
# ---------------------------------------------------------------------------

def run_sobol_analysis(
    simulate_fn: Callable[[dict[str, float]], dict[str, float]],
    parameters: Optional[list[SensitivityParameter]] = None,
    n_samples: int = 1024,
    outputs_of_interest: Optional[list[str]] = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SobolResult:
    """Run Sobol sensitivity analysis on a simulation function.

    This is the primary entry point for global sensitivity analysis.
    It generates quasi-random parameter combinations, runs the simulation
    for each, and computes first-order and total-order Sobol indices with
    bootstrap confidence intervals.

    Parameters
    ----------
    simulate_fn : callable
        A function ``f(params_dict) -> outputs_dict`` where:
        - ``params_dict`` maps parameter names to float values.
        - ``outputs_dict`` maps output names to float values.
        Example: ``{'ai_base_rate': 0.35, ...} -> {'ai_2039_median': 1e6}``.

        **Design note**: This interface decouples sensitivity analysis from
        the simulation implementation.  The simulation can use reduced Monte
        Carlo runs (100-500 per call) since we need ``N * (2 + D)``
        calls — with N=1024 and D=30, that's 32,768 simulations.  At 10K
        MC runs each, that would be 327M runs — infeasible.  At 200 MC
        runs each, it's ~6.5M — tractable.

    parameters : list[SensitivityParameter], optional
        Parameters to vary.  If ``None``, uses ``define_parameters()``.

    n_samples : int
        Base number of Sobol samples (N).  Must be a power of 2.
        Total model evaluations = ``N * (2 + D)``.
        Recommended: 1024 for exploratory, 4096+ for publication.

    outputs_of_interest : list[str], optional
        Which output keys to analyze.  If ``None``, analyzes all outputs
        returned by the first simulation call.

    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.

    seed : int
        Random seed for reproducibility.

    progress_callback : callable, optional
        ``callback(completed, total)`` called after each batch of
        evaluations, for progress tracking.

    Returns
    -------
    SobolResult
        Complete sensitivity analysis results.

    Raises
    ------
    ValueError
        If ``n_samples`` is not a power of 2.
    RuntimeError
        If the simulation function returns inconsistent output keys.
    """
    if parameters is None:
        parameters = define_parameters()

    D = len(parameters)
    N = n_samples
    total_evals = N * (2 + D)

    # --- Step 1: Generate Sobol samples ---
    A_unit, B_unit = _generate_saltelli_samples(N, D, seed=seed)

    # Scale to parameter ranges
    A_scaled = _scale_samples(A_unit, parameters)
    B_scaled = _scale_samples(B_unit, parameters)

    # Construct AB_i matrices (D matrices, each N x D)
    # AB_i is A with column i replaced by B's column i
    AB_unit = np.empty((D, N, D))
    for i in range(D):
        AB_unit[i] = A_unit.copy()
        AB_unit[i, :, i] = B_unit[:, i]

    AB_scaled = np.empty((D, N, D))
    for i in range(D):
        AB_scaled[i] = _scale_samples(AB_unit[i], parameters)

    # --- Step 2: Evaluate the model ---
    # Convert to parameter dicts
    A_dicts = _samples_to_param_dicts(A_scaled, parameters)
    B_dicts = _samples_to_param_dicts(B_scaled, parameters)
    AB_dicts = [
        _samples_to_param_dicts(AB_scaled[i], parameters) for i in range(D)
    ]

    # Run simulations — A samples
    completed = 0
    y_A_raw: list[dict[str, float]] = []
    for pd in A_dicts:
        y_A_raw.append(simulate_fn(pd))
        completed += 1
        if progress_callback:
            progress_callback(completed, total_evals)

    # Determine outputs of interest from first result
    all_output_keys = sorted(y_A_raw[0].keys())
    if outputs_of_interest is None:
        outputs_of_interest = all_output_keys
    else:
        # Validate requested outputs exist
        missing = set(outputs_of_interest) - set(all_output_keys)
        if missing:
            raise RuntimeError(
                f"Requested outputs not found in simulation: {missing}. "
                f"Available: {all_output_keys}"
            )

    # Run B samples
    y_B_raw: list[dict[str, float]] = []
    for pd in B_dicts:
        y_B_raw.append(simulate_fn(pd))
        completed += 1
        if progress_callback:
            progress_callback(completed, total_evals)

    # Run AB_i samples
    y_AB_raw: list[list[dict[str, float]]] = []
    for i in range(D):
        y_AB_i: list[dict[str, float]] = []
        for pd in AB_dicts[i]:
            y_AB_i.append(simulate_fn(pd))
            completed += 1
            if progress_callback:
                progress_callback(completed, total_evals)
        y_AB_raw.append(y_AB_i)

    # --- Step 3: Compute indices for each output ---
    first_order: dict[str, dict[str, float]] = {}
    total_order: dict[str, dict[str, float]] = {}
    first_order_ci: dict[str, dict[str, tuple[float, float]]] = {}
    total_order_ci: dict[str, dict[str, tuple[float, float]]] = {}
    convergence_status: dict[str, str] = {}

    param_names = [p.name for p in parameters]

    for out_key in outputs_of_interest:
        # Extract output arrays
        y_A = np.array([r[out_key] for r in y_A_raw])
        y_B = np.array([r[out_key] for r in y_B_raw])
        y_AB = np.array([
            [r[out_key] for r in y_AB_raw[i]]
            for i in range(D)
        ])  # shape (D, N)

        # Handle NaN/inf: replace with output mean
        all_y_finite = np.concatenate([y_A, y_B, y_AB.ravel()])
        finite_mask = np.isfinite(all_y_finite)
        if not finite_mask.all():
            n_bad = int((~finite_mask).sum())
            warnings.warn(
                f"Output '{out_key}': {n_bad} non-finite values replaced "
                f"with output mean.",
                stacklevel=2,
            )
            fill_val = np.nanmean(all_y_finite[finite_mask]) if finite_mask.any() else 0.0
            y_A = np.where(np.isfinite(y_A), y_A, fill_val)
            y_B = np.where(np.isfinite(y_B), y_B, fill_val)
            y_AB = np.where(np.isfinite(y_AB), y_AB, fill_val)

        # Compute indices
        S1, ST = _compute_sobol_indices(y_A, y_B, y_AB, N, D)

        # Bootstrap CIs
        rng = np.random.default_rng(seed=seed + hash(out_key) % 2**31)
        S1_lo, S1_hi, ST_lo, ST_hi = _bootstrap_sobol_ci(
            y_A, y_B, y_AB, N, D,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )

        # Store results
        first_order[out_key] = {
            name: float(S1[j]) for j, name in enumerate(param_names)
        }
        total_order[out_key] = {
            name: float(ST[j]) for j, name in enumerate(param_names)
        }
        first_order_ci[out_key] = {
            name: (float(S1_lo[j]), float(S1_hi[j]))
            for j, name in enumerate(param_names)
        }
        total_order_ci[out_key] = {
            name: (float(ST_lo[j]), float(ST_hi[j]))
            for j, name in enumerate(param_names)
        }
        convergence_status[out_key] = _assess_convergence(
            S1, ST, S1_lo, S1_hi, ST_lo, ST_hi
        )

    return SobolResult(
        parameters=parameters,
        first_order=first_order,
        total_order=total_order,
        first_order_ci=first_order_ci,
        total_order_ci=total_order_ci,
        n_samples=N,
        total_evaluations=total_evals,
        outputs_analyzed=outputs_of_interest,
        tornado_data={},  # filled separately via compute_tornado_data
        convergence_status=convergence_status,
    )


# ---------------------------------------------------------------------------
# Tornado diagram data
# ---------------------------------------------------------------------------

def compute_tornado_data(
    simulate_fn: Callable[[dict[str, float]], dict[str, float]],
    parameters: Optional[list[SensitivityParameter]] = None,
    output_key: str = "ai_2039_median",
    n_points: int = 20,
) -> list[dict]:
    """Generate tornado diagram data for one output variable.

    For each parameter, holds all others at their default value and sweeps
    this one from its low to high bound.  Records the output at each point.
    Returns results sorted by impact (descending).

    Tornado diagrams are the most intuitive visualisation of sensitivity:
    they show at a glance which parameter changes have the largest effect
    on the output.

    Parameters
    ----------
    simulate_fn : callable
        Function ``f(params_dict) -> outputs_dict``.
    parameters : list[SensitivityParameter], optional
        If ``None``, uses ``define_parameters()``.
    output_key : str
        Which output to track, e.g. ``"ai_2039_median"``.
    n_points : int
        Number of sweep points per parameter (default 20).

    Returns
    -------
    list[dict]
        Sorted by impact (descending).  Each dict contains::

            {
                "parameter": str,
                "description": str,
                "category": str,
                "low_value": float,
                "high_value": float,
                "default_value": float,
                "output_at_low": float,
                "output_at_high": float,
                "output_at_default": float,
                "impact": float,          # |output_at_high - output_at_low|
                "relative_impact": float,  # impact / output_at_default
                "sweep": list[dict],       # [{"param_value": ..., "output": ...}]
            }
    """
    if parameters is None:
        parameters = define_parameters()

    # Build default parameter dict
    defaults = {p.name: p.default for p in parameters}

    # Get output at defaults
    output_at_default = simulate_fn(defaults)[output_key]

    results: list[dict] = []

    for param in parameters:
        sweep_values = np.linspace(param.low, param.high, n_points)
        sweep_outputs: list[float] = []

        for val in sweep_values:
            params_copy = defaults.copy()
            params_copy[param.name] = float(val)
            out = simulate_fn(params_copy)
            sweep_outputs.append(out[output_key])

        sweep_arr = np.array(sweep_outputs)
        output_at_low = float(sweep_arr[0])
        output_at_high = float(sweep_arr[-1])
        impact = abs(output_at_high - output_at_low)

        # Relative impact — guarded against division by zero
        if abs(output_at_default) > 1e-30:
            relative_impact = impact / abs(output_at_default)
        else:
            relative_impact = float("inf") if impact > 0 else 0.0

        results.append({
            "parameter": param.name,
            "description": param.description,
            "category": param.category,
            "low_value": param.low,
            "high_value": param.high,
            "default_value": param.default,
            "output_at_low": output_at_low,
            "output_at_high": output_at_high,
            "output_at_default": output_at_default,
            "impact": impact,
            "relative_impact": relative_impact,
            "sweep": [
                {"param_value": float(v), "output": float(o)}
                for v, o in zip(sweep_values, sweep_outputs)
            ],
        })

    # Sort by impact, descending
    results.sort(key=lambda r: r["impact"], reverse=True)

    return results


# ---------------------------------------------------------------------------
# Utility: attach tornado data to a SobolResult
# ---------------------------------------------------------------------------

def add_tornado_to_result(
    result: SobolResult,
    simulate_fn: Callable[[dict[str, float]], dict[str, float]],
    outputs: Optional[list[str]] = None,
    n_points: int = 20,
) -> SobolResult:
    """Convenience function: compute tornado data for one or more outputs
    and attach it to an existing SobolResult.

    Parameters
    ----------
    result : SobolResult
        Existing Sobol result to augment.
    simulate_fn : callable
        Same simulation function used for the Sobol analysis.
    outputs : list[str], optional
        Outputs to generate tornado data for.  Defaults to all outputs
        in the Sobol result.
    n_points : int
        Number of sweep points per parameter.

    Returns
    -------
    SobolResult
        The same object, with ``tornado_data`` populated.
    """
    if outputs is None:
        outputs = result.outputs_analyzed

    for out_key in outputs:
        result.tornado_data[out_key] = compute_tornado_data(
            simulate_fn=simulate_fn,
            parameters=result.parameters,
            output_key=out_key,
            n_points=n_points,
        )

    return result
