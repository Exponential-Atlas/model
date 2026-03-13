"""
Exponential Atlas v6 -- Adoption Delay (Bass Diffusion with Trend-Accelerated Deployment)
=========================================================================================

Deployment speed itself follows an exponential trend. The time from
"technology available" to "mass deployment" has been shrinking for 150 years:

    Electricity (1882):  46 years to 25% adoption
    Telephone (1876):    35 years
    Television (1926):   26 years
    PC (1975):           16 years
    Internet (1991):      7 years
    Smartphone (2007):    5 years
    ChatGPT (2022):       0.17 years (2 months to 100M users)

Fitting log(years_to_25%) vs year gives:
    log(T) = -0.0214 * year + 44.28    (R² = 0.73)
    Deployment time halves every ~32 years.

This module:
1. Fits the deployment speed trend from historical data
2. Uses it to compute time-varying Bass diffusion parameters
3. Applies domain-specific friction multipliers (digital deploys faster
   than physical infrastructure, which deploys faster than regulated tech)
4. Each year in the simulation, the Bass q parameter is set by the trend,
   NOT by arbitrary estimates

Sources:
- Our World in Data, "Technology Adoption" (Comin & Hobijn 2004 dataset)
- Harvard Business Review, "The Pace of Technology Adoption is Speeding Up"
- Similarweb ChatGPT traffic data (2023)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import linregress

from model.data.domain_registry import SIMULATION_DOMAINS


# ===========================================================================
# DEPLOYMENT SPEED TREND — FITTED FROM HISTORICAL DATA
# ===========================================================================
# Each entry: (year_introduced, years_to_25%_adoption)
# Sources: Our World in Data "Technology Adoption"; Comin & Hobijn (2004);
# HBR "The Pace of Technology Adoption"; Similarweb; company filings

DEPLOYMENT_SPEED_DATA = [
    (1882, 46,   "Electricity",       "Our World in Data / Comin & Hobijn 2004"),
    (1876, 35,   "Telephone",         "Our World in Data / Comin & Hobijn 2004"),
    (1926, 26,   "Television",        "Our World in Data / Comin & Hobijn 2004"),
    (1947, 18,   "Credit card",       "Our World in Data / Comin & Hobijn 2004"),
    (1975, 16,   "Personal computer", "Our World in Data / Comin & Hobijn 2004"),
    (1983, 13,   "Mobile phone",      "Our World in Data / Comin & Hobijn 2004"),
    (1991,  7,   "World wide web",    "Our World in Data / Comin & Hobijn 2004"),
    (2004,  5,   "Social media",      "Pew Research Internet Adoption Surveys"),
    (2007,  5,   "Smartphone",        "Pew Research; Statista smartphone penetration"),
    (2009,  4,   "Tablet",            "IDC tablet tracker; Statista"),
    (2010,  3,   "Electric vehicles", "IEA Global EV Outlook 2024"),
    (2022,  0.17, "ChatGPT",          "Similarweb traffic data Jan 2023"),
]


def fit_deployment_trend() -> tuple[float, float, float]:
    """Fit exponential trend to historical deployment speed data.

    Returns
    -------
    slope : float
        Log-linear slope (negative — deployment is getting faster).
    intercept : float
        Log-linear intercept.
    r_squared : float
        R² of the fit.
    """
    years = np.array([d[0] for d in DEPLOYMENT_SPEED_DATA], dtype=float)
    times = np.array([d[1] for d in DEPLOYMENT_SPEED_DATA], dtype=float)
    slope, intercept, r_value, _, _ = linregress(years, np.log(times))
    return slope, intercept, r_value ** 2


# Pre-compute the trend at import time
_TREND_SLOPE, _TREND_INTERCEPT, _TREND_R2 = fit_deployment_trend()


def predicted_years_to_25pct(year: float) -> float:
    """Predict years to 25% adoption for a technology introduced in `year`.

    Based on the fitted exponential trend of deployment speed.

    Parameters
    ----------
    year : float
        Calendar year the technology becomes available.

    Returns
    -------
    float
        Predicted years to reach 25% market adoption.
        Minimum floor of 0.05 years (~18 days) for numerical stability.
    """
    t = math.exp(_TREND_INTERCEPT + _TREND_SLOPE * year)
    return max(t, 0.05)  # Floor: nothing deploys in less than ~18 days


# ===========================================================================
# DOMAIN FRICTION MULTIPLIERS
# ===========================================================================
# The deployment trend is a cross-technology average. Individual domains
# deploy faster or slower based on their nature. These multipliers scale
# the predicted time-to-adoption.
#
# < 1.0 = deploys FASTER than average (digital, no regulation)
# = 1.0 = deploys at average speed
# > 1.0 = deploys SLOWER than average (physical, regulated, capital-intensive)
#
# Each multiplier is justified by the historical analog cited.

DOMAIN_FRICTION: dict[str, dict] = {
    # Digital — faster than average
    "ai": {
        "multiplier": 0.3,
        "justification": "Pure software; OTA deployment. ChatGPT: 2 months to 100M.",
        "analog": "ChatGPT, cloud SaaS",
        "source": "Similarweb 2023",
    },
    "compute": {
        "multiplier": 0.5,
        "justification": "Hardware fab cycles + software layer. Cloud compute: ~2yr ramp.",
        "analog": "Cloud computing (AWS ramp 2006-2010)",
        "source": "Gartner cloud adoption reports",
    },
    "vr": {
        "multiplier": 0.6,
        "justification": "Consumer hardware + content ecosystem.",
        "analog": "Meta Quest adoption curve",
        "source": "IDC VR headset tracker 2024",
    },
    "sensors": {
        "multiplier": 0.6,
        "justification": "Mass-manufactured consumer/industrial electronics.",
        "analog": "MEMS accelerometer adoption (smartphones)",
        "source": "Yole Developpement MEMS reports",
    },

    # Average — infrastructure
    "robotics": {
        "multiplier": 0.8,
        "justification": "Manufacturing ramp + integration. Industrial robots: ~5yr to scale.",
        "analog": "Industrial robot deployment (IFR data)",
        "source": "IFR World Robotics 2024",
    },
    "manufacturing": {
        "multiplier": 1.0,
        "justification": "Factory retooling cycles. 3D printing: ~7yr to meaningful share.",
        "analog": "Additive manufacturing adoption",
        "source": "Wohlers Report 2024",
    },
    "materials": {
        "multiplier": 1.2,
        "justification": "Lab → pilot → scale pipeline. New materials: typically 10-15yr.",
        "analog": "LFP battery chemistry adoption",
        "source": "McKinsey Advanced Materials 2023",
    },

    # Slow — physical infrastructure
    "energy": {
        "multiplier": 1.5,
        "justification": "Grid infrastructure, permitting, construction. Solar: ~12yr to 5% of grid.",
        "analog": "Solar PV grid penetration",
        "source": "IRENA Renewable Capacity Statistics 2024",
    },
    "batteries": {
        "multiplier": 1.3,
        "justification": "Gigafactory buildout + supply chain. Battery packs: ~8yr cost learning.",
        "analog": "EV battery deployment curve",
        "source": "BloombergNEF EVO 2024",
    },
    "environment": {
        "multiplier": 1.5,
        "justification": "Large infrastructure (desal plants, DAC facilities). 5-10yr build cycles.",
        "analog": "Desalination plant deployment (MENA)",
        "source": "IDA Desalination Yearbook 2024",
    },
    "space": {
        "multiplier": 2.5,
        "justification": "Extreme capital, safety certification, launch infrastructure.",
        "analog": "Commercial launch cadence (SpaceX ramp)",
        "source": "FAA AST Annual Reports",
    },

    # Very slow — regulated
    "genomics": {
        "multiplier": 1.8,
        "justification": "Lab automation fast, but clinical/regulatory validation slow.",
        "analog": "Genetic testing adoption (23andMe)",
        "source": "23andMe SEC filings",
    },
    "drug": {
        "multiplier": 2.5,
        "justification": "FDA/EMA approval cycles dominate. Average drug: 3-7yr to peak adoption.",
        "analog": "New drug class adoption (biologics/mRNA)",
        "source": "IQVIA drug launch analytics 2024",
    },
    "bci": {
        "multiplier": 3.0,
        "justification": "Surgical procedures + extreme regulatory burden. Cochlear: 20yr to scale.",
        "analog": "Cochlear implant adoption",
        "source": "NIH NIDCD statistics",
    },
    "quantum": {
        "multiplier": 3.0,
        "justification": "Fundamental engineering + cryogenic infrastructure. Still pre-commercial.",
        "analog": "Quantum computing cloud access",
        "source": "IBM Quantum Network reports",
    },
}


def _years_to_bass_q(years_to_25pct: float, p: float = 0.03) -> float:
    """Convert years-to-25%-adoption into a Bass q parameter.

    Given that we want F(years_to_25pct) = 0.25, solve for q.
    Uses numerical approximation since the Bass CDF is transcendental.

    Parameters
    ----------
    years_to_25pct : float
        Target years to reach 25% adoption.
    p : float
        Innovation coefficient (held constant; only q varies).

    Returns
    -------
    float
        Bass q parameter that produces 25% adoption at the target time.
    """
    # Binary search for q that gives F(t) ≈ 0.25
    lo, hi = 0.01, 5.0
    target = 0.25
    for _ in range(50):  # Converges well within 50 iterations
        mid = (lo + hi) / 2
        f = bass_diffusion_weight(years_to_25pct, p, mid)
        if f < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ===========================================================================
# Domain-specific Bass p (innovation coefficient)
# ===========================================================================
# p controls early adoption (before network effects kick in).
# Digital products have higher p (easy to try); physical/regulated lower.

DOMAIN_P: dict[str, float] = {
    "ai": 0.05,
    "compute": 0.04,
    "vr": 0.04,
    "sensors": 0.04,
    "robotics": 0.03,
    "manufacturing": 0.02,
    "materials": 0.02,
    "energy": 0.02,
    "batteries": 0.02,
    "environment": 0.02,
    "genomics": 0.01,
    "drug": 0.01,
    "space": 0.01,
    "bci": 0.01,
    "quantum": 0.01,
}


def get_bass_params_for_year(
    domain: str,
    calendar_year: float,
) -> tuple[float, float]:
    """Get Bass (p, q) parameters for a domain at a given calendar year.

    The q parameter is derived from:
    1. The deployment speed trend (how fast does a new tech reach 25%?)
    2. The domain friction multiplier (this domain vs average)
    3. Conversion from years-to-25% into a Bass q

    This means q automatically increases each year — deployment gets faster
    because the TREND says it does, not because of arbitrary acceleration.

    Parameters
    ----------
    domain : str
        Simulation domain name.
    calendar_year : float
        The calendar year (e.g., 2030).

    Returns
    -------
    (p, q) : tuple[float, float]
        Bass diffusion parameters for this domain at this year.
    """
    p = DOMAIN_P.get(domain, 0.02)

    # Get the trend-predicted years to 25% for a generic tech at this year
    base_years = predicted_years_to_25pct(calendar_year)

    # Apply domain-specific friction
    friction = DOMAIN_FRICTION.get(domain, {"multiplier": 1.0})["multiplier"]
    domain_years = base_years * friction

    # Floor: even the slowest domain eventually deploys
    domain_years = max(domain_years, 0.1)

    # Convert to Bass q
    q = _years_to_bass_q(domain_years, p)

    return p, q


def bass_diffusion_weight(
    t_since_available: float,
    p: float = 0.03,
    q: float = 0.38,
) -> float:
    """Compute the Bass diffusion adoption fraction at time t.

    Parameters
    ----------
    t_since_available : float
        Time since the technology capability became available (years).
    p : float
        Innovation coefficient (external influence).
    q : float
        Imitation coefficient (internal influence / word of mouth).

    Returns
    -------
    float
        Fraction adopted, in [0, 1].
    """
    if t_since_available <= 0:
        return 0.0

    exponent = -(p + q) * t_since_available
    if exponent < -50:
        return 1.0

    exp_term = math.exp(exponent)

    if p <= 0:
        return 1.0 - exp_term

    numerator = 1.0 - exp_term
    denominator = 1.0 + (q / p) * exp_term

    if denominator <= 0:
        return 1.0

    return max(0.0, min(1.0, numerator / denominator))


def apply_adoption_delay(
    capability_history: np.ndarray,
    adoption_params: dict | None = None,
    sim_domains: list[str] | None = None,
    start_year: int = 2026,
) -> np.ndarray:
    """Apply trend-accelerated Bass diffusion adoption delay.

    The Bass q parameter is recomputed each simulation year using the
    deployment speed trend. This means capability gains in 2035 deploy
    faster than gains in 2026 — because the data shows deployment itself
    is on an exponential.

    Parameters
    ----------
    capability_history : np.ndarray
        Shape (n_steps, n_domains). Raw improvement factors.
    adoption_params : dict, optional
        Override params. If None, uses trend-derived params.
    sim_domains : list[str], optional
        Domain names. Defaults to SIMULATION_DOMAINS.
    start_year : int
        Calendar year of simulation start (default 2026).

    Returns
    -------
    np.ndarray
        Shape (n_steps, n_domains). Deployed improvement factors.
    """
    domains = sim_domains or list(SIMULATION_DOMAINS)
    n_steps, n_domains = capability_history.shape

    deployed = np.ones_like(capability_history)

    for d in range(n_domains):
        domain = domains[d] if d < len(domains) else "default"

        for t in range(1, n_steps):
            calendar_year = start_year + t

            # Get trend-derived Bass parameters for this year
            p_val, q_val = get_bass_params_for_year(domain, calendar_year)

            # Accumulate deployed improvement from all past capability gains
            deployed_value = 1.0

            for s in range(1, t + 1):
                cap_increment = capability_history[s, d] / capability_history[s - 1, d]
                if cap_increment <= 1.0:
                    continue

                time_lag = t - s
                adoption_fraction = bass_diffusion_weight(
                    float(time_lag), p_val, q_val
                )

                log_increment = math.log(cap_increment)
                deployed_increment = math.exp(log_increment * adoption_fraction)
                deployed_value *= deployed_increment

            deployed[t, d] = deployed_value

    return deployed


def apply_fixed_lag(
    capability_history: np.ndarray,
    lag_years: int = 3,
) -> np.ndarray:
    """Apply a fixed integer lag (v5 compatibility mode).

    Parameters
    ----------
    capability_history : np.ndarray
        Shape (n_steps, n_domains).
    lag_years : int
        Number of years to delay deployment.

    Returns
    -------
    np.ndarray
        Shape (n_steps, n_domains) with fixed lag applied.
    """
    n_steps, n_domains = capability_history.shape
    deployed = np.ones_like(capability_history)

    if lag_years < n_steps:
        deployed[lag_years:, :] = capability_history[:n_steps - lag_years, :]

    return deployed


def deployment_trend_summary() -> dict:
    """Return summary of the deployment speed trend for model card / website.

    Returns
    -------
    dict
        Trend parameters, data points, and predictions.
    """
    return {
        "description": "Deployment speed follows its own exponential trend",
        "fit": {
            "slope": _TREND_SLOPE,
            "intercept": _TREND_INTERCEPT,
            "r_squared": _TREND_R2,
            "halving_time_years": -math.log(2) / _TREND_SLOPE,
        },
        "data_points": [
            {"year": d[0], "years_to_25pct": d[1], "technology": d[2], "source": d[3]}
            for d in DEPLOYMENT_SPEED_DATA
        ],
        "predictions": {
            year: predicted_years_to_25pct(year)
            for year in range(2025, 2041)
        },
        "domain_friction": {
            domain: info["multiplier"]
            for domain, info in DOMAIN_FRICTION.items()
        },
    }
