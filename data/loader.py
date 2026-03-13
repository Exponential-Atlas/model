"""
Exponential Atlas v6 — Domain Data Loader
==========================================
Load domain JSON files from disk, validate, and provide convenient accessors.
"""

import json
import os
from pathlib import Path
from typing import Optional

from .schema import validate_domain


# Default path to domain JSON files
_DOMAINS_DIR = Path(__file__).parent / "domains"


def load_domain(
    domain_id: str,
    domains_dir: Optional[Path] = None,
    validate: bool = True,
) -> dict:
    """
    Load a single domain JSON file by its id (filename without .json).

    Parameters
    ----------
    domain_id : str
        The domain identifier, e.g. "solar_module".
    domains_dir : Path, optional
        Override the default domains directory.
    validate : bool
        If True (default), validate the loaded data and raise on errors.

    Returns
    -------
    dict
        The parsed domain data.

    Raises
    ------
    FileNotFoundError
        If the domain JSON file does not exist.
    ValueError
        If validation fails and ``validate`` is True.
    """
    base = domains_dir or _DOMAINS_DIR
    filepath = base / f"{domain_id}.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Domain file not found: {filepath}"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if validate:
        errors = validate_domain(data)
        if errors:
            raise ValueError(
                f"Validation errors in '{domain_id}':\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    return data


def load_all_domains(
    domains_dir: Optional[Path] = None,
    validate: bool = True,
) -> dict[str, dict]:
    """
    Load every domain JSON file from the domains directory.

    Parameters
    ----------
    domains_dir : Path, optional
        Override the default domains directory.
    validate : bool
        If True (default), validate each file and raise on first error.

    Returns
    -------
    dict[str, dict]
        Mapping of domain_id -> domain data.
    """
    base = domains_dir or _DOMAINS_DIR

    if not base.exists():
        raise FileNotFoundError(f"Domains directory not found: {base}")

    domains: dict[str, dict] = {}

    for filepath in sorted(base.glob("*.json")):
        domain_id = filepath.stem
        domains[domain_id] = load_domain(
            domain_id, domains_dir=base, validate=validate
        )

    return domains


def get_domain_data_points(domain: dict) -> tuple[list[float], list[float]]:
    """
    Extract parallel lists of years and values from a domain dict.

    Parameters
    ----------
    domain : dict
        A loaded domain data dictionary.

    Returns
    -------
    tuple[list[float], list[float]]
        (years, values) — both lists have the same length, in chronological order.
    """
    years: list[float] = []
    values: list[float] = []

    for pt in domain.get("data_points", []):
        years.append(float(pt["year"]))
        values.append(float(pt["value"]))

    return years, values


def get_wrights_law_data(
    domain: dict,
) -> Optional[tuple[list[float], list[float], list[float]]]:
    """
    Extract Wright's Law cumulative-production data from a domain dict.

    Parameters
    ----------
    domain : dict
        A loaded domain data dictionary.

    Returns
    -------
    tuple[list[float], list[float], list[float]] or None
        (years, cumulative_production_values, price_values) if Wright's Law
        data is present; None otherwise.
    """
    wl = domain.get("wrights_law")
    if wl is None:
        return None

    cp = wl.get("cumulative_production", [])
    if not cp:
        return None

    # Get price data aligned to production years
    _, price_values = get_domain_data_points(domain)
    price_years = [pt["year"] for pt in domain["data_points"]]

    cp_years: list[float] = []
    cp_vals: list[float] = []
    aligned_prices: list[float] = []

    for entry in cp:
        cy = float(entry["year"])
        cv = float(entry["value"])
        # Find closest price data point within 2 years
        best_idx = None
        best_dist = float("inf")
        for idx, py in enumerate(price_years):
            dist = abs(py - cy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None and best_dist <= 2.0:
            cp_years.append(cy)
            cp_vals.append(cv)
            aligned_prices.append(price_values[best_idx])

    if len(cp_years) < 3:
        return None

    return cp_years, cp_vals, aligned_prices
