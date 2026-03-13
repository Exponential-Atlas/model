"""
Exponential Atlas v6 — Domain JSON Schema & Validation
=======================================================
Defines the canonical schema for domain data files and provides
validation to ensure data integrity before any fitting or simulation.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

VALID_DIRECTIONS = {"increasing", "decreasing"}
VALID_CATEGORIES = {
    "AI", "Energy", "Biology", "Compute", "Space",
    "Robotics", "Environment", "BCI", "VR/AR", "Manufacturing",
}
VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}
VALID_BEST_FIT_METHODS = {"log_linear", "wrights_law", "piecewise"}

# Minimum data points required for any domain to be usable
MIN_DATA_POINTS = 3


# ---------------------------------------------------------------------------
# Dataclass representation (for documentation and IDE support)
# ---------------------------------------------------------------------------

@dataclass
class DataPoint:
    year: float
    value: float
    source: str
    source_url: str


@dataclass
class CumulativeProductionPoint:
    year: float
    value: float
    unit: str
    source_url: str


@dataclass
class WrightsLawSpec:
    learning_rate: float
    cumulative_production: list[CumulativeProductionPoint] = field(default_factory=list)


@dataclass
class PiecewiseSpec:
    breakpoint_year: Optional[float] = None


@dataclass
class DomainSchema:
    id: str
    name: str
    description: str
    unit: str
    direction: str  # "increasing" or "decreasing"
    category: str
    confidence: str  # "high", "medium", "low"
    best_fit: str  # "log_linear", "wrights_law", "piecewise"
    physical_floor: Optional[float] = None
    physical_ceiling: Optional[float] = None
    wrights_law: Optional[WrightsLawSpec] = None
    piecewise: Optional[PiecewiseSpec] = None
    data_points: list[DataPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_domain(data: dict) -> list[str]:
    """
    Validate a domain data dictionary against the schema.

    Returns a list of error strings. An empty list means the data is valid.
    """
    errors: list[str] = []

    # --- Required top-level fields ---
    required_fields = [
        "id", "name", "description", "unit", "direction",
        "category", "confidence", "best_fit", "data_points",
    ]
    for f in required_fields:
        if f not in data:
            errors.append(f"Missing required field: '{f}'")

    # If critical fields are missing, further checks are not meaningful
    if errors:
        return errors

    # --- Type checks ---
    if not isinstance(data["id"], str) or not data["id"].strip():
        errors.append("'id' must be a non-empty string")
    if not isinstance(data["name"], str) or not data["name"].strip():
        errors.append("'name' must be a non-empty string")
    if not isinstance(data["description"], str):
        errors.append("'description' must be a string")
    if not isinstance(data["unit"], str) or not data["unit"].strip():
        errors.append("'unit' must be a non-empty string")

    # --- Enum checks ---
    if data["direction"] not in VALID_DIRECTIONS:
        errors.append(
            f"'direction' must be one of {VALID_DIRECTIONS}, got '{data['direction']}'"
        )
    if data["category"] not in VALID_CATEGORIES:
        errors.append(
            f"'category' must be one of {VALID_CATEGORIES}, got '{data['category']}'"
        )
    if data["confidence"] not in VALID_CONFIDENCE_LEVELS:
        errors.append(
            f"'confidence' must be one of {VALID_CONFIDENCE_LEVELS}, "
            f"got '{data['confidence']}'"
        )
    if data["best_fit"] not in VALID_BEST_FIT_METHODS:
        errors.append(
            f"'best_fit' must be one of {VALID_BEST_FIT_METHODS}, "
            f"got '{data['best_fit']}'"
        )

    # --- Physical bounds ---
    floor = data.get("physical_floor")
    ceiling = data.get("physical_ceiling")
    if floor is not None and not isinstance(floor, (int, float)):
        errors.append("'physical_floor' must be a number or null")
    if ceiling is not None and not isinstance(ceiling, (int, float)):
        errors.append("'physical_ceiling' must be a number or null")
    if (
        floor is not None
        and ceiling is not None
        and isinstance(floor, (int, float))
        and isinstance(ceiling, (int, float))
        and floor >= ceiling
    ):
        errors.append("'physical_floor' must be less than 'physical_ceiling'")

    # --- Data points ---
    pts = data["data_points"]
    if not isinstance(pts, list):
        errors.append("'data_points' must be a list")
    elif len(pts) < MIN_DATA_POINTS:
        errors.append(
            f"Need at least {MIN_DATA_POINTS} data points, got {len(pts)}"
        )
    else:
        prev_year = None
        for i, pt in enumerate(pts):
            prefix = f"data_points[{i}]"
            if not isinstance(pt, dict):
                errors.append(f"{prefix}: must be a dict")
                continue
            # Required sub-fields
            for sf in ["year", "value", "source", "source_url"]:
                if sf not in pt:
                    errors.append(f"{prefix}: missing '{sf}'")
            if "year" in pt:
                if not isinstance(pt["year"], (int, float)):
                    errors.append(f"{prefix}: 'year' must be a number")
                elif pt["year"] < 1800 or pt["year"] > 2100:
                    errors.append(
                        f"{prefix}: 'year' {pt['year']} outside plausible range [1800, 2100]"
                    )
                elif prev_year is not None and pt["year"] < prev_year:
                    errors.append(
                        f"{prefix}: data points must be in chronological order "
                        f"(year {pt['year']} < previous {prev_year})"
                    )
                else:
                    prev_year = pt["year"]
            if "value" in pt and not isinstance(pt["value"], (int, float)):
                errors.append(f"{prefix}: 'value' must be a number")
            if "source" in pt and not isinstance(pt["source"], str):
                errors.append(f"{prefix}: 'source' must be a string")
            if "source_url" in pt and not isinstance(pt["source_url"], str):
                errors.append(f"{prefix}: 'source_url' must be a string")

    # --- Wright's Law ---
    if data["best_fit"] == "wrights_law":
        wl = data.get("wrights_law")
        if wl is None:
            errors.append("'wrights_law' section required when best_fit is 'wrights_law'")
        elif not isinstance(wl, dict):
            errors.append("'wrights_law' must be a dict")
        else:
            if "learning_rate" not in wl:
                errors.append("'wrights_law.learning_rate' is required")
            elif not isinstance(wl["learning_rate"], (int, float)):
                errors.append("'wrights_law.learning_rate' must be a number")
            elif not (0 < wl["learning_rate"] < 1):
                errors.append(
                    "'wrights_law.learning_rate' must be between 0 and 1 "
                    f"(exclusive), got {wl['learning_rate']}"
                )
            cp = wl.get("cumulative_production")
            if cp is None:
                errors.append(
                    "'wrights_law.cumulative_production' is required"
                )
            elif not isinstance(cp, list) or len(cp) < 3:
                errors.append(
                    "'wrights_law.cumulative_production' must be a list with >= 3 entries"
                )
            else:
                for j, entry in enumerate(cp):
                    cpx = f"wrights_law.cumulative_production[{j}]"
                    if not isinstance(entry, dict):
                        errors.append(f"{cpx}: must be a dict")
                        continue
                    for sf in ["year", "value", "unit", "source_url"]:
                        if sf not in entry:
                            errors.append(f"{cpx}: missing '{sf}'")

    # --- Piecewise ---
    if data["best_fit"] == "piecewise":
        pw = data.get("piecewise")
        if pw is None:
            errors.append("'piecewise' section required when best_fit is 'piecewise'")
        elif not isinstance(pw, dict):
            errors.append("'piecewise' must be a dict")
        else:
            bp = pw.get("breakpoint_year")
            if bp is not None and not isinstance(bp, (int, float)):
                errors.append("'piecewise.breakpoint_year' must be a number or null")

    return errors
