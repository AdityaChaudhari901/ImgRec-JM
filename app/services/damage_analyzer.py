"""Damage classification + severity scoring.

Gemini returns the raw damage observation; this module validates it against the
allowed taxonomy and maps severity to a numeric score the decision engine can
reason about deterministically.
"""

from typing import Optional

VALID_TYPES = {
    "crushed_packaging",
    "tear",
    "broken_seal",
    "leakage",
    "dent",
    "discoloration",
    "mold",
    "tamper",
    "resealed",
    "missing_component",
}

VALID_SEVERITIES = {"minor", "moderate", "severe"}

SEVERITY_SCORE = {"minor": 1, "moderate": 2, "severe": 3}


def normalize_damage(damage: dict) -> dict:
    """Validate/clean a raw damage dict against the allowed taxonomy."""
    data = dict(damage or {})

    detected = bool(data.get("detected", False))
    dtype = data.get("type")
    severity = data.get("severity")

    if dtype not in VALID_TYPES:
        dtype = None
    if severity not in VALID_SEVERITIES:
        severity = None

    if not detected:
        return {"detected": False, "type": None, "severity": None, "description": None}

    return {
        "detected": True,
        "type": dtype,
        "severity": severity,
        "description": data.get("description"),
    }


def severity_score(severity: Optional[str]) -> int:
    """Numeric severity: 0 (none) .. 3 (severe)."""
    return SEVERITY_SCORE.get(severity or "", 0)
