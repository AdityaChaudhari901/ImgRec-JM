"""Labelled eval cases loaded from a JSONL manifest.

Each line is one labelled dispute. `observations` is the Gemini observation the
engine would receive (used in engine mode); `images` are real image paths (used
in e2e mode). `expected` carries the ground-truth decision and, optionally, the
ground-truth category.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EvalCase:
    id: str
    shipment: Dict
    expected_decision: str
    category: Optional[str] = None          # provided dispute_category (None => classify)
    expected_category: Optional[str] = None
    is_rebuttal: bool = False
    ticket: Dict = field(default_factory=dict)
    observations: Dict = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    signals: Optional[Dict] = None
    today: Optional[str] = None             # ISO date the expiry cases are written against


def _to_case(row: Dict) -> EvalCase:
    expected = row.get("expected") or {}
    return EvalCase(
        id=row["id"],
        shipment=row["shipment"],
        expected_decision=expected["decision"],
        category=row.get("category"),
        expected_category=expected.get("category"),
        is_rebuttal=bool(row.get("is_rebuttal", False)),
        ticket=row.get("ticket") or {},
        observations=row.get("observations") or {},
        images=row.get("images") or [],
        signals=row.get("signals"),
        today=row.get("today"),
    )


def load_manifest(path: str | Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        cases.append(_to_case(json.loads(line)))
    return cases
