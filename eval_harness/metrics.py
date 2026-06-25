"""Accuracy metrics for dispute eval predictions.

Decision accuracy is the headline number (does the engine reach the right
approve/reject/agent verdict). Approve precision/recall are tracked separately
because a false approve is a money loss, while a false reject is a CX cost — they
are not symmetric.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_DECISIONS = ("approve", "reject", "agent")


@dataclass
class CasePrediction:
    id: str
    expected_decision: str
    predicted_decision: str
    expected_category: Optional[str] = None
    predicted_category: Optional[str] = None


@dataclass
class Metrics:
    total: int
    decision_accuracy: float
    category_accuracy: float
    approve_precision: float
    approve_recall: float
    per_category: Dict[str, dict] = field(default_factory=dict)
    decision_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    mismatches: List[str] = field(default_factory=list)


def _ratio(num: int, denom: int, *, empty: float = 0.0) -> float:
    return round(num / denom, 4) if denom else empty


def compute_metrics(preds: List[CasePrediction]) -> Metrics:
    total = len(preds)
    if total == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0)

    correct_decisions = sum(1 for p in preds if p.expected_decision == p.predicted_decision)

    # Category accuracy only over cases that carry an expected category.
    cat_cases = [p for p in preds if p.expected_category is not None]
    correct_cats = sum(1 for p in cat_cases if p.expected_category == p.predicted_category)

    # Confusion matrix: expected -> predicted -> count.
    confusion: Dict[str, Dict[str, int]] = {
        e: {pdec: 0 for pdec in _DECISIONS} for e in _DECISIONS
    }
    for p in preds:
        if p.expected_decision in confusion and p.predicted_decision in confusion[p.expected_decision]:
            confusion[p.expected_decision][p.predicted_decision] += 1

    predicted_approve = sum(1 for p in preds if p.predicted_decision == "approve")
    expected_approve = sum(1 for p in preds if p.expected_decision == "approve")
    true_approve = sum(
        1 for p in preds if p.predicted_decision == "approve" and p.expected_decision == "approve"
    )

    # Per-category decision accuracy.
    by_cat: Dict[str, List[CasePrediction]] = defaultdict(list)
    for p in preds:
        key = p.expected_category or "(unlabelled)"
        by_cat[key].append(p)
    per_category = {
        cat: {
            "n": len(items),
            "decision_accuracy": _ratio(
                sum(1 for p in items if p.expected_decision == p.predicted_decision), len(items)
            ),
        }
        for cat, items in sorted(by_cat.items())
    }

    mismatches = [
        f"{p.id}: expected {p.expected_decision}, got {p.predicted_decision}"
        for p in preds
        if p.expected_decision != p.predicted_decision
    ]

    return Metrics(
        total=total,
        decision_accuracy=_ratio(correct_decisions, total),
        category_accuracy=_ratio(correct_cats, len(cat_cases), empty=1.0),
        approve_precision=_ratio(true_approve, predicted_approve, empty=1.0),
        approve_recall=_ratio(true_approve, expected_approve, empty=1.0),
        per_category=per_category,
        decision_confusion=confusion,
        mismatches=mismatches,
    )
