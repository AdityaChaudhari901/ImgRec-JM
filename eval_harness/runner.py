"""Run labelled cases through the real decision path and collect predictions.

Engine mode mirrors the router: classify the category, inject the description
length the smell branch needs, build fraud signals, pin "today" for expiry cases,
then call the deterministic engine. Using the production classifier + engine (not
a copy) is the point — the eval measures what actually ships.
"""

from __future__ import annotations

from datetime import date
from typing import List

from app.models.dispute_request import Shipment, Ticket
from app.services import dispute_engine
from app.services.category_classifier import classify_category

from eval_harness.dataset import EvalCase
from eval_harness.metrics import CasePrediction


def _signals(case: EvalCase) -> dict:
    if case.signals is not None:
        return case.signals
    ai = (case.observations.get("ai_generated") or {}).get("ai_probability", 0.0)
    return {"ai_probability": float(ai), "dedup_cross": False, "web_hard": False}


def _predict(case: EvalCase, observations: dict) -> CasePrediction:
    """Run the shared decision path over a given observation set."""
    ticket = Ticket(**case.ticket)
    category, source = classify_category(case.category, ticket)

    obs = dict(observations)
    obs["_desc_len"] = len((ticket.description or "").strip())

    shipment = Shipment(**case.shipment)

    # Pin the engine's notion of "today" so expiry cases are deterministic.
    original_today = dispute_engine._today
    if case.today:
        pinned = date.fromisoformat(case.today)
        dispute_engine._today = lambda: pinned
    try:
        decision = dispute_engine.decide(
            category, source, obs, shipment, case.is_rebuttal, _signals(case)
        )
    finally:
        dispute_engine._today = original_today

    return CasePrediction(
        id=case.id,
        expected_decision=case.expected_decision,
        predicted_decision=decision.decision,
        expected_category=case.expected_category,
        predicted_category=category,
    )


def run_case(case: EvalCase) -> CasePrediction:
    """Engine mode: use the labelled observations already in the case."""
    return _predict(case, case.observations)


def run_dataset(cases: List[EvalCase]) -> List[CasePrediction]:
    return [run_case(c) for c in cases]


async def run_case_e2e(case: EvalCase) -> CasePrediction:
    """End-to-end mode: fetch observations from Gemini over the real images, then
    run the same engine. Requires real images + provider credentials."""
    # Imported lazily so engine-mode eval never needs the Gemini SDK/credentials.
    from app.services.dispute_service import analyze_dispute

    comment = (case.ticket.get("description") or case.ticket.get("title") or "")
    observations = await analyze_dispute(
        case.images, case.category, case.shipment.get("product_name", ""), comment
    )
    return _predict(case, observations)


async def run_dataset_e2e(cases: List[EvalCase]) -> List[CasePrediction]:
    return [await run_case_e2e(c) for c in cases]
