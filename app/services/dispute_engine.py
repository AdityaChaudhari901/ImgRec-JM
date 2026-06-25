"""Deterministic dispute decisions. The model observes; this engine decides.

decide() dispatches per category to a pure function that returns a base
DisputeDecision (approve/reject + refund), then applies the shared escalation
gates (counterfeit, rebuttal, fraud signals, refund ceiling, assist/autonomous).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.config.settings import settings
from app.models.dispute_request import Shipment
from app.services.ocr_parser import final_printed_mrp


@dataclass
class DisputeDecision:
    decision: str  # approve | reject | agent
    refund: dict
    agent_flags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    recommendation: str = ""


def _no_refund() -> dict:
    return {"eligible": False, "amount": 0.0, "type": "none",
            "assign_to_mpt": False, "seller_debit": False}


def _full_refund(shipment: Shipment) -> dict:
    unit = shipment.selling_price if shipment.selling_price is not None else 0.0
    amount = round(unit * shipment.quantity, 2)
    refund = {"eligible": True, "amount": amount, "type": "full_selling_price",
              "assign_to_mpt": False, "seller_debit": False}
    if shipment.seller_type == "3P":
        refund["assign_to_mpt"] = True
        refund["seller_debit"] = True
    return refund


def _agent(flag: str, recommendation: str, refund: Optional[dict] = None,
           confidence: float = 0.4) -> DisputeDecision:
    return DisputeDecision(decision="agent", refund=refund or _no_refund(),
                           agent_flags=[flag], confidence=confidence,
                           recommendation=recommendation)


# ---- per-category branches -------------------------------------------------

def _decide_mrp(observations: dict, shipment: Shipment) -> DisputeDecision:
    values = (observations.get("ocr") or {}).get("printed_mrp_values")
    printed = final_printed_mrp(values)
    if printed is None:
        return _agent("low_confidence", "MRP not readable from the image; agent to verify.")
    if shipment.mrp is None:
        return _agent("missing_shipment_data", "Invoice MRP missing; agent to verify overcharge.")
    if printed >= shipment.mrp:
        return DisputeDecision(
            decision="reject", refund=_no_refund(), confidence=0.9,
            recommendation=(f"Reject: printed MRP ₹{printed} ≥ invoice MRP ₹{shipment.mrp}; "
                            "no overcharge."),
        )
    # Overcharge confirmed: printed MRP on pack is below the MRP charged.
    if shipment.seller_type == "3P":
        refund = _full_refund(shipment)
        rec = (f"Approve full refund ₹{refund['amount']} (3P overcharge: printed ₹{printed} < "
               f"invoice MRP ₹{shipment.mrp}); assign ticket to MPT to debit seller.")
        return DisputeDecision(decision="approve", refund=refund, confidence=0.9, recommendation=rec)
    charged_unit = shipment.selling_price if shipment.selling_price is not None else shipment.mrp
    extra = round(max(0.0, charged_unit - printed) * shipment.quantity, 2)
    refund = {"eligible": extra > 0, "amount": extra, "type": "price_difference",
              "assign_to_mpt": False, "seller_debit": False}
    rec = (f"Approve ₹{extra} price-difference refund (1P overcharge: printed ₹{printed} < "
           f"invoice MRP ₹{shipment.mrp} × qty {shipment.quantity}).")
    return DisputeDecision(decision="approve", refund=refund, confidence=0.9, recommendation=rec)


# ---- escalation gates ------------------------------------------------------

def _autonomous_categories() -> set:
    return {c.strip() for c in settings.dispute_autonomous_categories.split(",") if c.strip()}


def _apply_gates(base: DisputeDecision, category: str, shipment: Shipment,
                 observations: dict, is_rebuttal: bool, signals: dict) -> DisputeDecision:
    flags = list(base.agent_flags)
    decision = base.decision

    # Hard fraud signals -> reject (defensible, auditable).
    if signals.get("dedup_cross") or signals.get("web_hard"):
        return DisputeDecision(decision="reject", refund=_no_refund(),
                               agent_flags=["fraud_signal"], confidence=1.0,
                               recommendation="Reject: image reused across claims / found on public web.")

    # Counterfeit & rebuttal & advisory-AI -> always human.
    if observations.get("counterfeit_suspected"):
        return _agent("counterfeit", "Counterfeit suspected; route to agent investigation.",
                      refund=base.refund)
    if is_rebuttal:
        return _agent("rebuttal", f"Post-rejection rebuttal. AI recommendation: {base.recommendation}",
                      refund=base.refund)
    if signals.get("ai_probability", 0.0) >= settings.ai_detection_min_confidence:
        return _agent("fraud_signal", "Image may be AI-generated; agent to verify. "
                      f"AI recommendation: {base.recommendation}", refund=base.refund)

    if decision == "agent":
        return DisputeDecision(decision="agent", refund=base.refund, agent_flags=flags or ["low_confidence"],
                               confidence=base.confidence, recommendation=base.recommendation)

    # Refund ceiling.
    if decision == "approve" and base.refund.get("amount", 0.0) >= settings.refund_auto_approve_max:
        flags.append("high_value")
        return DisputeDecision(decision="agent", refund=base.refund, agent_flags=flags,
                               confidence=base.confidence,
                               recommendation=f"High-value refund. AI recommendation: {base.recommendation}")

    # Assist mode / progressive autonomy is enforced at the router (route=agent),
    # leaving the engine's approve/reject recommendation intact here.
    return DisputeDecision(decision=decision, refund=base.refund, agent_flags=flags,
                           confidence=base.confidence, recommendation=base.recommendation)


def decide(category: Optional[str], source: str, observations: dict, shipment: Shipment,
           is_rebuttal: bool, signals: dict) -> DisputeDecision:
    if not category:
        return _agent("insufficient_data",
                      "No category could be resolved from description, notes, or disposition.")
    branch = _BRANCHES.get(category)
    if branch is None:
        return _agent("low_confidence", f"Category '{category}' not yet automated; route to agent.")
    base = branch(observations, shipment)
    return _apply_gates(base, category, shipment, observations, is_rebuttal, signals)


# Registered at module bottom so every _decide_* is defined before lookup.
_BRANCHES = {
    "mrp_abuse": _decide_mrp,
}
