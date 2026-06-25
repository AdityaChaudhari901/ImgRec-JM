"""Deterministic dispute decisions. The model observes; this engine decides.

decide() dispatches per category to a pure function that returns a base
DisputeDecision (approve/reject + refund), then applies the shared escalation
gates (counterfeit, rebuttal, fraud signals, refund ceiling, assist/autonomous).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from app.config.settings import settings
from app.models.dispute_request import Shipment
from app.services.damage_analyzer import normalize_damage
from app.services.ocr_parser import days_until_expiry, final_printed_mrp, shelf_left_pct


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


def _full_refund(shipment: Optional[Shipment]) -> dict:
    # No shipment pricing -> the verdict stands but the caller computes the amount
    # from its own order data (amount left 0, type full_selling_price as the intent).
    if shipment is None or shipment.selling_price is None:
        return {"eligible": True, "amount": 0.0, "type": "full_selling_price",
                "assign_to_mpt": False, "seller_debit": False}
    amount = round(shipment.selling_price * shipment.quantity, 2)
    refund = {"eligible": True, "amount": amount, "type": "full_selling_price",
              "assign_to_mpt": False, "seller_debit": False}
    if shipment.seller_type == "3P":
        refund["assign_to_mpt"] = True
        refund["seller_debit"] = True
    return refund


def _product_type(shipment: Optional[Shipment]) -> str:
    # Without shipment we can't know the product class; default to non-FNV (the
    # 45-day Legal Metrology rule), the common case for packaged goods.
    return shipment.product_type if shipment else "non_fnv"


def _agent(flag: str, recommendation: str, refund: Optional[dict] = None,
           confidence: float = 0.4) -> DisputeDecision:
    return DisputeDecision(decision="agent", refund=refund or _no_refund(),
                           agent_flags=[flag], confidence=confidence,
                           recommendation=recommendation)


# ---- per-category branches -------------------------------------------------

def _decide_mrp(observations: dict, shipment: Optional[Shipment]) -> DisputeDecision:
    values = (observations.get("ocr") or {}).get("printed_mrp_values")
    printed = final_printed_mrp(values)
    if printed is None:
        return _agent("low_confidence", "MRP not readable from the image; agent to verify.")
    if shipment is None:
        return _agent("missing_shipment_data", "Shipment pricing missing; agent to verify overcharge.")

    # What the customer was actually billed per unit. Prioritise the invoice
    # amount over the selling price (per the requirement tip). Overcharge means
    # the customer paid MORE than the MRP printed on the delivered pack — the same
    # quantity drives both the decision and the refund, so they can't contradict.
    charged_unit = None
    if shipment.invoice_amount is not None and shipment.quantity:
        charged_unit = shipment.invoice_amount / shipment.quantity
    elif shipment.selling_price is not None:
        charged_unit = shipment.selling_price
    if charged_unit is None:
        return _agent("missing_shipment_data", "No charged price on the order; agent to verify overcharge.")

    charged_unit = round(charged_unit, 2)
    if charged_unit <= printed:
        return DisputeDecision(
            decision="reject", refund=_no_refund(), confidence=0.9,
            recommendation=f"Reject: charged ₹{charged_unit} ≤ printed MRP ₹{printed}; no overcharge.")

    # Overcharge confirmed: charged above the printed MRP.
    if shipment.seller_type == "3P":
        refund = _full_refund(shipment)
        rec = (f"Approve full refund ₹{refund['amount']} (3P overcharge: charged ₹{charged_unit} > "
               f"printed MRP ₹{printed}); assign ticket to MPT to debit seller.")
        return DisputeDecision(decision="approve", refund=refund, confidence=0.9, recommendation=rec)
    extra = round((charged_unit - printed) * shipment.quantity, 2)
    refund = {"eligible": True, "amount": extra, "type": "price_difference",
              "assign_to_mpt": False, "seller_debit": False}
    rec = (f"Approve ₹{extra} price-difference refund (1P overcharge: charged ₹{charged_unit} > "
           f"printed MRP ₹{printed} × qty {shipment.quantity}).")
    return DisputeDecision(decision="approve", refund=refund, confidence=0.9, recommendation=rec)


def _today() -> date:
    return date.today()


def _decide_expiry(observations: dict, shipment: Optional[Shipment]) -> DisputeDecision:
    ocr = observations.get("ocr") or {}
    product_type = _product_type(shipment)
    if product_type == "fnv":
        # FNV has variable shelf life — judge by visible quality instead.
        return _decide_poor_quality(observations, shipment)

    if product_type == "dairy":
        pct = shelf_left_pct(ocr.get("manufacture_date"), ocr.get("expiry_date"), _today())
        if pct is None:
            return _agent("low_confidence", "Dairy MFG/EXP not readable; agent to verify shelf life.")
        if pct < settings.dairy_min_shelf_pct:
            return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.9,
                                   recommendation=f"Approve: dairy shelf life {pct}% < {settings.dairy_min_shelf_pct}%.")
        return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.9,
                               recommendation=f"Reject: dairy shelf life {pct}% ≥ {settings.dairy_min_shelf_pct}%.")

    days = days_until_expiry(ocr.get("expiry_date"), _today())
    if days is None:
        return _agent("low_confidence", "Expiry date not readable; agent to verify.")
    if days <= settings.non_fnv_near_expiry_days:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.9,
                               recommendation=f"Approve: {days} days to expiry ≤ {settings.non_fnv_near_expiry_days}.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.9,
                           recommendation=f"Reject: {days} days to expiry > {settings.non_fnv_near_expiry_days}.")


def _decide_wrong_product(observations: dict, shipment: Shipment) -> DisputeDecision:
    pm = observations.get("product_match") or {}
    if pm.get("matches") is True:
        return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.85,
                               recommendation="Reject: delivered product matches the ordered item.")
    if pm.get("matches") is False:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.85,
                               recommendation="Approve: delivered product does not match the ordered item.")
    return _agent("low_confidence", "Could not confirm product match; agent to verify.")


def _decide_damaged(observations: dict, shipment: Shipment) -> DisputeDecision:
    dmg = normalize_damage(observations.get("damage") or {})
    if dmg.get("detected"):
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.85,
                               recommendation=f"Approve: {dmg.get('type') or 'damage'} visually confirmed.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.8,
                           recommendation="Reject: packaging appears intact, no damage visible.")


def _decide_poor_quality(observations: dict, shipment: Shipment) -> DisputeDecision:
    q = observations.get("quality") or {}
    if q.get("internal_defect"):
        return _agent("internal_defect",
                      "Internal defect / warranty / performance issue; route to agent.")
    if q.get("poor_quality") and q.get("supports_claim"):
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.75,
                               recommendation="Approve: visible quality defect supports the claim.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.7,
                           recommendation="Reject: product appears normal in the image.")


def _decide_smell(observations: dict, shipment: Shipment) -> DisputeDecision:
    spoil = (observations.get("spoilage") or {}).get("mold_or_visible_spoilage")
    detailed = int(observations.get("_desc_len", 0)) >= 30
    if spoil and detailed:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.65,
                               recommendation="Approve: detailed report plus visible spoilage proxy.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.6,
                           recommendation="Reject: insufficient evidence for a smell claim.")


def _decide_quantity(observations: dict, shipment: Optional[Shipment]) -> DisputeDecision:
    if shipment is None:
        return _agent("missing_shipment_data", "Ordered quantity unknown; agent to verify shortfall.")
    count = observations.get("count") or {}
    units = count.get("counted_units")
    conf = float(count.get("confidence", 0.0))
    if units is None or conf < 0.6:
        return _agent("low_confidence", "Could not count units confidently; agent to verify quantity.")
    if int(units) < shipment.quantity:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=conf,
                               recommendation=f"Approve: counted {int(units)} < ordered {shipment.quantity}.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=conf,
                           recommendation=f"Reject: counted {int(units)} ≥ ordered {shipment.quantity}.")


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
    "expiry": _decide_expiry,
    "wrong_product": _decide_wrong_product,
    "damaged": _decide_damaged,
    "poor_quality": _decide_poor_quality,
    "smell": _decide_smell,
    "quantity_mismatch": _decide_quantity,
}
