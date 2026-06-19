"""Final action resolution: refund / exchange / no_action.

The action is computed deterministically from `status` + damage `severity` rather
than trusted blindly from the model output — the model supplies observations, the
business rules here supply the decision. This keeps refund eligibility auditable.
"""

import random
import string
import time
from datetime import datetime, timezone
from typing import Optional

from app.config.settings import settings
from app.models.response import (
    ActionResult,
    DamageResult,
    OCRResult,
    ScanResponse,
)
from app.models.verify_response import AIGeneratedCheck
from app.services.damage_analyzer import normalize_damage
from app.services.ocr_parser import (
    calculate_days_since_expiry,
    normalize_ocr_dates,
)

VALID_STATUSES = {"expired", "damaged", "valid", "unclear"}


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def generate_request_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"req_{int(time.time())}_{suffix}"


def determine_action(
    status: str,
    severity: Optional[str],
    days_since_expiry: Optional[int] = None,
    model_message: Optional[str] = None,
) -> ActionResult:
    """Apply the JioMart refund/exchange decision matrix."""
    if status == "expired":
        days = f"{days_since_expiry} days ago" if days_since_expiry else "recently"
        return ActionResult(
            type="initiate_refund",
            message=model_message or f"Product expired {days}. Refund process triggered.",
            refund_eligible=True,
            priority="high",
        )

    if status == "damaged":
        if severity == "severe":
            return ActionResult(
                type="initiate_refund",
                message=model_message or "Severe damage detected. Refund process triggered.",
                refund_eligible=True,
                priority="high",
            )
        if severity == "moderate":
            return ActionResult(
                type="initiate_exchange",
                message=model_message or "Moderate damage detected. Exchange initiated.",
                refund_eligible=False,
                priority="medium",
            )
        # minor (or unspecified) damage
        return ActionResult(
            type="initiate_exchange",
            message=model_message or "Minor damage detected. Exchange initiated.",
            refund_eligible=False,
            priority="low",
        )

    # valid / unclear -> no action
    default_msg = (
        "Product appears valid. No action required."
        if status == "valid"
        else "Could not determine product condition. No action taken."
    )
    return ActionResult(
        type="no_action",
        message=default_msg,
        refund_eligible=False,
        priority="low",
    )


def _manual_review_action(reason: str) -> ActionResult:
    return ActionResult(
        type="no_action",
        message=(
            f"{reason} No automated refund or exchange has been initiated; "
            "route this claim to manual review."
        ),
        refund_eligible=False,
        priority="high",
    )


def _synthetic_evidence_action(ai_check: AIGeneratedCheck) -> ActionResult:
    return _manual_review_action(
        f"AI-generated or synthetic evidence suspected ({ai_check.ai_probability:.0%})."
    )


def _action_looks_automated(model_action: dict) -> bool:
    message = str(model_action.get("message") or "").lower()
    automated_terms = ("refund", "exchange", "replace", "replacement", "initiated")
    return (
        model_action.get("type") in {"initiate_refund", "initiate_exchange"}
        or model_action.get("refund_eligible") is True
        or any(term in message for term in automated_terms)
    )


def _needs_manual_review(status: str, damage_data: dict, model_action: dict) -> bool:
    """Detect contradictory model observations before any automated action."""
    if status not in {"valid", "unclear"}:
        return False

    has_reviewable_damage = damage_data.get("detected") and damage_data.get("severity") in {
        "moderate",
        "severe",
    }
    return bool(has_reviewable_damage or _action_looks_automated(model_action))


def _has_authenticity_assessment(gemini_output: dict) -> bool:
    ai_generated = gemini_output.get("ai_generated")
    return isinstance(ai_generated, dict) and "ai_probability" in ai_generated


def _requires_authenticity_assessment(status: str, model_action: dict) -> bool:
    return status in {"expired", "damaged"} or _action_looks_automated(model_action)


def build_response(
    gemini_output: dict,
    order_id: str,
    user_id: str,
    ai_check: Optional[AIGeneratedCheck] = None,
) -> ScanResponse:
    ocr_data = normalize_ocr_dates(gemini_output.get("ocr", {}))
    damage_data = normalize_damage(gemini_output.get("damage", {}))

    status = gemini_output.get("status", "unclear")
    if status not in VALID_STATUSES:
        status = "unclear"

    days_since_expiry = calculate_days_since_expiry(ocr_data.get("expiry_date"))
    model_action = gemini_output.get("action") or {}

    ai_guarded = (
        ai_check is not None
        and ai_check.is_ai_generated
        and ai_check.ai_probability >= settings.ai_detection_min_confidence
    )

    if ai_guarded:
        return ScanResponse(
            success=True,
            request_id=generate_request_id(),
            order_id=order_id,
            user_id=user_id,
            status="unclear",
            confidence=round(1.0 - ai_check.ai_probability, 3),
            ocr=OCRResult(
                manufacture_date=ocr_data.get("manufacture_date"),
                expiry_date=ocr_data.get("expiry_date"),
                batch_no=ocr_data.get("batch_no"),
                days_since_expiry=days_since_expiry,
                raw_text=ocr_data.get("raw_text"),
            ),
            damage=DamageResult(
                detected=False,
                type=None,
                severity=None,
                description="Synthetic evidence suspected; visual damage was ignored.",
            ),
            ai_generated=ai_check,
            action=_synthetic_evidence_action(ai_check),
            processed_at=datetime.now(timezone.utc),
            model_used=settings.gemini_model,
        )

    if _requires_authenticity_assessment(status, model_action) and not _has_authenticity_assessment(
        gemini_output
    ):
        return ScanResponse(
            success=True,
            request_id=generate_request_id(),
            order_id=order_id,
            user_id=user_id,
            status="unclear",
            confidence=_confidence(gemini_output.get("confidence")),
            ocr=OCRResult(
                manufacture_date=ocr_data.get("manufacture_date"),
                expiry_date=ocr_data.get("expiry_date"),
                batch_no=ocr_data.get("batch_no"),
                days_since_expiry=days_since_expiry,
                raw_text=ocr_data.get("raw_text"),
            ),
            damage=DamageResult(
                detected=False,
                type=None,
                severity=None,
                description="Evidence authenticity was not assessed; visual damage was ignored.",
            ),
            ai_generated=ai_check,
            action=_manual_review_action("Evidence authenticity was not assessed."),
            processed_at=datetime.now(timezone.utc),
            model_used=settings.gemini_model,
        )

    if _needs_manual_review(status, damage_data, model_action):
        return ScanResponse(
            success=True,
            request_id=generate_request_id(),
            order_id=order_id,
            user_id=user_id,
            status="unclear",
            confidence=_confidence(gemini_output.get("confidence")),
            ocr=OCRResult(
                manufacture_date=ocr_data.get("manufacture_date"),
                expiry_date=ocr_data.get("expiry_date"),
                batch_no=ocr_data.get("batch_no"),
                days_since_expiry=days_since_expiry,
                raw_text=ocr_data.get("raw_text"),
            ),
            damage=DamageResult(
                detected=False,
                type=None,
                severity=None,
                description="Inconsistent model evidence; visual damage requires manual review.",
            ),
            ai_generated=ai_check,
            action=_manual_review_action("Product evidence is inconsistent."),
            processed_at=datetime.now(timezone.utc),
            model_used=settings.gemini_model,
        )

    action = determine_action(
        status=status,
        severity=damage_data.get("severity"),
        days_since_expiry=days_since_expiry,
        model_message=model_action.get("message"),
    )

    return ScanResponse(
        success=True,
        request_id=generate_request_id(),
        order_id=order_id,
        user_id=user_id,
        status=status,
        confidence=_confidence(gemini_output.get("confidence")),
        ocr=OCRResult(
            manufacture_date=ocr_data.get("manufacture_date"),
            expiry_date=ocr_data.get("expiry_date"),
            batch_no=ocr_data.get("batch_no"),
            days_since_expiry=days_since_expiry,
            raw_text=ocr_data.get("raw_text"),
        ),
        damage=DamageResult(
            detected=damage_data.get("detected", False),
            type=damage_data.get("type"),
            severity=damage_data.get("severity"),
            description=damage_data.get("description"),
        ),
        ai_generated=ai_check,
        action=action,
        processed_at=datetime.now(timezone.utc),
        model_used=settings.gemini_model,
    )
