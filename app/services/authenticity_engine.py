"""Deterministic authenticity scoring + routing.

Takes Gemini's observations + the AI-detector verdict and computes the final
0..1 authenticity score, verdict, and recommended action *in code* (never from
the model), so refund routing is auditable and tunable via settings.

Scoring:
  base   = w_align * alignment + w_product * product_match
  score  = base
           * (1 - ai_penalty * ai_confidence)   if image looks AI-generated
           - flag_penalty * len(other_flags)
  clamped to [0, 1].

Routing:
  - A confident AI-generated verdict from a *reliable* source (sightengine)
    forces "reject"; from the internal/advisory provider it forces at most
    "manual_review" (we never auto-reject a real customer on a weak signal).
  - Otherwise: score >= auto_approve_threshold  -> authentic / auto_approve
               score >= review_threshold        -> review    / manual_review
               else                              -> likely_fraud / reject
"""

import random
import string
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from app.config.settings import settings
from app.models.verify_response import (
    AIGeneratedCheck,
    AlignmentCheck,
    AuthenticityChecks,
    ProductMatchCheck,
    RecognitionResult,
    VerifyClaimResponse,
    WebProvenanceCheck,
)
from app.services.dedup_service import DedupResult
from app.services.web_provenance import WebProvenanceResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_request_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"vfy_{int(time.time())}_{suffix}"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _web_hard(web_result: Optional[WebProvenanceResult]) -> bool:
    """A confirmed full match across enough distinct domains is a hard fraud
    signal — a genuine damage photo does not live on multiple unrelated sites."""
    return bool(
        web_result
        and web_result.checked
        and web_result.full_match_count > 0
        and web_result.distinct_domains >= settings.web_match_hard_min_domains
    )


def score_claim(
    gemini: dict,
    ai_check: AIGeneratedCheck,
    dedup_result: Optional[DedupResult] = None,
    web_result: Optional[WebProvenanceResult] = None,
) -> Tuple[float, str, str, AuthenticityChecks]:
    """Return (authenticity_score, verdict, recommended_action, checks)."""
    align_raw = gemini.get("image_comment_alignment", {}) or {}
    prod_raw = gemini.get("product_match", {}) or {}

    alignment = AlignmentCheck(
        score=_clamp(_f(align_raw.get("score"))),
        aligned=bool(align_raw.get("aligned", False)),
        reason=align_raw.get("reason"),
    )
    product = ProductMatchCheck(
        matches=bool(prod_raw.get("matches", False)),
        score=_clamp(_f(prod_raw.get("score"))),
        detected_product=prod_raw.get("detected_product"),
        reason=prod_raw.get("reason"),
    )
    other_flags = [str(f) for f in (gemini.get("other_flags") or []) if str(f).strip()]

    # HARD signal: the same photo already backs a different claim (different order
    # or user). This is deterministic and auditable, so it may drive an automated
    # reject — unlike the advisory AI score.
    hard_duplicate = bool(dedup_result and dedup_result.is_cross_claim_duplicate)
    if hard_duplicate:
        prior = ", ".join(dedup_result.matched_order_ids())
        other_flags.append(f"duplicate_image_across_claims: prior orders [{prior}]")

    web_hard = _web_hard(web_result)
    web_check = None
    if web_result is not None and web_result.checked:
        reason = (
            f"image found on {web_result.distinct_domains} domain(s), "
            f"{web_result.full_match_count} full match(es)"
        )
        web_check = WebProvenanceCheck(
            checked=True,
            full_matches=web_result.full_match_count,
            partial_matches=web_result.partial_match_count,
            distinct_domains=web_result.distinct_domains,
            reason=reason,
        )
        if web_hard:
            other_flags.append(f"web_download_match: {reason}")
    elif web_result is not None:
        web_check = WebProvenanceCheck(checked=False)

    checks = AuthenticityChecks(
        ai_generated=ai_check,
        image_comment_alignment=alignment,
        product_match=product,
        other_flags=other_flags,
        web_provenance=web_check,
    )

    # Base score from the two reliable signals.
    base = (
        settings.authenticity_weight_alignment * alignment.score
        + settings.authenticity_weight_product_match * product.score
    )

    score = base
    ai_confident = (
        ai_check.is_ai_generated
        and ai_check.ai_probability >= settings.ai_detection_min_confidence
    )
    if ai_check.is_ai_generated:
        score *= 1.0 - settings.authenticity_ai_penalty * ai_check.ai_probability
    score -= settings.authenticity_flag_penalty * len(other_flags)
    if web_result is not None and web_result.checked and not web_hard:
        web_hits = min(
            web_result.full_match_count + web_result.partial_match_count,
            settings.web_match_penalty_cap,
        )
        score -= settings.web_match_soft_penalty * web_hits
    score = _clamp(score)
    if hard_duplicate or web_hard:
        score = 0.0  # a confirmed reuse / web-download is inauthentic by definition

    verdict, action = _route(score, ai_confident, ai_check.source, hard_duplicate or web_hard)
    return round(score, 3), verdict, action, checks


def decision_confidence(
    score: float, action: str, ai_check: AIGeneratedCheck, ai_confident: bool
) -> float:
    """How decisively the recommendation sits in its band, in [0,1].

    - AI-forced routing -> confidence is the AI probability itself.
    - auto_approve / reject -> grows with distance past the band boundary.
    - manual_review (the uncertain middle) -> peaks at the band centre.
    """
    approve_t = settings.authenticity_auto_approve_threshold
    review_t = settings.authenticity_review_threshold

    if ai_confident:
        return round(ai_check.ai_probability, 3)

    if action == "auto_approve":
        span = max(1e-6, 1.0 - approve_t)
        return round(_clamp(0.5 + 0.5 * (score - approve_t) / span), 3)
    if action == "reject":
        span = max(1e-6, review_t)
        return round(_clamp(0.5 + 0.5 * (review_t - score) / span), 3)

    # manual_review: most confident it's a genuine "review" at the band centre.
    centre = (approve_t + review_t) / 2
    half = max(1e-6, (approve_t - review_t) / 2)
    return round(_clamp(0.5 + 0.5 * (1 - abs(score - centre) / half)), 3)


def _route(
    score: float, ai_confident: bool, ai_source: str, hard_duplicate: bool = False
) -> Tuple[str, str]:
    # A confirmed cross-claim image reuse is the strongest, hardest signal.
    if hard_duplicate:
        return "likely_fraud", "reject"

    # A confident AI-generated image dominates routing.
    if ai_confident:
        if ai_source == "sightengine":
            return "likely_fraud", "reject"
        # Advisory internal detector -> always send to a human, never auto-reject.
        return "review", "manual_review"

    if score >= settings.authenticity_auto_approve_threshold:
        return "authentic", "auto_approve"
    if score >= settings.authenticity_review_threshold:
        return "review", "manual_review"
    return "likely_fraud", "reject"


def _agent_comment(
    gemini: dict, checks: AuthenticityChecks, score: float, verdict: str, action: str
) -> str:
    parts = [gemini.get("summary", "").strip() or "Claim analysed."]
    a = checks.image_comment_alignment
    p = checks.product_match
    parts.append(
        f"Image-vs-comment alignment {a.score:.0%}; "
        f"product match {p.score:.0%} ({p.detected_product or 'unidentified'})."
    )
    if checks.ai_generated.is_ai_generated:
        parts.append(
            f"⚠ Possible AI-generated/edited image "
            f"({checks.ai_generated.ai_probability:.0%} prob, {checks.ai_generated.source})."
        )
    if checks.other_flags:
        parts.append("Flags: " + ", ".join(checks.other_flags) + ".")
    parts.append(
        f"Authenticity {score:.0%} → {verdict.replace('_', ' ')}; "
        f"recommend {action.replace('_', ' ')}."
    )
    return " ".join(parts)


def _recognition(gemini: dict) -> RecognitionResult:
    raw = gemini.get("recognition", {}) or {}
    objects = [str(o) for o in (raw.get("objects") or []) if str(o).strip()]
    text = raw.get("extracted_text")
    text = str(text).strip() if text else None
    return RecognitionResult(
        scene=raw.get("scene") or None,
        objects=objects,
        extracted_text=text or None,
    )


def build_verify_response(
    gemini: dict,
    ai_check: AIGeneratedCheck,
    order_id: str,
    user_id: str,
    dedup_result: Optional[DedupResult] = None,
    web_result: Optional[WebProvenanceResult] = None,
) -> VerifyClaimResponse:
    score, verdict, action, checks = score_claim(gemini, ai_check, dedup_result, web_result)
    ai_confident = (
        ai_check.is_ai_generated
        and ai_check.ai_probability >= settings.ai_detection_min_confidence
    )
    hard_signal = bool(dedup_result and dedup_result.is_cross_claim_duplicate) or _web_hard(web_result)
    conf = 1.0 if hard_signal else decision_confidence(score, action, ai_check, ai_confident)
    return VerifyClaimResponse(
        success=True,
        request_id=generate_request_id(),
        order_id=order_id,
        user_id=user_id,
        authenticity_score=score,
        score_out_of_100=round(score * 100),
        decision_confidence=conf,
        verdict=verdict,  # type: ignore[arg-type]
        recommended_action=action,  # type: ignore[arg-type]
        recognition=_recognition(gemini),
        checks=checks,
        agent_comment=_agent_comment(gemini, checks, score, verdict, action),
        processed_at=datetime.now(timezone.utc),
        model_used=settings.gemini_model,
    )
