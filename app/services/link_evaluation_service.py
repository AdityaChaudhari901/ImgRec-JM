from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from typing import Any

from google.genai import types

from app.config.settings import settings
from app.models.link_evaluation import (
    BusinessDecision,
    EvaluationCheck,
    LinkedImageEvaluationResponse,
    ProductStatusCheck,
)
from app.models.verify_response import AIGeneratedCheck
from app.services.gemini_service import get_client
from app.services.image_url_fetcher import FetchedImage
from app.services.web_provenance import WebProvenanceResult
from app.utils.date_utils import is_expired, parse_indian_date
from app.utils.logger import get_logger

logger = get_logger(__name__)

_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)
_DATE_TOKEN = (
    rf"(?:\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|"
    rf"\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}|"
    rf"\d{{1,2}}\s+{_MONTH_NAME}\s+\d{{2,4}}|"
    rf"{_MONTH_NAME}\s+\d{{1,2}},?\s+\d{{2,4}}|"
    rf"{_MONTH_NAME}[-/\s]+\d{{2,4}})"
)
_EXPIRY_DATE_RE = re.compile(
    rf"\b(?:exp(?:iry|iration)?|expires?|use\s*by|best\s*before|bb|b/b)\b"
    rf"(?:\s*(?:date|dt|on|is|by|:|-|\.))*\s*"
    rf"(?P<date>{_DATE_TOKEN})",
    re.IGNORECASE,
)

LINK_EVALUATION_PROMPT = """
You are a high-precision product evidence analyst for a retail customer-support
image recognition engine.

INPUTS:
- Image 1: customer/user submitted evidence image.
- Image 2: official product/catalog image.
- User query: "{query}"
- Today's date: {today}

Return observations only. The server computes final policy decisions.

AUTHENTICITY:
Estimate whether Image 1 is a real mobile-clicked customer photo, not
AI-generated, not a screenshot, and not downloaded from the web. Use visual
evidence only: camera perspective, natural lighting, background/context,
screen borders, watermarks, stock-photo styling, synthetic artifacts, warped
text, impossible packaging geometry, and editing artifacts. Do not over-trust
metadata, because CDNs and messaging apps often strip it.

PRODUCT MATCH ACCURACY:
Compare Image 1 against Image 2 very carefully. Product accuracy is critical.
Use brand/logo, packaging shape, color blocks, label layout, visible text/OCR,
SKU/variant/flavor/size, cap/seal style, and product category. Ignore normal
customer-photo changes such as lighting, crop, rotation, damage, glare, and
partial occlusion. Penalize a different variant, different size, different
flavor, private-label substitution, or same category but different SKU.

Product score rubric:
- 0.95-1.00: exact same SKU/package or overwhelming visual/OCR evidence.
- 0.85-0.94: same product with minor visibility/lighting uncertainty.
- 0.65-0.84: plausible match, but variant/SKU evidence is incomplete.
- 0.35-0.64: same broad category, weak or conflicting product evidence.
- 0.00-0.34: different product/category or not enough product visible.

PRODUCT STATUS AUTO-CHECK:
Independently classify Image 1 into exactly one product status:
- "expired": readable expiry/use-by/best-before date is before today's date, or
  there is strong visible spoilage evidence.
- "damaged": visible physical damage exists: torn, crushed, leaking, broken
  seal, dented, moldy, stained, spoiled, or packaging/product deformation.
- "valid": product is visible, product matches the reference, no visible damage
  is present, and no expiry/spoilage evidence contradicts validity. If expiry
  date is hidden, valid can still be selected only with moderate confidence.
- "unclear": evidence is missing, unreadable, conflicting, or product is not
  sufficiently visible.
This status must be based on Image 1 evidence, not on the user query wording.

QUERY MATCH:
Decide whether the user query is supported by Image 1.
- "expired product": high only when a readable expiry/use-by/best-before date
  is before today's date, or there is strong visible spoilage evidence. If no
  date is visible, score conservatively.
- "damaged product": high only when visible physical damage exists: torn,
  crushed, leaking, broken seal, dented, moldy, stained, or spoiled.
- "valid product": high only when product is visible, no damage is visible, and
  no expiry/spoilage evidence contradicts validity. If the expiry date is hidden,
  do not give a perfect score.
- For any other query, match the literal visual evidence in Image 1.

RESPOND ONLY WITH THIS EXACT JSON SHAPE, NO MARKDOWN:
{
  "recognition": {
    "user_image_scene": "one-line description",
    "user_image_text": "all visible text in image 1, or empty string",
    "product_image_text": "all visible text in image 2, or empty string"
  },
  "product_status": {
    "status": "expired | damaged | valid | unclear",
    "score": 0.0,
    "verdict": "one sentence",
    "detail": "one sentence",
    "evidence": ["short evidence", "..."]
  },
  "authenticity": {
    "mobile_capture_score": 0.0,
    "ai_generated": {
      "ai_probability": 0.0,
      "signals": ["short signal", "..."]
    },
    "verdict": "one sentence",
    "detail": "one sentence"
  },
  "product_match": {
    "score": 0.0,
    "matches": true,
    "detected_product": "what image 1 appears to show",
    "reference_product": "what image 2 appears to show",
    "verdict": "one sentence",
    "detail": "one sentence",
    "match_evidence": ["short evidence", "..."],
    "mismatch_evidence": ["short evidence", "..."]
  },
  "query_match": {
    "score": 0.0,
    "matches": true,
    "query_type": "expired | damaged | valid | other",
    "verdict": "one sentence",
    "detail": "one sentence",
    "evidence": ["short evidence", "..."]
  },
  "quality_flags": ["short flag", "..."]
}
"""

_LINK_EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recognition": {
            "type": "object",
            "properties": {
                "user_image_scene": {"type": "string"},
                "user_image_text": {"type": "string"},
                "product_image_text": {"type": "string"},
            },
        },
        "product_status": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "score": {"type": "number"},
                "verdict": {"type": "string"},
                "detail": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
        },
        "authenticity": {
            "type": "object",
            "properties": {
                "mobile_capture_score": {"type": "number"},
                "ai_generated": {
                    "type": "object",
                    "properties": {
                        "ai_probability": {"type": "number"},
                        "signals": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "verdict": {"type": "string"},
                "detail": {"type": "string"},
            },
        },
        "product_match": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "matches": {"type": "boolean"},
                "detected_product": {"type": "string"},
                "reference_product": {"type": "string"},
                "verdict": {"type": "string"},
                "detail": {"type": "string"},
                "match_evidence": {"type": "array", "items": {"type": "string"}},
                "mismatch_evidence": {"type": "array", "items": {"type": "string"}},
            },
        },
        "query_match": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "matches": {"type": "boolean"},
                "query_type": {"type": "string"},
                "verdict": {"type": "string"},
                "detail": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
        },
        "quality_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product_status", "authenticity", "product_match", "query_match"],
}


async def analyze_linked_images(
    user_image: FetchedImage,
    product_image: FetchedImage,
    query: str,
) -> dict:
    """Call Gemini with user + product images and return structured observations."""
    client = get_client()
    config = _generation_config()
    prompt = (
        LINK_EVALUATION_PROMPT
        .replace("{query}", query.strip().replace('"', "'"))
        .replace("{today}", date.today().isoformat())
    )
    contents = [
        types.Part.from_bytes(
            data=user_image.model_bytes,
            mime_type=user_image.model_content_type,
        ),
        types.Part.from_bytes(
            data=product_image.model_bytes,
            mime_type=product_image.model_content_type,
        ),
        prompt,
    ]

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=config,
            ),
            timeout=settings.gemini_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Gemini API timed out after {settings.gemini_timeout_seconds}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("link_evaluation_gemini_failed", error=str(exc))
        raise

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("link_evaluation_bad_json", error=str(exc), raw=text[:500])
        raise ValueError("Gemini returned malformed JSON") from exc


def build_link_evaluation_response(
    observations: dict,
    ai_check: AIGeneratedCheck,
    web_result: WebProvenanceResult,
    query: str = "",
    today: date | None = None,
) -> LinkedImageEvaluationResponse:
    authenticity_raw = observations.get("authenticity", {}) or {}
    status_raw = observations.get("product_status", {}) or {}
    product_raw = observations.get("product_match", {}) or {}
    query_raw = observations.get("query_match", {}) or {}
    today_value = today or date.today()
    labeled_expiry_date = _extract_labeled_expiry_date(observations)
    product_status = _product_status_check(status_raw)
    product_status = _apply_expiry_status_override(
        product_status,
        expiry_iso=labeled_expiry_date,
        today=today_value,
    )
    authenticity = _authenticity_check(authenticity_raw, ai_check, web_result)
    product_match = _model_check(
        product_raw,
        high="User image strongly matches the product image.",
        mid="User image is a plausible product match with incomplete SKU evidence.",
        low="User image is only a weak product match.",
        fail="User image does not match the product image.",
    )
    query_match = _model_check(
        query_raw,
        high="User image supports the user query.",
        mid="User image partially supports the user query.",
        low="User image weakly supports the user query.",
        fail="User image does not clearly support the user query.",
    )
    query_match = _apply_query_expiry_override(
        query_match,
        query=query,
        query_raw=query_raw,
        expiry_iso=labeled_expiry_date,
        today=today_value,
    )

    return LinkedImageEvaluationResponse(
        decision=_business_decision(
            product_status=product_status,
            authenticity=authenticity,
            product_match=product_match,
            query_match=query_match,
        ),
        product_status=product_status,
        authenticity=authenticity,
        product_match=product_match,
        query_match=query_match,
    )


def _business_decision(
    *,
    product_status: ProductStatusCheck,
    authenticity: EvaluationCheck,
    product_match: EvaluationCheck,
    query_match: EvaluationCheck,
) -> BusinessDecision:
    reasons: list[str] = []

    if authenticity.score < settings.link_decision_min_authenticity_score:
        reasons.append("authenticity_below_threshold")
        return BusinessDecision(
            decision="review",
            verdict="Review required before trusting this image.",
            detail=(
                f"Authenticity score {authenticity.score}% is below the "
                f"{settings.link_decision_min_authenticity_score}% gate."
            ),
            reason_codes=reasons,
        )

    if product_match.score < settings.link_decision_min_product_match_score:
        reasons.append("product_match_below_threshold")
        return BusinessDecision(
            decision="review",
            verdict="Review required because the image may not be the same product.",
            detail=(
                f"Product match score {product_match.score}% is below the "
                f"{settings.link_decision_min_product_match_score}% gate."
            ),
            reason_codes=reasons,
        )

    if (
        product_status.status == "unclear"
        or product_status.score < settings.link_decision_min_status_score
    ):
        reasons.append("product_status_unclear")
        return BusinessDecision(
            decision="review",
            verdict="Review required because product condition is unclear.",
            detail=(
                f"Detected status is {product_status.status} with "
                f"{product_status.score}% confidence."
            ),
            reason_codes=reasons,
        )

    if product_status.status == "valid":
        reasons.append("product_valid")
        return BusinessDecision(
            decision="reject_claim",
            verdict="Reject claim because the product appears valid.",
            detail=(
                "The image passed authenticity and product-match gates, and "
                "the automatic condition check did not find expiry or damage."
            ),
            reason_codes=reasons,
        )

    if query_match.score < settings.link_decision_min_query_match_score:
        reasons.append("query_match_below_threshold")
        return BusinessDecision(
            decision="review",
            verdict="Review required because the query does not match the visible issue.",
            detail=(
                f"Product appears {product_status.status}, but query match score "
                f"{query_match.score}% is below the "
                f"{settings.link_decision_min_query_match_score}% gate."
            ),
            reason_codes=reasons,
        )

    reasons.append(f"{product_status.status}_claim_supported")
    return BusinessDecision(
        decision="accept_claim",
        verdict=f"Accept claim because the product appears {product_status.status}.",
        detail=(
            "Authenticity, product match, product status, and query-match gates "
            "all passed."
        ),
        reason_codes=reasons,
    )


def _product_status_check(raw: dict) -> ProductStatusCheck:
    status = str(raw.get("status") or "unclear").strip().lower()
    if status not in {"expired", "damaged", "valid", "unclear"}:
        status = "unclear"

    score = _score100(_score01(raw.get("score")))
    verdict = _one_line(raw.get("verdict"))
    if not verdict:
        verdict = {
            "expired": "Product appears expired.",
            "damaged": "Product appears damaged.",
            "valid": "Product appears valid.",
            "unclear": "Product status is unclear from the image.",
        }[status]

    evidence = _join_evidence(raw.get("evidence") or [])
    detail = _one_line(raw.get("detail")) or evidence or None
    return ProductStatusCheck(
        status=status, score=score, verdict=verdict, detail=detail
    )


def _apply_expiry_status_override(
    product_status: ProductStatusCheck,
    *,
    expiry_iso: str | None,
    today: date,
) -> ProductStatusCheck:
    if not expiry_iso or not is_expired(expiry_iso, today=today):
        return product_status

    detail = (
        f"Readable expiry date {expiry_iso} is before today's date "
        f"{today.isoformat()}."
    )
    if product_status.detail:
        detail = f"{detail} Model detail: {product_status.detail}"
    return ProductStatusCheck(
        status="expired",
        score=max(product_status.score, 95),
        verdict="Product appears expired.",
        detail=detail,
    )


def _apply_query_expiry_override(
    query_match: EvaluationCheck,
    *,
    query: str,
    query_raw: dict,
    expiry_iso: str | None,
    today: date,
) -> EvaluationCheck:
    if _query_claim_type(query, query_raw) != "expired":
        return query_match
    if not expiry_iso or not is_expired(expiry_iso, today=today):
        return query_match

    detail = (
        f"Readable expiry date {expiry_iso} is before today's date "
        f"{today.isoformat()}."
    )
    return EvaluationCheck(
        score=max(query_match.score, 95),
        verdict="User query is supported by the visible expiry date.",
        detail=detail,
    )


def _query_claim_type(query: str, query_raw: dict) -> str:
    text = f"{query} {query_raw.get('query_type') or ''}".lower()
    if any(word in text for word in ("expired", "expiry", "expiration", "expire")):
        return "expired"
    if any(
        word in text
        for word in (
            "damage",
            "damaged",
            "broken",
            "leak",
            "leaking",
            "torn",
            "crushed",
            "dented",
        )
    ):
        return "damaged"
    if any(word in text for word in ("valid", "ok", "okay", "fine")):
        return "valid"
    return "other"


def _extract_labeled_expiry_date(observations: dict) -> str | None:
    for text in _expiry_text_candidates(observations):
        for match in _EXPIRY_DATE_RE.finditer(text):
            parsed = parse_indian_date(match.group("date"))
            if parsed:
                return parsed
    return None


def _expiry_text_candidates(observations: dict) -> list[str]:
    recognition = observations.get("recognition", {}) or {}
    product_status = observations.get("product_status", {}) or {}
    query_match = observations.get("query_match", {}) or {}
    candidates: list[Any] = [
        recognition.get("user_image_text"),
        recognition.get("user_image_scene"),
        product_status.get("verdict"),
        product_status.get("detail"),
        query_match.get("verdict"),
        query_match.get("detail"),
    ]
    candidates.extend(product_status.get("evidence") or [])
    candidates.extend(query_match.get("evidence") or [])
    candidates.extend(observations.get("quality_flags") or [])
    return [_one_line(item) for item in candidates if _one_line(item)]


def _generation_config() -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {
        "temperature": 0.05,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
        "response_schema": _LINK_EVALUATION_SCHEMA,
    }
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:  # noqa: BLE001
        pass
    return types.GenerateContentConfig(**kwargs)


def _authenticity_check(
    authenticity_raw: dict,
    ai_check: AIGeneratedCheck,
    web_result: WebProvenanceResult,
) -> EvaluationCheck:
    gemini_ai = authenticity_raw.get("ai_generated", {}) or {}
    visual_ai_probability = _score01(gemini_ai.get("ai_probability"))
    ai_probability = max(visual_ai_probability, _score01(ai_check.ai_probability))
    mobile_capture_score = _score01(authenticity_raw.get("mobile_capture_score"), default=0.5)

    web_score = 0.55
    web_detail = "web provenance was not checked"
    web_hard = False
    if web_result.checked:
        total_matches = web_result.full_match_count + web_result.partial_match_count
        web_score = max(0.0, 1.0 - min(total_matches, settings.web_match_penalty_cap) * 0.25)
        web_detail = (
            f"web check found {web_result.full_match_count} full match(es), "
            f"{web_result.partial_match_count} partial match(es), "
            f"{web_result.distinct_domains} domain(s)"
        )
        web_hard = (
            web_result.full_match_count > 0
            and web_result.distinct_domains >= settings.web_match_hard_min_domains
        )

    score = (
        0.50 * (1.0 - ai_probability)
        + 0.30 * mobile_capture_score
        + 0.20 * web_score
    )
    if ai_check.is_ai_generated:
        score -= 0.15 * ai_probability
    if web_hard:
        score = 0.0
    score_out_of_100 = _score100(score)

    if web_hard:
        verdict = "Image is likely web-downloaded or reused from public web results."
    elif ai_check.is_ai_generated and ai_probability >= settings.ai_detection_min_confidence:
        verdict = "Image has strong AI-generated or edited-image signals."
    elif score_out_of_100 >= 80:
        verdict = authenticity_raw.get("verdict") or "Image is likely an original mobile customer photo."
    elif score_out_of_100 >= 55:
        verdict = "Image authenticity is plausible but needs review."
    else:
        verdict = "Image authenticity is weak or suspicious."

    signals = [str(s) for s in (ai_check.signals or []) if str(s).strip()]
    signals.extend(
        str(s) for s in (gemini_ai.get("signals") or []) if str(s).strip()
    )
    detail_parts = [
        _one_line(authenticity_raw.get("detail")),
        web_detail,
        _join_evidence(signals[:3]),
    ]
    detail = "; ".join(p for p in detail_parts if p)
    return EvaluationCheck(score=score_out_of_100, verdict=_one_line(verdict), detail=detail or None)


def _model_check(raw: dict, *, high: str, mid: str, low: str, fail: str) -> EvaluationCheck:
    score = _score100(_score01(raw.get("score")))
    verdict = _one_line(raw.get("verdict"))
    if not verdict:
        if score >= 85:
            verdict = high
        elif score >= 65:
            verdict = mid
        elif score >= 40:
            verdict = low
        else:
            verdict = fail

    detail = _one_line(raw.get("detail"))
    evidence = []
    for key in ("match_evidence", "mismatch_evidence", "evidence"):
        evidence.extend(str(item) for item in (raw.get(key) or []) if str(item).strip())
    if evidence:
        detail = f"{detail}; {_join_evidence(evidence[:4])}" if detail else _join_evidence(evidence[:4])
    return EvaluationCheck(score=score, verdict=verdict, detail=detail or None)


def _score01(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _score100(value: float) -> int:
    return max(0, min(100, round(value * 100)))


def _one_line(value: Any) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return " ".join(text.split())


def _join_evidence(items: list[str]) -> str:
    clean = [_one_line(item) for item in items if _one_line(item)]
    return "Evidence: " + "; ".join(clean) if clean else ""
