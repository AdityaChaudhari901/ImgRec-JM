import time

from fastapi import APIRouter, Depends, HTTPException, Request

try:  # google-genai SDK error types (present in prod; guarded for safety)
    from google.genai import errors as genai_errors

    _UPSTREAM_API_ERROR = (genai_errors.APIError,)
except Exception:  # noqa: BLE001
    genai_errors = None
    _UPSTREAM_API_ERROR = ()

from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.models.request import ScanRequest
from app.models.response import ScanResponse
from app.services.ai_image_detector import detect_ai_generated
from app.services.audit_service import build_idempotency_key, find_replay, persist_decision
from app.services.decision_engine import build_response
from app.services.gemini_service import analyze_image
from app.utils.image_utils import compute_image_phash, validate_image_size
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/api/v1/imgrecog/scan",
    response_model=ScanResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("100/minute")
async def scan_product(request: Request, body: ScanRequest) -> ScanResponse:
    start = time.time()

    # 413 — payload too large
    try:
        validate_image_size(body.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    # Idempotency — a repeat claim returns the prior decision verbatim, with no
    # second model call and no second action.
    image_phash = compute_image_phash(body.image_base64)
    idempotency_key = build_idempotency_key(
        body.idempotency_key, body.claim_id, body.order_id, body.user_id,
        image_phash, body.image_base64,
    )
    replay = await find_replay("scan", idempotency_key)
    if replay is not None:
        logger.info(
            "idempotent_replay",
            endpoint="scan",
            order_id=body.order_id,
            request_id=replay.request_id,
        )
        return replay

    # Gemini analysis — map upstream conditions to honest status codes.
    try:
        gemini_output = await analyze_image(body.image_base64, body.scan_type)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Image analysis timed out")
    except HTTPException:
        raise
    except _UPSTREAM_API_ERROR as exc:  # type: ignore[misc]
        code = getattr(exc, "code", None)
        if code in (429, 503):
            # Upstream quota / billing / rate limit / overload.
            logger.error("gemini_quota_exhausted", code=code, order_id=body.order_id)
            raise HTTPException(
                status_code=503,
                detail="Image analysis temporarily unavailable (upstream quota/overload)",
            )
        logger.error("gemini_upstream_error", code=code, error=str(exc), order_id=body.order_id)
        raise HTTPException(status_code=502, detail="Upstream image analysis error")
    except Exception as exc:  # noqa: BLE001
        logger.error("scan_failed", error=str(exc), order_id=body.order_id)
        raise HTTPException(status_code=500, detail="Analysis failed")

    ai_check = await detect_ai_generated(
        body.image_base64, gemini_output.get("ai_generated", {})
    )
    response = build_response(gemini_output, body.order_id, body.user_id, ai_check)

    latency_ms = round((time.time() - start) * 1000)
    response = await persist_decision(
        endpoint="scan",
        idempotency_key=idempotency_key,
        correlation_id=request.headers.get("x-request-id"),
        order_id=body.order_id,
        user_id=body.user_id,
        image_base64=body.image_base64,
        image_phash=image_phash,
        response=response,
        latency_ms=latency_ms,
        raw_observations=gemini_output,
        computed={"days_since_expiry": response.ocr.days_since_expiry},
    )

    logger.info(
        "scan_complete",
        request_id=response.request_id,
        order_id=body.order_id,
        status=response.status,
        action=response.action.type,
        ai_flagged=ai_check.is_ai_generated,
        latency_ms=latency_ms,
    )

    return response
