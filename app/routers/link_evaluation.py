import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Request

try:  # google-genai SDK error types (present in prod; guarded for safety)
    from google.genai import errors as genai_errors

    _UPSTREAM_API_ERROR = (genai_errors.APIError,)
except Exception:  # noqa: BLE001
    _UPSTREAM_API_ERROR = ()

from app.config.settings import settings
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.models.link_evaluation import (
    LinkedImageEvaluationRequest,
    LinkedImageEvaluationResponse,
)
from app.services.ai_image_detector import detect_ai_generated
from app.services.image_url_fetcher import ImageUrlError, download_image_url
from app.services.link_evaluation_service import (
    analyze_linked_images,
    build_link_evaluation_response,
)
from app.services.web_provenance import WebProvenanceResult, detect_web_provenance
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/api/v1/imgrecog/evaluate-links",
    response_model=LinkedImageEvaluationResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("60/minute")
async def evaluate_links(
    request: Request,
    body: LinkedImageEvaluationRequest,
) -> LinkedImageEvaluationResponse:
    start = time.time()

    try:
        user_image, product_image = await _download_inputs(body)
    except ImageUrlError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    web_task = (
        asyncio.create_task(detect_web_provenance(user_image.data_uri))
        if settings.web_provenance_enabled
        else None
    )
    ai_task = (
        asyncio.create_task(detect_ai_generated(user_image.data_uri, None))
        if settings.ai_detector_provider == "sightengine"
        else None
    )

    try:
        observations = await analyze_linked_images(user_image, product_image, body.query)
    except TimeoutError:
        _cancel_pending(web_task, ai_task)
        raise HTTPException(status_code=504, detail="Image analysis timed out")
    except HTTPException:
        _cancel_pending(web_task, ai_task)
        raise
    except _UPSTREAM_API_ERROR as exc:  # type: ignore[misc]
        _cancel_pending(web_task, ai_task)
        code = getattr(exc, "code", None)
        if code in (429, 503):
            logger.error("link_evaluation_quota_exhausted", code=code)
            raise HTTPException(
                status_code=503,
                detail="Image analysis temporarily unavailable (upstream quota/overload)",
            )
        logger.error("link_evaluation_upstream_error", code=code, error=str(exc))
        raise HTTPException(status_code=502, detail="Upstream image analysis error")
    except Exception as exc:  # noqa: BLE001
        _cancel_pending(web_task, ai_task)
        logger.error("link_evaluation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Image link evaluation failed")

    try:
        if ai_task is not None:
            ai_check = await ai_task
        else:
            gemini_ai = (observations.get("authenticity", {}) or {}).get(
                "ai_generated",
                {},
            )
            ai_check = await detect_ai_generated(user_image.data_uri, gemini_ai)
    except Exception as exc:  # noqa: BLE001
        if web_task is not None and not web_task.done():
            web_task.cancel()
        logger.error("link_evaluation_ai_signal_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="AI signal resolution failed")

    if web_task is not None:
        try:
            web_result = await web_task
        except Exception:  # noqa: BLE001
            logger.warning("link_evaluation_web_signal_failed")
            web_result = WebProvenanceResult(checked=False)
    else:
        web_result = WebProvenanceResult(checked=False)

    response = build_link_evaluation_response(
        observations,
        ai_check,
        web_result,
        query=body.query,
    )
    latency_ms = round((time.time() - start) * 1000)
    logger.info(
        "link_evaluation_complete",
        status_code=200,
        latency_ms=latency_ms,
        user_image_bytes=user_image.size_bytes,
        product_image_bytes=product_image.size_bytes,
        user_model_image_bytes=user_image.model_size_bytes,
        product_model_image_bytes=product_image.model_size_bytes,
        decision=response.decision.decision,
        decision_reasons=response.decision.reason_codes,
        product_status=response.product_status.status,
        product_status_score=response.product_status.score,
        authenticity=response.authenticity.score,
        product_match=response.product_match.score,
        query_match=response.query_match.score,
        correlation_id=request.headers.get("x-request-id"),
    )
    return response


async def _download_inputs(body: LinkedImageEvaluationRequest):
    user_task = asyncio.create_task(download_image_url(body.user_image_url, "user image"))
    product_task = asyncio.create_task(
        download_image_url(body.product_image_url, "product image")
    )
    try:
        return await asyncio.gather(user_task, product_task)
    except BaseException:
        _cancel_pending(user_task, product_task)
        raise


def _cancel_pending(*tasks) -> None:
    for task in tasks:
        if task is not None and not task.done():
            task.cancel()
