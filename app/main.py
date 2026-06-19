import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config.settings import settings
from app.middleware.error_handler import register_exception_handlers
from app.middleware.rate_limit import limiter
from app.routers.scan import router
from app.routers.verify import router as verify_router
from app.services.gemini_service import get_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", model=settings.gemini_model, env=settings.environment)
    print(f"[ImgRec] Server running on port {settings.port}")
    print(f"[ImgRec] Model: {settings.gemini_model}")
    print(f"[ImgRec] Environment: {settings.environment}")
    print("[ImgRec] Kaily endpoint: POST /api/v1/imgrecog/scan")
    print("[ImgRec] Claim verify:   POST /api/v1/imgrecog/verify-claim")
    print("[ImgRec] Health check: GET /health")
    print("[ImgRec] Readiness:    GET /ready")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Kaily ImgRec API",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting (slowapi) — register limiter + 429 handler.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Bind a correlation id to every log line in the request and echo it back.

    Honours an inbound `X-Request-ID` (so a caller/Kaily can propagate its own
    trace id) or generates one. structlog's merge_contextvars processor then
    stamps it onto every log emitted while handling the request.
    """
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id, method=request.method, path=request.url.path
    )
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["X-Request-ID"] = request_id
    return response


register_exception_handlers(app)
app.include_router(router)
app.include_router(verify_router)


@app.get("/health")
async def health():
    """Liveness — the process is up. Cheap, no dependencies; safe to poll often."""
    return {"status": "ok", "model": settings.gemini_model}


@app.get("/ready")
async def ready():
    """Readiness — can we serve traffic right now? Gate the LB on this.

    Validates required config is present and the Gemini client initialises. It
    deliberately does NOT make a billed model call; for a deep check, probe
    list_models out-of-band.
    """
    checks: dict[str, bool] = {}

    if settings.use_vertex:
        checks["config"] = settings.vertex_project_id not in {"", "missing-project-id"}
    else:
        checks["config"] = settings.google_api_key not in {"", "missing-google-api-key"}

    try:
        get_client()
        checks["gemini_client"] = True
    except Exception as exc:  # noqa: BLE001
        logger.error("readiness_client_init_failed", error=str(exc))
        checks["gemini_client"] = False

    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "checks": checks},
    )


_DEMO_INDEX = Path(__file__).resolve().parent.parent / "demo" / "index.html"


@app.get("/", include_in_schema=False)
async def demo_ui():
    """Serve the bundled demo UI so the deployment has a working frontend."""
    if _DEMO_INDEX.is_file():
        return FileResponse(_DEMO_INDEX)
    return JSONResponse({"service": "Kaily ImgRec API", "docs": "/docs"})


# Boltic serverless compatibility — export the ASGI app.
handler = app
