from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder sentinels — these are dev-only defaults and must never be the
# effective value in production.
_PLACEHOLDERS = {
    "missing-google-api-key",
    "missing-project-id",
    "missing-kaily-secret",
    "",
}


class Settings(BaseSettings):
    """All configuration via environment variables — zero hardcoded values."""

    google_api_key: str = "missing-google-api-key"
    vertex_project_id: str = "missing-project-id"
    vertex_region: str = "us-central1"
    # Inline service-account key JSON for Vertex auth on serverless runtimes that
    # can't mount a key file (e.g. Boltic). When set and USE_VERTEX is true, the
    # Gemini client authenticates with these credentials. Empty -> fall back to
    # Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS / metadata).
    google_credentials_json: str = ""
    kaily_api_secret: str = "missing-kaily-secret"
    # When True, route Gemini calls through Vertex AI (bills the GCP project /
    # uses its credits, auth via Application Default Credentials). When False,
    # use the AI Studio API key (google_api_key) and its prepay billing.
    use_vertex: bool = False
    gemini_vertex_fallback_enabled: bool = True
    provider_error_details_enabled: bool = False
    gemini_model: str = "gemini-2.0-flash-001"
    gemini_timeout_seconds: int = 45
    max_image_size_mb: int = 10
    environment: str = "development"
    log_level: str = "INFO"
    port: int = 8000

    # ---- Claim authenticity verification (/verify-claim) -------------------
    # AI-generated-image detector provider. "internal" = free in-process
    # (EXIF/C2PA metadata + Gemini visual heuristic). "sightengine" = call the
    # paid specialist detector (~$0.01/image) for higher accuracy.
    ai_detector_provider: str = "internal"
    sightengine_api_user: str = ""
    sightengine_api_secret: str = ""

    # Deterministic scoring weights (alignment + product match form the base
    # score; must sum to 1.0). The authenticity score is computed in code, not
    # by the model, so eligibility is auditable.
    authenticity_weight_alignment: float = 0.5
    authenticity_weight_product_match: float = 0.5
    # Penalty applied to the score when the image looks AI-generated (scaled by
    # detector confidence). Each additional fraud flag subtracts this much.
    authenticity_ai_penalty: float = 0.6
    authenticity_flag_penalty: float = 0.1
    # Min detector confidence before an AI-generated result influences routing.
    ai_detection_min_confidence: float = 0.6
    # Verdict thresholds on the final 0..1 authenticity score.
    authenticity_auto_approve_threshold: float = 0.75
    authenticity_review_threshold: float = 0.45

    # ---- Web reverse-image-search (req 1b): "website-downloaded" detection ---
    # Google Cloud Vision WEB_DETECTION. Off -> signal skipped (checked=False).
    web_provenance_enabled: bool = True
    # Full matches across at least this many DISTINCT domains -> hard fraud signal
    # (auto-reject, like a cross-claim duplicate). Raise high to disable hard-reject.
    web_match_hard_min_domains: int = 2
    # Soft, proportional score penalty per web match below the hard threshold.
    web_match_soft_penalty: float = 0.15
    # Max matches counted toward the soft penalty.
    web_match_penalty_cap: int = 3
    # Hard timeout (seconds) for the Vision call.
    vision_timeout_seconds: int = 8

    # ---- Phase 1: durable audit store + object storage --------------------
    # Async Postgres DSN (postgresql+asyncpg://...). Empty -> in-memory audit
    # repository (dev only; logs a loud warning). Production REQUIRES a real DSN.
    database_url: str = ""
    # Object store for uploaded images. "memory" (dev/test, no side effects),
    # "local" (filesystem under object_store_local_dir), "gcs" (production).
    object_store_provider: str = "memory"
    object_store_local_dir: str = "/tmp/imgrecog-objects"
    gcs_bucket: str = ""
    # India data residency for image storage (DPDP Act 2023; enforced in Phase 6).
    gcs_region: str = "asia-south1"
    # Retention hook — actual deletion job is wired in Phase 6. Placeholder pending
    # legal sign-off (see README open question).
    image_retention_days: int = 90
    # Bumped whenever the analysis prompts change, recorded on every audit row so a
    # decision is reproducible against the exact prompt that produced it.
    scan_prompt_version: str = "scan-v1"
    verify_prompt_version: str = "verify-v1"

    # ---- Phase 2: reused-image dedup (Redis-backed pHash index) -----------
    # Redis DSN (redis://... / rediss://...). Empty -> in-memory dedup index
    # (DEV ONLY; production refuses to boot without it — dedup is a fraud control).
    redis_url: str = ""
    # Max Hamming distance (on the 64-bit dHash) to treat two images as the same.
    # 0 = byte-identical only; higher tolerates re-compression/cropping. The band
    # index guarantees recall up to distance 15.
    dedup_hamming_threshold: int = 10
    # Only consider prior claims within this many days a duplicate (the "window").
    dedup_window_days: int = 30

    # ---- URL-based image evaluation (/evaluate-links) -----------------------
    # The public API accepts image links, but the existing engine works on bytes.
    # These controls make URL ingestion bounded and SSRF-resistant.
    url_fetch_timeout_seconds: float = 10.0
    url_fetch_max_redirects: int = 3
    url_fetch_processing_retries: int = 2
    url_fetch_processing_retry_delay_seconds: float = 0.5
    url_fetch_user_agent: str = "Kaily-ImgRec/1.0"
    link_eval_model_max_edge_px: int = 1280
    link_eval_model_image_quality: int = 85
    link_decision_min_authenticity_score: int = 70
    link_decision_min_product_match_score: int = 75
    link_decision_min_status_score: int = 60
    link_decision_min_query_match_score: int = 60

    # PixelBin is a CDN/preprocessing layer, not a detector. Configure a template
    # when you want every safe source URL normalized through PixelBin before model
    # analysis. Supported placeholders:
    #   {url} {url_encoded} {host} {path} {path_encoded}
    # Example:
    #   https://cdn.pixelbin.io/v2/<cloud>/<zone>/wrkr/t.resize(w:1280)/{path}
    pixelbin_enabled: bool = False
    pixelbin_url_template: str = ""
    pixelbin_allow_direct_fallback: bool = True

    # ---- Grocery dispute verification (/dispute) ---------------------------
    # Approved refund >= this (INR) routes to a human agent with the AI
    # recommendation attached, even when the category decision is "approve".
    refund_auto_approve_max: float = 500
    # When true, every dispute decision becomes recommend-only (route=agent) —
    # a shadow/assist period without a redeploy.
    dispute_assist_mode: bool = False
    # Comma-separated categories allowed to auto-act. Others are recommend-only
    # until they clear the accuracy bar (progressive rollout).
    dispute_autonomous_categories: str = "mrp_abuse,expiry,wrong_product,damaged"
    # Dairy: approve near-expiry when remaining shelf life is below this percent.
    dairy_min_shelf_pct: float = 30
    # Non-FNV: approve when days until expiry is at or below this (Legal Metrology).
    non_fnv_near_expiry_days: int = 45
    # Max customer images accepted per dispute (bounded input).
    dispute_max_images: int = 5
    # Per-instance concurrency cap on Gemini calls (quota/backpressure).
    gemini_max_concurrency: int = 8
    dispute_prompt_version: str = "dispute-v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _require_real_secrets_in_production(self):
        """Fail fast: refuse to boot in production with placeholder/missing secrets.

        Dev/test keep working with defaults; only ENVIRONMENT=production is gated,
        so a misconfigured prod service crashes loudly instead of silently serving.
        """
        if self.environment.strip().lower() != "production":
            return self

        missing = []
        if self.kaily_api_secret in _PLACEHOLDERS:
            missing.append("KAILY_API_SECRET")
        if self.use_vertex:
            if self.vertex_project_id in _PLACEHOLDERS:
                missing.append("VERTEX_PROJECT_ID")
        elif self.google_api_key in _PLACEHOLDERS:
            missing.append("GOOGLE_API_KEY")
        if self.ai_detector_provider == "sightengine" and (
            not self.sightengine_api_user or not self.sightengine_api_secret
        ):
            missing.append("SIGHTENGINE_API_USER/SIGHTENGINE_API_SECRET")
        # A money-affecting service must have a durable audit trail in prod —
        # refuse to boot on the in-memory fallback (no record = silent payouts).
        if not self.database_url:
            missing.append("DATABASE_URL")
        if self.object_store_provider == "memory":
            missing.append("OBJECT_STORE_PROVIDER (memory is dev-only)")
        elif self.object_store_provider == "gcs" and not self.gcs_bucket:
            missing.append("GCS_BUCKET")
        # Reused-image dedup is a fraud control — it must be durable/shared in prod.
        if not self.redis_url:
            missing.append("REDIS_URL")

        if missing:
            raise ValueError(
                "Refusing to start in production — missing required config: "
                + ", ".join(missing)
            )
        return self


settings = Settings()
