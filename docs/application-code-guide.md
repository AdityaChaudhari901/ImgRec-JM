# Kaily ImgRec Application Code Guide

> ⚠️ **Partially superseded (2026-06-25).** The service was consolidated to a
> **single endpoint**, `POST /api/v1/imgrecog/dispute`. The `/scan`,
> `/verify-claim`, and `/evaluate-links` endpoints — and their engines
> (`decision_engine`, `authenticity_engine`, `link_evaluation_service`,
> `ai_image_detector`, `web_provenance`) — were **removed**. Sections below that
> describe those endpoints are historical. For the current system, see the
> **README** (§9, the dispute endpoint) and the **"Request Flow 3: Grocery Dispute
> Verification"** section in this guide, which is current.

This guide explains the application from the live source code. Read it when you
want to understand how the API starts, how each endpoint works, which files own
which behavior, and where to make changes safely.

## What This App Does

Kaily ImgRec is a FastAPI service for JioMart support workflows. It receives
customer product images and returns structured decisions for Kaily.

The app has two main workflows:

1. Product inspection:
   `POST /api/v1/imgrecog/scan`

   This endpoint checks whether a product photo shows an expired, damaged,
   valid, or unclear product. Gemini extracts observations. Python code computes
   the final refund, exchange, or no-action decision.

2. Claim authenticity verification:
   `POST /api/v1/imgrecog/verify-claim`

   This endpoint checks whether a customer's claim photo looks genuine. Gemini
   extracts observations. Python code combines those observations with AI-image,
   duplicate-image, and web-provenance signals to route the claim to
   auto-approve, manual review, or reject.

The most important design choice in this codebase:

The model observes. The application decides.

Gemini reads images, OCR text, damage evidence, and claim signals. The Python
services compute money-affecting decisions with deterministic rules, audit
records, idempotency, and safety downgrades.

## How To Run It Locally

From the project root:

```bash
cd /Users/adityachaudhari/Desktop/ImgRec/imgrecog-kaily
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Useful URLs:

- API root and demo UI: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Readiness: `http://127.0.0.1:8000/ready`

Prefer opening the demo through `http://127.0.0.1:8000/`. If you open
`demo/index.html` directly as a file, the JavaScript falls back to
`http://localhost:8000`, which matches the normal README command on port
`8000`.

## High-Level Architecture

```text
Browser/demo or Kaily
        |
        v
FastAPI app in app/main.py
        |
        +--> auth, rate limit, request id, error handlers
        |
        +--> app/routers/scan.py
        |       +--> Gemini image observations
        |       +--> AI-image evidence check
        |       +--> deterministic scan decision
        |       +--> idempotency + audit persistence
        |
        +--> app/routers/verify.py
                +--> Gemini claim observations
                +--> AI-image evidence check
                +--> duplicate-image check
                +--> web reverse-image-search signal
                +--> deterministic authenticity score
                +--> idempotency + audit persistence
```

Main package responsibilities:

| Folder | Responsibility |
| --- | --- |
| `app/routers` | HTTP endpoint orchestration. Validates request flow, calls services, maps failures to API status codes. |
| `app/services` | Core business logic: Gemini calls, decision rules, claim scoring, AI-image detection, dedup, audit persistence. |
| `app/models` | Pydantic request and response contracts. |
| `app/middleware` | API key auth, rate limiting, error responses. |
| `app/db` | Audit repository abstraction plus SQLAlchemy/Postgres implementation. |
| `app/storage` | Object storage abstraction for uploaded images. |
| `app/utils` | Shared helpers for images, dates, metadata, and logging. |
| `demo` | Static browser UI for trying scan and verify flows. |
| `tests` | Unit and endpoint tests. Gemini and external services are mocked. |

## Request Flow 1: Product Scan

Entrypoint: `app/routers/scan.py`

Endpoint:

```http
POST /api/v1/imgrecog/scan
x-api-key: <KAILY_API_SECRET>
Content-Type: application/json
```

Request model: `app/models/request.py`

```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "order_id": "JM-001",
  "user_id": "u_123",
  "scan_type": "auto",
  "idempotency_key": "optional",
  "claim_id": "optional"
}
```

Execution steps:

1. FastAPI authenticates the request through `verify_api_key`.
2. SlowAPI applies the `100/minute` rate limit.
3. `validate_image_size` rejects images larger than `MAX_IMAGE_SIZE_MB`.
4. `compute_image_phash` computes a best-effort 64-bit image hash.
5. `build_idempotency_key` derives or accepts a replay key.
6. `find_replay("scan", key)` returns the previous decision if this request was already handled.
7. `analyze_image` sends the image to Gemini.
8. `detect_ai_generated` combines metadata and Gemini's AI-image hint.
9. `build_response` computes the final status and action.
10. `persist_decision` stores the image reference and audit row.
11. The router logs `scan_complete` and returns a `ScanResponse`.

Scan decision matrix:

| Status / severity | Final action |
| --- | --- |
| `expired` | `initiate_refund`, high priority |
| `damaged` + `severe` | `initiate_refund`, high priority |
| `damaged` + `moderate` | `initiate_exchange`, medium priority |
| `damaged` + `minor` or unspecified | `initiate_exchange`, low priority |
| `valid` | `no_action`, low priority |
| `unclear` | `no_action`, low priority |

Safety guards in `app/services/decision_engine.py`:

- If image evidence looks AI-generated above the configured confidence threshold,
  the app changes the status to `unclear`, ignores damage, and routes to manual
  review through a high-priority `no_action`.
- If a damaged or expired result does not include an authenticity assessment,
  the app routes to manual review.
- If the model says `valid` or `unclear` but includes severe/moderate damage or
  refund-like action text, the app treats the model output as inconsistent and
  routes to manual review.

## Request Flow 2: Claim Verification

Entrypoint: `app/routers/verify.py`

Endpoint:

```http
POST /api/v1/imgrecog/verify-claim
x-api-key: <KAILY_API_SECRET>
Content-Type: application/json
```

Request model: `app/models/verify_request.py`

```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "user_comment": "The oil bottle was leaking when it arrived",
  "claimed_product": "JioMart Sunflower Oil 1L",
  "order_id": "JM-77",
  "user_id": "u_9",
  "reference_image_base64": "optional",
  "idempotency_key": "optional",
  "claim_id": "optional"
}
```

Execution steps:

1. FastAPI authenticates the request through `verify_api_key`.
2. SlowAPI applies the `100/minute` rate limit.
3. `validate_image_size` rejects oversized images.
4. `compute_image_phash` computes the dedup identity.
5. `build_idempotency_key` and `find_replay("verify_claim", key)` prevent duplicate actions.
6. The router starts independent signal work:
   - web provenance check if `WEB_PROVENANCE_ENABLED=true`
   - Sightengine AI detection if `AI_DETECTOR_PROVIDER=sightengine`
7. `analyze_claim` calls Gemini for visible observations.
8. The router resolves AI-generated, duplicate-image, and web-provenance signals.
9. `build_verify_response` computes the authenticity score and recommended action.
10. `persist_decision` writes the audit record.
11. `register_image` stores the hash in the dedup index for future claims.

Authenticity scoring lives in `app/services/authenticity_engine.py`.

Default formula:

```text
base = 0.5 * image_comment_alignment + 0.5 * product_match

score = base
score *= (1 - AUTHENTICITY_AI_PENALTY * ai_probability) when AI image is flagged
score -= AUTHENTICITY_FLAG_PENALTY * number_of_other_flags
score -= WEB_MATCH_SOFT_PENALTY * web_hits when web matches are below hard threshold
score is clamped to [0, 1]
```

Routing:

| Condition | Verdict | Action |
| --- | --- | --- |
| Score >= `AUTHENTICITY_AUTO_APPROVE_THRESHOLD` | `authentic` | `auto_approve` |
| Score >= `AUTHENTICITY_REVIEW_THRESHOLD` | `review` | `manual_review` |
| Score below review threshold | `likely_fraud` | `reject` |
| Confident internal AI-generated signal | `review` | `manual_review` |
| Confident Sightengine AI-generated signal | `likely_fraud` | `reject` |
| Same image on another order or user | `likely_fraud` | `reject` |
| Full web matches across enough distinct domains | `likely_fraud` | `reject` |

The internal AI detector is advisory. It can force manual review, but it does
not auto-reject a customer by itself. Hard rejections come from stronger signals:
Sightengine, cross-claim duplicate image reuse, or public web reuse across the
configured domain threshold.

## Request Flow 3: Grocery Dispute Verification

Entrypoint: `app/routers/dispute.py`

Endpoint:

```http
POST /api/v1/imgrecog/dispute
x-api-key: <KAILY_API_SECRET>
Content-Type: application/json
```

Request model: `app/models/dispute_request.py` (`DisputeRequest` with `images`,
optional `dispute_category`, `is_rebuttal`, `ticket`, and `shipment`). The caller
supplies the Shipment Details API + Kapture fields; ImgRec does not call those
APIs.

Execution steps:

1. `verify_api_key` + the `100/minute` limiter.
2. `validate_image_size` on the primary image; `compute_image_phash` for dedup.
3. `build_idempotency_key` + `find_replay("dispute", key)` — a repeat returns the
   prior decision verbatim.
4. `classify_category` resolves the category via the fallback chain
   (`dispute_category` → description → notes → disposition → `insufficient_data`).
   If unresolved, the dispute is escalated to an agent with **no model call**.
5. `analyze_dispute` runs ONE Gemini observation call over all images (OCR,
   product match, damage, quality, spoilage, unit count, counterfeit/AI hints).
6. `find_duplicates` adds the reused-image fraud signal.
7. `dispute_engine.decide` computes the deterministic decision + refund, then
   applies the escalation gates (counterfeit, rebuttal, refund ceiling, fraud).
8. The router maps assist-mode / non-autonomous categories to `route="agent"`.
9. `persist_decision` writes the audit row (safe downgrade-to-agent on failure);
   `register_image` records the hash for future dedup.

Decision authority lives in `app/services/dispute_engine.py`. The model only
observes; the engine makes every approve/reject/refund call, so eligibility is
auditable and tunable via env (`REFUND_AUTO_APPROVE_MAX`, `DAIRY_MIN_SHELF_PCT`,
`NON_FNV_NEAR_EXPIRY_DAYS`, `DISPUTE_AUTONOMOUS_CATEGORIES`, `DISPUTE_ASSIST_MODE`).

New files for this flow:

| File | Purpose |
| --- | --- |
| `app/routers/dispute.py` | Dispute endpoint orchestration. |
| `app/models/dispute_request.py` | `DisputeRequest`, `Ticket`, `Shipment`. |
| `app/models/dispute_response.py` | `DisputeResponse`, `RefundResult`. |
| `app/services/dispute_service.py` | One multi-image Gemini observation call. |
| `app/services/dispute_engine.py` | Deterministic per-category decision + refund + gates. |
| `app/services/category_classifier.py` | Fallback-chain category resolution. |

## Common API Infrastructure

`app/main.py`

- Creates the FastAPI app.
- Registers CORS, rate limiting, exception handlers, and routers.
- Adds request IDs through an HTTP middleware.
- Serves the bundled demo UI at `/`.
- Exposes `/health` and `/ready`.
- Exports `handler = app` for Boltic/serverless ASGI compatibility.

`app/middleware/auth.py`

- Requires `x-api-key`.
- Compares the header with `KAILY_API_SECRET`.
- Returns `401` for missing or invalid keys.

`app/middleware/rate_limit.py`

- Creates the shared SlowAPI limiter.
- Routers apply `@limiter.limit("100/minute")`.

`app/middleware/error_handler.py`

- Converts FastAPI request validation errors to `422`.
- Converts Pydantic validation errors to `400`.
- Converts unhandled exceptions to a generic `500` with the exception type.

`app/utils/logger.py`

- Configures structlog JSON logging.
- Binds request context from `app/main.py`.
- Quiets noisy Google and HTTPX loggers.

## Models And Contracts

`app/models/request.py`

- Defines `ScanRequest`.
- Requires `image_base64`, `order_id`, and `user_id`.
- Allows `scan_type`: `auto`, `ocr`, or `damage`.
- Allows optional `idempotency_key` and `claim_id`.

`app/models/response.py`

- Defines `ScanResponse`.
- Contains OCR, damage, AI-generated evidence, final action, timestamp, and model name.
- Action domain: `initiate_refund`, `initiate_exchange`, `no_action`.

`app/models/verify_request.py`

- Defines `VerifyClaimRequest`.
- Requires image, user comment, claimed product, order ID, and user ID.
- Allows optional reference image, idempotency key, and claim ID.

`app/models/verify_response.py`

- Defines claim verification response objects:
  - AI-generated check
  - image/comment alignment
  - product match
  - recognition and OCR
  - web provenance
  - final authenticity score, verdict, action, and agent comment

## Gemini Integration

`app/services/gemini_service.py`

- Lazily creates a `google.genai.Client`.
- Supports two providers:
  - Vertex AI when `USE_VERTEX=true`
  - AI Studio API key when `USE_VERTEX=false`
- Uses JSON response mode.
- Disables Gemini 2.5 thinking budget when the SDK supports it, to reduce JSON
  truncation risk and latency.
- `analyze_image` powers the `/scan` endpoint.

`app/services/claim_service.py`

- Calls Gemini for `/verify-claim`.
- Sends the customer image, comment, claimed product, and optional reference image.
- Asks Gemini for observations only:
  - scene
  - objects
  - OCR text
  - AI-generated probability hint
  - image/comment alignment
  - product match
  - other flags
  - analyst summary

Both files parse Gemini's JSON output. If Gemini times out, routers map it to
`504`. If Gemini returns malformed or empty JSON, routers treat that as an
analysis failure.

## Deterministic Decision Services

`app/services/decision_engine.py`

- Owns scan response construction.
- Normalizes OCR dates and damage taxonomy.
- Computes expiry days.
- Applies refund/exchange/no-action rules.
- Guards against synthetic evidence and contradictory model outputs.

`app/services/authenticity_engine.py`

- Owns claim authenticity scoring and final routing.
- Computes `authenticity_score`, `score_out_of_100`, `verdict`,
  `recommended_action`, and `decision_confidence`.
- Adds clear agent comments explaining the route.

`app/services/damage_analyzer.py`

- Validates damage type and severity against the allowed taxonomy.
- Converts unknown or invalid damage values to safe `None` values.

`app/services/ocr_parser.py`

- Normalizes date strings to ISO format.
- Converts Indian date formats such as `DD/MM/YYYY`, `MMM YYYY`, and `MON-YY`.
- Computes `days_since_expiry`.

`app/utils/date_utils.py`

- Implements Indian date parsing with day-first behavior.
- Applies the last-day-of-month rule for month/year-only labels.

## Fraud And Authenticity Signals

`app/services/ai_image_detector.py`

- Provides one interface for AI-generated-image checks.
- `internal` provider:
  - uses image metadata fingerprints
  - folds in Gemini's visual hint
  - stays free and advisory-grade
- `sightengine` provider:
  - calls Sightengine's paid detector
  - still folds in metadata signals
  - can force reject when confidence is high

`app/utils/image_metadata.py`

- Extracts camera EXIF as evidence for a real photo.
- Looks for generator fingerprints such as Midjourney, DALL-E, Stable
  Diffusion, SynthID, or explicit AI provenance assertions.
- Avoids ambiguous false positives such as plain `c2pa`, `gemini`, or
  `openai`.

`app/services/dedup_index.py`

- Implements a perceptual-hash index for near-duplicate images.
- Uses a banded lookup over a 16-character hex dHash.
- Supports:
  - in-memory index for dev/tests
  - Redis index when `REDIS_URL` is set

`app/services/dedup_service.py`

- Classifies duplicate matches:
  - same order and same user: benign resubmission
  - different order or user: cross-claim duplicate fraud
- Handles index outages by returning no duplicate signal instead of failing the request.

`app/services/web_provenance.py`

- Calls Google Cloud Vision Web Detection when enabled.
- Counts full matches, partial matches, distinct pages, and distinct domains.
- Uses a hard timeout.
- Degrades to `checked=false` on missing credentials, bad base64, API errors,
  or timeouts.

## Audit, Idempotency, And Storage

`app/services/audit_service.py`

This file protects money-affecting workflows.

It owns:

- idempotency key construction
- replay lookup
- response reconstruction
- final audit persistence
- safe downgrade when audit persistence fails

Idempotency key priority:

1. Explicit `idempotency_key`
2. `claim_id` combined with `order_id`
3. Derived hash from `order_id`, `user_id`, and image identity

Persistence flow:

1. Decode image bytes.
2. Store image bytes through `ObjectStore`.
3. Store only the image reference and perceptual hash in the audit row.
4. Store raw model observations, computed values, and the exact response snapshot.
5. If another request inserted the same idempotency key first, replay that winner.
6. If the audit write fails, downgrade to manual review instead of returning an
   unrecorded automated action.

`app/db/repository.py`

- Defines `DecisionRecord`.
- Defines the `DecisionRepository` protocol.
- Provides in-memory repository for dev/tests.
- Selects SQLAlchemy repository when `DATABASE_URL` is set.

`app/db/sql_repository.py`

- Implements Postgres-compatible persistence with SQLAlchemy async sessions.
- Converts unique-key violations into `DuplicateDecision`.

`app/db/engine.py`

- Lazily creates the async SQLAlchemy engine and sessionmaker.
- Uses `pool_pre_ping=True` for stale connection handling.

`app/db/models.py`

- Defines the `claim_decisions` ORM model.
- Uses UUID primary key, unique idempotency key, JSON/JSONB columns, and indexes
  for order, user/time, and image hash lookup.

`app/storage/object_store.py`

- Defines the `ObjectStore` protocol.
- Supports:
  - memory store for dev/tests
  - local filesystem store
  - Google Cloud Storage for production
- Stores objects under keys like:

```text
<endpoint>/<YYYY>/<MM>/<DD>/<uuid>.<ext>
```

## Configuration

Config lives in `app/config/settings.py` and loads from `.env` through
`pydantic-settings`.

Core runtime variables:

| Variable | Meaning |
| --- | --- |
| `PORT` | Runtime port. README uses `8000`; Docker/Boltic uses `8080`. |
| `USE_VERTEX` | `true` uses Vertex AI; `false` uses AI Studio API key. |
| `GOOGLE_API_KEY` | AI Studio key when `USE_VERTEX=false`. |
| `VERTEX_PROJECT_ID` | GCP project when `USE_VERTEX=true`. |
| `VERTEX_REGION` | Vertex region. |
| `KAILY_API_SECRET` | Shared API key required in `x-api-key`. |
| `GEMINI_MODEL` | Gemini model ID. |
| `GEMINI_TIMEOUT_SECONDS` | Hard timeout for Gemini calls. |
| `MAX_IMAGE_SIZE_MB` | Decoded image size limit. |
| `ENVIRONMENT` | `production` enables strict config validation. |
| `LOG_LEVEL` | Structlog/logging level. |

Claim verification variables:

| Variable | Meaning |
| --- | --- |
| `AI_DETECTOR_PROVIDER` | `internal` or `sightengine`. |
| `SIGHTENGINE_API_USER` / `SIGHTENGINE_API_SECRET` | Sightengine credentials. |
| `AUTHENTICITY_WEIGHT_ALIGNMENT` | Alignment weight in score. |
| `AUTHENTICITY_WEIGHT_PRODUCT_MATCH` | Product-match weight in score. |
| `AUTHENTICITY_AI_PENALTY` | AI-image penalty multiplier. |
| `AUTHENTICITY_FLAG_PENALTY` | Penalty per fraud flag. |
| `AI_DETECTION_MIN_CONFIDENCE` | Minimum AI probability for forced routing. |
| `AUTHENTICITY_AUTO_APPROVE_THRESHOLD` | Score threshold for auto-approve. |
| `AUTHENTICITY_REVIEW_THRESHOLD` | Score threshold for manual review. |

Persistence and fraud-control variables:

| Variable | Meaning |
| --- | --- |
| `DATABASE_URL` | Async Postgres DSN for runtime audit store. |
| `OBJECT_STORE_PROVIDER` | `memory`, `local`, or `gcs`. |
| `OBJECT_STORE_LOCAL_DIR` | Local object store directory. |
| `GCS_BUCKET` | GCS bucket for production images. |
| `GCS_REGION` | Intended data residency region. |
| `REDIS_URL` | Redis DSN for shared dedup index. |
| `DEDUP_HAMMING_THRESHOLD` | Hamming distance threshold for near-duplicate images. |
| `DEDUP_WINDOW_DAYS` | Lookback window for duplicate detection. |
| `WEB_PROVENANCE_ENABLED` | Enables Google Vision Web Detection. |
| `WEB_MATCH_HARD_MIN_DOMAINS` | Distinct-domain threshold for hard reject. |
| `WEB_MATCH_SOFT_PENALTY` | Score penalty for web matches below hard threshold. |
| `VISION_TIMEOUT_SECONDS` | Web Detection timeout. |

Production guard:

When `ENVIRONMENT=production`, the app refuses to start with placeholder or
missing values for money-affecting dependencies:

- `KAILY_API_SECRET`
- Gemini provider credentials
- Sightengine credentials when Sightengine is selected
- `DATABASE_URL`
- non-memory object storage
- `GCS_BUCKET` when object storage is GCS
- `REDIS_URL`

## Database Migration

Alembic files:

- `alembic/env.py`
- `alembic/versions/0001_create_claim_decisions.py`
- `alembic.ini`

The migration creates `claim_decisions` with:

- UUID primary key
- unique `idempotency_key`
- endpoint and routing check constraints
- JSON/JSONB columns for raw observations, computed values, and response snapshot
- indexes on `order_id`, `(user_id, created_at)`, and `image_phash`

Runtime uses an async DSN such as `postgresql+asyncpg://...`.

Alembic should use a sync DSN through `ALEMBIC_URL`, for example:

```bash
ALEMBIC_URL=postgresql+psycopg2://user:pass@host:5432/imgrecog alembic upgrade head
```

## Demo Frontend

File: `demo/index.html`

The demo is a static HTML/CSS/JS console focused on the Scan workflow. The
visible screen is intentionally limited to the product condition demo:

- Expired
- Damaged
- Valid

It supports:

- PixelBin/user image URL input
- PixelBin/product image URL input
- customer query input
- automatic product status verdict: expired, damaged, valid, or unclear
- deterministic final decision: accept claim, reject claim, or review
- one `Scan product` action that posts to `/api/v1/imgrecog/evaluate-links`
- product status and decision summary cards

The demo no longer shows editable `order_id` or `user_id` fields. The frontend
does not ask the operator to pick or upload an image; the scan flow uses the
provided image links directly.

Important current behavior:

- When served through FastAPI at `/`, it calls the same origin.
- When opened directly through `file://`, it falls back to `http://localhost:8000`.
- It hardcodes `API_KEY = "missing-kaily-secret"`. The backend accepts that key
  only when `ENVIRONMENT=development`, so the local demo works without exposing
  the real configured `KAILY_API_SECRET`. Staging and production still require
  the configured shared secret.

## Deployment Files

`Dockerfile`

- Uses `python:3.11-slim`.
- Installs `requirements.txt`.
- Runs `uvicorn app.main:handler --host 0.0.0.0 --port ${PORT:-8080}`.

`Procfile`

- Defines the same ASGI process for platform runtimes:

```text
web: uvicorn app.main:handler --host 0.0.0.0 --port $PORT
```

`boltic.yaml`

- Configures Boltic serverless deployment.
- Uses Dockerfile build.
- Region: `asia-south1`.
- Port: `8080`.
- Staging env currently points to Vertex AI with `GEMINI_MODEL=gemini-2.5-flash`.
- Scaling is fixed at 10 min and 10 max instances.

`runtime.txt`

- Pins Python `3.11.9`.

## Tests

The test suite lives in `tests/`.

Test bootstrap:

- `tests/conftest.py` sets safe test environment values before app import.
- It pins `KAILY_API_SECRET=test-secret`.
- It resets the in-memory audit store, object store, and dedup index before and
  after each test.
- Gemini and external providers are mocked at service/router boundaries.

Test areas:

| Test file | Coverage |
| --- | --- |
| `test_scan.py` | Scan endpoint: refund/exchange/no-action, AI evidence guard, missing fields, wrong key, timeout. |
| `test_verify.py` | Verify endpoint: auto-approve, manual review, web provenance, web hard reject, task cancellation. |
| `test_authenticity_engine.py` | Score bands, AI routing, duplicate hard reject, web penalties, confidence. |
| `test_decision_engine.py` | Scan decision matrix and manual-review safeguards. |
| `test_idempotency.py` | Replay behavior, audit row completeness, no image blobs in DB rows, safe audit-write downgrade. |
| `test_verify_dedup.py` | Duplicate photo fraud across users/orders, benign same-claim resubmission. |
| `test_dedup_index.py` | Hamming distance and in-memory band index behavior. |
| `test_web_provenance.py` | Domain counting, clean images, Vision API degradation paths. |
| `test_image_metadata.py` | AI metadata fingerprints and false-positive avoidance. |
| `test_image_phash.py` | Image hash determinism and undecodable image behavior. |
| `test_image_url_fetcher.py` | Public URL validation, image fetch limits, and Gemini image optimization. |
| `test_link_evaluation.py` | URL evaluation endpoint, decision gates, product status, and deterministic expiry override. |
| `test_ocr_parser.py` | Indian date parsing and expiry math. |
| `test_sql_repository.py` | SQL repository roundtrip and duplicate handling with async SQLite. |
| `test_claim_schema.py` | Gemini claim generation config uses JSON MIME and response schema. |
| `test_verify_response_model.py` | Verify response defaults. |
| `test_demo_ui.py` | Root route serves the bundled demo HTML. |
| `test_settings_web.py` | Web provenance settings defaults. |
| `test_ai_image_detector.py` | Missing Gemini AI hint is not treated as generated. |

Run tests:

```bash
venv/bin/pytest -q
```

## File-By-File Map

Top-level files:

| File | Purpose |
| --- | --- |
| `README.md` | Main setup, endpoint contract, and integration notes. |
| `.env.example` | Safe template of all runtime environment variables. |
| `requirements.txt` | Runtime Python dependencies. |
| `requirements-dev.txt` | Test and developer dependencies. |
| `pytest.ini` | Pytest configuration. |
| `Dockerfile` | Container build and server command. |
| `Procfile` | Platform process command. |
| `boltic.yaml` | Boltic deployment configuration. |
| `runtime.txt` | Python runtime pin. |
| `alembic.ini` | Alembic configuration. |

Application files:

| File | Purpose |
| --- | --- |
| `app/main.py` | FastAPI app factory, startup logs, middleware, routers, health, readiness, demo UI. |
| `app/config/settings.py` | Pydantic settings and production config guard. |
| `app/middleware/auth.py` | Shared-secret API key validation. |
| `app/middleware/error_handler.py` | API error response handlers. |
| `app/middleware/rate_limit.py` | SlowAPI limiter singleton. |
| `app/models/request.py` | Scan request schema. |
| `app/models/response.py` | Scan response schema. |
| `app/models/verify_request.py` | Claim verification request schema. |
| `app/models/verify_response.py` | Claim verification response schema. |
| `app/routers/scan.py` | Product scan endpoint orchestration. |
| `app/routers/verify.py` | Claim verification endpoint orchestration. |
| `app/services/gemini_service.py` | Gemini client, scan prompt, JSON-mode scan analysis. |
| `app/services/claim_service.py` | Gemini claim-analysis prompt and parser. |
| `app/services/decision_engine.py` | Deterministic scan decision engine. |
| `app/services/authenticity_engine.py` | Deterministic claim scoring and routing engine. |
| `app/services/damage_analyzer.py` | Damage type/severity normalization. |
| `app/services/ocr_parser.py` | OCR date normalization and expiry calculation. |
| `app/services/ai_image_detector.py` | Internal/Sightengine AI-generated-image detector. |
| `app/services/web_provenance.py` | Google Vision reverse-image-search signal. |
| `app/services/dedup_index.py` | In-memory/Redis perceptual hash index. |
| `app/services/dedup_service.py` | Duplicate classification and resilient wrappers. |
| `app/services/audit_service.py` | Idempotency, replay, audit persistence, safe downgrade. |
| `app/db/repository.py` | Repository protocol, decision record, in-memory repo. |
| `app/db/sql_repository.py` | SQLAlchemy-backed audit repository. |
| `app/db/engine.py` | Async SQLAlchemy engine/session factory. |
| `app/db/models.py` | `claim_decisions` ORM model. |
| `app/storage/object_store.py` | Memory/local/GCS object storage abstraction. |
| `app/utils/image_utils.py` | Base64 image parsing, MIME detection, size validation, dHash. |
| `app/utils/image_metadata.py` | EXIF and AI-generator metadata inspection. |
| `app/utils/date_utils.py` | Indian date parsing helpers. |
| `app/utils/logger.py` | Structlog JSON logger configuration. |
| `app/__init__.py` and package `__init__.py` files | Package markers. |

Frontend and docs:

| File | Purpose |
| --- | --- |
| `demo/index.html` | Static browser console with scan, verify, and schema tabs. |
| `docs/superpowers/specs/2026-06-19-verify-claim-accuracy-speed-design.md` | Historical design/spec for verify-claim accuracy and speed work. |
| `docs/superpowers/plans/2026-06-19-verify-claim-accuracy-speed.md` | Historical implementation plan for verify-claim accuracy and speed work. |
| `docs/application-code-guide.md` | This code guide. |

Alembic files:

| File | Purpose |
| --- | --- |
| `alembic/env.py` | Migration environment and DB URL override handling. |
| `alembic/script.py.mako` | Alembic migration template. |
| `alembic/versions/0001_create_claim_decisions.py` | Creates and drops the audit table. |

Fixtures:

| File | Purpose |
| --- | --- |
| `tests/fixtures/expired_product.b64` | Sample expired product image base64. |
| `tests/fixtures/damaged_product.b64` | Sample damaged product image base64. |
| `tests/fixtures/valid_product.b64` | Sample valid product image base64. |

## Where To Change Common Things

Change endpoint behavior:

- Product scan HTTP flow: `app/routers/scan.py`
- Claim verification HTTP flow: `app/routers/verify.py`

Change Gemini prompts:

- Scan prompt: `app/services/gemini_service.py`
- Verify-claim prompt: `app/services/claim_service.py`
- Bump `SCAN_PROMPT_VERSION` or `VERIFY_PROMPT_VERSION` when prompt behavior changes.

Change refund/exchange rules:

- `app/services/decision_engine.py`
- Add or update tests in `tests/test_decision_engine.py` and `tests/test_scan.py`.

Change authenticity scoring:

- `app/services/authenticity_engine.py`
- Add or update tests in `tests/test_authenticity_engine.py` and `tests/test_verify.py`.

Change duplicate-image fraud logic:

- Hashing: `app/utils/image_utils.py`
- Index: `app/services/dedup_index.py`
- Classification: `app/services/dedup_service.py`
- Endpoint behavior: `tests/test_verify_dedup.py`

Change audit storage:

- Domain object and repo selection: `app/db/repository.py`
- SQL implementation: `app/db/sql_repository.py`
- Schema: `app/db/models.py` and Alembic migration
- Audit orchestration: `app/services/audit_service.py`

Change frontend demo:

- `demo/index.html`
- Watch the API base URL and API key constants near the bottom of the file.

## Codebase Recon Notes

Git history is small:

- Commits: 21
- First and latest commit date in this checkout: 2026-06-19
- Main contributor in history: Aditya Chaudhari

Most-changed files in current history:

- `tests/test_verify.py`
- `app/routers/verify.py`
- `tests/test_web_provenance.py`
- `tests/test_authenticity_engine.py`
- `requirements.txt`
- `boltic.yaml`

Files that appear in both churn and bug-fix history:

- `app/routers/verify.py`
- `tests/test_verify.py`
- `tests/test_web_provenance.py`
- `app/services/web_provenance.py`
- `app/utils/image_metadata.py`
- `requirements.txt`
- `boltic.yaml`

Start reading here when debugging production behavior:

1. `app/main.py`
2. `app/routers/scan.py`
3. `app/routers/verify.py`
4. `app/services/decision_engine.py`
5. `app/services/authenticity_engine.py`
6. `app/services/audit_service.py`

## Operational Pitfalls

- Do not let Gemini decide refunds directly. Keep money-moving actions in
  deterministic Python services.
- Do not remove audit downgrade behavior. A missing audit record should route to
  human review, not automated payout.
- Do not store base64 images in Postgres. Store an object-store key plus image hash.
- Do not treat internal AI-generated detection as enough for auto-reject. Current
  code only lets the internal provider force manual review.
- Do not run production with in-memory audit, object storage, or dedup. The
  settings guard intentionally blocks that.
- Keep `DATABASE_URL` async for runtime and `ALEMBIC_URL` sync for migrations.
- Prefer `http://127.0.0.1:8000/` for the demo UI so the frontend calls the same
  FastAPI origin.
- Do not extend the local demo key bypass beyond `ENVIRONMENT=development`.
  Staging and production must keep requiring `KAILY_API_SECRET`.
