# Kaily · ImgRec API

Production-ready **Image Recognition API** for **Kaily**, the AI agent on JioMart's
customer-support workflow. It inspects user-uploaded product photos, detects
**expired** and **damaged** goods using **Gemini 2.0 Flash**, and returns a
structured decision that Kaily uses to trigger **refund / exchange / no-action**.

Stateless, single-endpoint, no database. Built with **FastAPI** + the
**google-generativeai** SDK.

---

## 1. Prerequisites

- **Python 3.11+**
- **pip**
- A **Google AI Studio API key** ([aistudio.google.com](https://aistudio.google.com/app/apikey))

---

## 2. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For development/testing tooling as well:

```bash
pip install -r requirements-dev.txt
```

---

## 3. Configure environment

```bash
cp .env.example .env
```

Then fill in `.env`:

| Variable | Purpose |
|---|---|
| `USE_VERTEX` | `true` = call Gemini via **Vertex AI** (bills GCP project, auth via ADC); `false` = use the AI Studio `GOOGLE_API_KEY` (its own prepay billing) |
| `GOOGLE_API_KEY` | Google AI Studio API key (used only when `USE_VERTEX=false`) |
| `KAILY_API_SECRET` | Shared secret Kaily sends as `x-api-key` |
| `GEMINI_MODEL` | Model id (default `gemini-2.0-flash-001`) |
| `GEMINI_TIMEOUT_SECONDS` | Hard timeout for the Gemini call (default 15) |
| `MAX_IMAGE_SIZE_MB` | Max decoded image size before a 413 (default 10) |
| `VERTEX_PROJECT_ID` / `VERTEX_REGION` | GCP project/region (used when `USE_VERTEX=true`) |
| `GEMINI_MODEL` | Model id. Note: on Vertex, the id must be provisioned for your project (this project exposes `gemini-2.5-flash`, not `gemini-2.0-flash-001`) |
| `ENVIRONMENT` / `LOG_LEVEL` / `PORT` | Runtime config |

### Using Vertex AI (`USE_VERTEX=true`)

Vertex authenticates via **Application Default Credentials**, not an API key.
Once per machine:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>
```

Then set in `.env`: `USE_VERTEX=true`, `VERTEX_PROJECT_ID`, `VERTEX_REGION`,
and a `GEMINI_MODEL` provisioned for that project. No `GOOGLE_API_KEY` needed.

---

## 4. Run

```bash
uvicorn app.main:app --reload --port 8000
```

On startup you'll see:

```
[ImgRec] Server running on port 8000
[ImgRec] Model: gemini-2.0-flash-001
[ImgRec] Environment: development
[ImgRec] Kaily endpoint: POST /api/v1/imgrecog/scan
[ImgRec] Health check: GET /health
```

Health check: `GET http://localhost:8000/health`
Interactive docs: `http://localhost:8000/docs`

### Try the browser demo

Open `demo/index.html` directly in a browser. It has drag-and-drop upload plus
three one-click demo scenarios (Expired / Damaged / Valid) that call your local
server. Set the **Base URL** and **x-api-key** at the top to match your `.env`.

---

## 5. Run tests

```bash
pytest tests/ -v
```

All tests mock Gemini at the `analyze_image` boundary — **no real API calls, no
key required**. `tests/conftest.py` pins `KAILY_API_SECRET=test-secret` for auth.

---

## 6. Endpoint contract

### `POST /api/v1/imgrecog/scan`

**Headers**

```
Content-Type: application/json
x-api-key: <KAILY_API_SECRET>
```

**Request**

```json
{
  "image_base64": "data:image/jpeg;base64,/9j/...",
  "order_id": "JM-29384",
  "user_id": "u_kaily_123",
  "scan_type": "auto"
}
```

`scan_type`: `auto` (default) · `ocr` (bias to dates) · `damage` (bias to damage).
`image_base64` accepts a full data URI **or** a raw base64 string (JPEG/PNG/WebP).

**Response `200`**

```json
{
  "success": true,
  "request_id": "req_1718612345_a3f9",
  "order_id": "JM-29384",
  "user_id": "u_kaily_123",
  "status": "expired",
  "confidence": 0.96,
  "ocr": {
    "manufacture_date": "2024-01-15",
    "expiry_date": "2025-05-20",
    "batch_no": "B2401K",
    "days_since_expiry": 28,
    "raw_text": "MFG JAN 2024  EXP 20/05/2025  BATCH B2401K"
  },
  "damage": { "detected": false, "type": null, "severity": null, "description": null },
  "action": {
    "type": "initiate_refund",
    "message": "Product expired 28 days ago. Refund process triggered.",
    "refund_eligible": true,
    "priority": "high"
  },
  "processed_at": "2026-06-17T10:30:00Z",
  "model_used": "gemini-2.0-flash-001"
}
```

**Field domains**

- `status`: `expired` · `damaged` · `valid` · `unclear`
- `damage.type`: `crushed_packaging` · `tear` · `broken_seal` · `leakage` · `dent` · `discoloration` · `mold`
- `damage.severity`: `minor` · `moderate` · `severe`
- `action.type`: `initiate_refund` · `initiate_exchange` · `no_action`
- `action.priority`: `high` · `medium` · `low`

**Errors**

| Code | Meaning |
|---|---|
| `400` / `422` | Validation error (missing fields / malformed body) |
| `401` | Invalid or missing `x-api-key` |
| `413` | Image exceeds `MAX_IMAGE_SIZE_MB` |
| `429` | Rate limit exceeded (100/min per IP) |
| `504` | Gemini analysis timed out |

> **Decision authority:** the model supplies the *observations* (OCR text, damage
> type/severity); the refund/exchange *decision* and `days_since_expiry` are computed
> deterministically in `decision_engine.py` / `ocr_parser.py`, so eligibility is
> auditable and never hallucinated.

---

## 7. Kaily integration

- **Endpoint URL:** `POST https://<your-host>/api/v1/imgrecog/scan`
- **Required header:** `x-api-key: <KAILY_API_SECRET>`
- **Branch on `action.type`:**

| `action.type` | Kaily should… |
|---|---|
| `initiate_refund` | Start the refund workflow (expired, or severe damage) |
| `initiate_exchange` | Start the exchange workflow (minor/moderate damage) |
| `no_action` | Take no automated action (valid, or unclear image) |

Use `action.refund_eligible` and `action.priority` for queue routing/SLAs.

---

## 7b. Claim authenticity verification (fraud scoring)

A second endpoint scores whether a customer's *damage/destroyed-product* claim is
authentic, so Kaily can auto-approve genuine refunds and route suspicious ones to
a human.

### `POST /api/v1/imgrecog/verify-claim`

**Request**

```json
{
  "image_base64": "data:image/jpeg;base64,/9j/...",
  "user_comment": "The oil bottle was leaking when it arrived",
  "claimed_product": "JioMart Sunflower Oil 1L",
  "order_id": "JM-77",
  "user_id": "u_9",
  "reference_image_base64": "optional catalog image"
}
```

**Response `200`**

```json
{
  "success": true,
  "request_id": "vfy_1781679817_kjc0",
  "order_id": "JM-77",
  "user_id": "u_9",
  "authenticity_score": 0.94,
  "decision_confidence": 0.88,
  "verdict": "authentic",
  "recommended_action": "auto_approve",
  "recognition": {
    "scene": "A leaking oil bottle on a kitchen counter",
    "objects": ["oil bottle"],
    "extracted_text": "JioMart Sunflower Oil 1L  MFG 02/2026  BATCH F2602"
  },
  "checks": {
    "ai_generated": { "is_ai_generated": false, "ai_probability": 0.05, "source": "internal", "signals": [] },
    "image_comment_alignment": { "score": 0.9, "aligned": true, "reason": "Leak visible" },
    "product_match": { "matches": true, "score": 0.98, "detected_product": "...", "reason": "..." },
    "other_flags": []
  },
  "agent_comment": "Human-readable analyst summary for the support agent.",
  "processed_at": "2026-06-17T07:03:37Z",
  "model_used": "gemini-2.5-flash"
}
```

**What Kaily branches on** — `recommended_action`:

| value | meaning |
|---|---|
| `auto_approve` | Genuine claim — proceed with refund/exchange |
| `manual_review` | Send to a human (mid-confidence, or an *advisory* AI-generated flag) |
| `reject` | Likely fraud — comment/product mismatch, or a confident specialist AI-detection |

### What it returns (per claim)

- **`recognition`** — image + text understanding: a one-line scene description,
  the objects/products seen, and **OCR** (`extracted_text`) of all text on the
  packaging/label.
- **`checks`** — the four fraud signals:
  1. **AI-generated image** — is the photo synthetic/edited rather than a real phone photo?
  2. **Image ↔ comment alignment** — does the photo actually show the problem described?
  3. **Product match** — is it the product the ticket is about?
  4. **Other flags** — stock photo, screenshot, screen-of-a-screen, watermark, unrelated item…

The final `authenticity_score` and routing are computed **deterministically** in
`authenticity_engine.py` from those signals — not by the model — so it's auditable
and tunable via env weights/thresholds. `ai_probability` is always
"probability the image is AI-generated" (0 = real photo, 1 = AI), and
`decision_confidence` (0–1) says how decisively the recommended action sits in its
band — so Kaily can require e.g. `decision_confidence >= 0.8` before acting
unattended.

### Design choice & cost (why this approach)

AI-generated-image detection is the only check a general vision model is weak at,
so it's handled by a **pluggable detector** with two providers:

| Provider (`AI_DETECTOR_PROVIDER`) | Accuracy | Cost / image | Notes |
|---|---|---|---|
| **`internal`** (default) | advisory | **~$0** | Free in-process EXIF/C2PA/SynthID-marker scan **+** Gemini visual heuristic. Never auto-rejects on this signal alone — routes to `manual_review`. |
| **`sightengine`** | ~high | **~$0.01** (5 ops/call) | Paid specialist API; a confident verdict here *can* `reject`. Sends the image to a third party. |

The rest of the verification (alignment, product match, reasoning) runs in the
**same Gemini call** at **~$0.001/request** (vs Sightengine's ~$0.01 for the AI
check alone). Specialist alternatives we evaluated and rejected for v1: **SynthID**
(only detects Google's own AI tools), **Reality Defender** (50/mo free cap, opaque
enterprise pricing), **Sensity** (SaaS quota, not per-request).

**Recommendation:** stay on `internal` (advisory) until AI-image fraud proves
material, then flip `AI_DETECTOR_PROVIDER=sightengine` (+ keys) — no code change.

### Tunable env vars

```
AI_DETECTOR_PROVIDER=internal              # or "sightengine"
SIGHTENGINE_API_USER=                       # only if using sightengine
SIGHTENGINE_API_SECRET=
AUTHENTICITY_WEIGHT_ALIGNMENT=0.5           # base-score weights (sum to 1.0)
AUTHENTICITY_WEIGHT_PRODUCT_MATCH=0.5
AUTHENTICITY_AI_PENALTY=0.6                 # score penalty when image looks AI-made
AUTHENTICITY_FLAG_PENALTY=0.1               # per other_flag
AI_DETECTION_MIN_CONFIDENCE=0.6             # min conf before AI verdict affects routing
AUTHENTICITY_AUTO_APPROVE_THRESHOLD=0.75
AUTHENTICITY_REVIEW_THRESHOLD=0.45
```

> **Stateless note:** duplicate/reused-image detection across tickets needs
> persistence and is intentionally out of scope for this stateless service — add a
> hash store (e.g. perceptual-hash + Redis) when you're ready to break statelessness.

---

## 7c. Production hardening

What's already in place and what to wire up before a fully-autonomous launch.

**In the app already:**
- **Fail-fast config** — with `ENVIRONMENT=production`, the service refuses to boot
  if `KAILY_API_SECRET`, the active provider's credentials, or (when enabled)
  Sightengine keys are missing/placeholder. A misconfigured prod service crashes
  loudly instead of silently serving errors.
- **Health vs readiness** — `GET /health` is liveness (process up, restart on hang);
  `GET /ready` is readiness (config valid + Gemini client initialises, no billed
  call) — gate the load balancer on `/ready`.
- **Correlation IDs** — every request gets a `request_id` (inbound `X-Request-ID`
  honoured, else generated), stamped on every structured log line and echoed back
  as the `X-Request-ID` response header.
- **Structured JSON logs**, **auth** on every endpoint (default-closed), **input
  validation**, **timeouts** + upstream-error mapping, and **no stack traces** in
  responses.

**Wire up before production (not in scope of this repo):**
- **Secrets** → move out of `.env` into **GCP Secret Manager** (or your platform's
  secret store), injected at runtime. Never commit `.env`.
- **Vertex auth on serverless** → ADC/`gcloud` doesn't exist there; attach a
  **service account** (`GOOGLE_APPLICATION_CREDENTIALS`) or use workload identity.
- **Rate limiting across instances** → slowapi is **in-memory per instance**. With
  >1 replica, back it with Redis (`slowapi` + `storage_uri="redis://…"`) so the
  100/min limit is global.
- **Observability** → export metrics (the four golden signals) + alerts on
  latency/error SLOs; add tracing if Kaily propagates a trace context.
- **Audit store** → persist each verdict (who/what/why/score) durably for a
  money-affecting refund decision — stdout logs aren't an audit trail.
- **Decision tuning** → validate thresholds on **real labelled claim photos** and
  run a **shadow/assist period** before auto-approving/rejecting unattended.

---

## 7d. Durable audit store + idempotency (Phase 1)

Every scan/verify decision is now **persisted and idempotent** — the minimum state
needed to run autonomously.

- **Audit store** — each call writes a `claim_decisions` row: who/what, the image
  *reference* + perceptual hash (never the blob), model + prompt version, raw
  observations, computed scores, the final action/routing, and a verbatim snapshot
  of the response. Backed by **Postgres** (SQLAlchemy async + asyncpg) in prod;
  an in-memory fallback runs in dev (production refuses to boot without `DATABASE_URL`).
- **Object storage** — uploaded images go to **GCS** (`OBJECT_STORE_PROVIDER=gcs`,
  India region by default); the DB holds only the storage key. `local`/`memory`
  providers exist for dev/test.
- **Idempotency** — send an `idempotency_key` (or `claim_id`) on the request, or
  one is derived from `(order_id, user_id, image phash)`. A repeat returns the
  **prior decision verbatim** — no second model call, no second refund.
- **Safe failure** — if the audit write fails, the response is **downgraded to
  manual review** rather than auto-approving without a record.

**Migrations (Alembic, reversible):**

```bash
# Use a SYNC driver for migrations (psycopg2), not asyncpg:
ALEMBIC_URL=postgresql+psycopg2://user:pass@host:5432/imgrecog alembic upgrade head
ALEMBIC_URL=...                                                alembic downgrade base   # clean rollback
```

New request fields (both endpoints, optional — backward-compatible):
`idempotency_key`, `claim_id`. New env vars: see `.env.example` (`DATABASE_URL`,
`OBJECT_STORE_PROVIDER`, `GCS_BUCKET`, `GCS_REGION`, `IMAGE_RETENTION_DAYS`,
`SCAN_PROMPT_VERSION`, `VERIFY_PROMPT_VERSION`).

## 7e. Reused-image fraud detection (Phase 2)

The most common refund-fraud vector is the **same photo submitted across many
orders/accounts**. The service now catches it.

- Every `/verify-claim` image gets a **perceptual hash** (dHash) and is checked
  against prior claims via a **band (LSH) index** in **Redis** (in-memory fallback
  in dev). Near-duplicates are found within a configurable **Hamming distance**
  (`DEDUP_HAMMING_THRESHOLD`, default 10) and **window** (`DEDUP_WINDOW_DAYS`),
  so re-compressed/cropped copies are caught — not just byte-identical ones.
- **Same photo on a different order/user → hard fraud signal → `reject`**, with the
  matched prior claim ids recorded in the audit row as the justification. This is a
  *deterministic, explainable* signal (principle #2) — unlike the advisory AI score,
  it may drive an automated rejection. `decision_confidence` is `1.0` for such a
  hard match.
- **Same photo re-submitted on the same order+user → benign** (a re-shoot), routed
  normally — not flagged as fraud.
- Resilient: a Redis outage degrades to "no duplicate signal" (logged) rather than
  failing the claim — it never wrongly auto-rejects on a missing index.

> Cross-account dedup is **why this service needs state** — see Phase 1. Production
> requires `REDIS_URL` (it refuses to boot without it).

---

## 8. Deploy to Boltic (serverless)

1. In the Boltic dashboard, set **all** env vars from `.env.example`
   (`GOOGLE_API_KEY`, `KAILY_API_SECRET`, `GEMINI_MODEL`, …).
2. Point Boltic at the ASGI export: **`app.main:handler`**
   (`handler = app` is exported at the bottom of `app/main.py`).
3. **No build step required** — it's a pure-Python ASGI app; Boltic installs
   `requirements.txt` and serves the handler directly.

> **Auth on serverless:** `gcloud` ADC isn't available in a serverless runtime.
> For `USE_VERTEX=true`, attach a **service account** (set
> `GOOGLE_APPLICATION_CREDENTIALS` to a mounted key file, or use the platform's
> workload identity). Alternatively run with `USE_VERTEX=false` and a
> `GOOGLE_API_KEY` — no ADC needed.

---

## 9. Swapping the Gemini model

Change one env var — no code change:

```bash
GEMINI_MODEL=gemini-2.0-flash-001   # or gemini-1.5-pro, gemini-2.5-flash, …
```

Restart the server; the new model id appears in startup logs and in
`response.model_used`.

---

## Project layout

```
imgrecog-kaily/
├── app/
│   ├── main.py                 # FastAPI app + lifespan + Boltic handler export
│   ├── routers/scan.py         # POST /api/v1/imgrecog/scan
│   ├── services/
│   │   ├── gemini_service.py   # Gemini 2.0 Flash call (SDK, async + timeout)
│   │   ├── ocr_parser.py       # Date normalisation + expiry maths
│   │   ├── damage_analyzer.py  # Damage taxonomy validation + severity score
│   │   └── decision_engine.py  # Deterministic refund/exchange decision
│   ├── models/                 # Pydantic v2 request/response schemas
│   ├── middleware/             # auth · error handlers · rate limit
│   ├── config/settings.py      # Pydantic BaseSettings (all env vars)
│   └── utils/                  # image · date · structured logger
├── demo/index.html             # Standalone browser demo (zero deps)
├── tests/                      # pytest suite (Gemini mocked) + fixtures
├── requirements.txt / -dev.txt
└── .env.example
```
