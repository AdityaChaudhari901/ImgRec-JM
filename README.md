# Kaily · ImgRec API

Production-ready **AI grocery dispute-verification API** for **Kaily**, the AI
agent on JioMart's customer-support workflow. It exposes **one endpoint** —
`POST /api/v1/imgrecog/dispute` — that takes a customer's images + ticket text +
shipment data and returns a **deterministic** approve / reject / route-to-agent
decision with a refund amount, across all dispute categories (wrong product,
poor quality, damage, expiry/near-expiry, smell, MRP abuse, quantity mismatch).

The model only **observes** (OCR + visual evidence via **Gemini**); the Python
**dispute engine** makes every money-affecting decision, so eligibility is
auditable, idempotent, and never hallucinated. Built with **FastAPI** + the
**google-genai** SDK, backed by Postgres (audit), Redis (dedup), and GCS (images).
See [§9](#9-the-endpoint--grocery-dispute-verification) for the full contract.

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
| `GEMINI_VERTEX_FALLBACK_ENABLED` | When `USE_VERTEX=false`, retry via Vertex if AI Studio returns depleted-credit/quota errors (default `true`) |
| `GOOGLE_API_KEY` | Google AI Studio API key (used only when `USE_VERTEX=false`) |
| `KAILY_API_SECRET` | Shared secret Kaily sends as `x-api-key` |
| `GEMINI_MODEL` | Model id (default `gemini-2.0-flash-001`) |
| `GEMINI_TIMEOUT_SECONDS` | Hard timeout for the Gemini call (default 45) |
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
[ImgRec] Dispute endpoint: POST /api/v1/imgrecog/dispute
[ImgRec] Health check: GET /health
```

Health check: `GET http://localhost:8000/health`
Interactive docs: `http://localhost:8000/docs`

### Try the browser demo

Open `http://127.0.0.1:8000/` (served by the app). The console takes a customer
image URL + a query and calls the single `/dispute` endpoint, showing the
decision, route, refund, agent flags, and the raw JSON response.

---

## 5. Run tests

```bash
venv/bin/python -m pytest -q
```

All tests mock Gemini at the service boundary — **no real API calls, no key
required**. `tests/conftest.py` pins `KAILY_API_SECRET=test-secret` for auth.

---

## 6. Production hardening

What's already in place and what to wire up before a fully-autonomous launch.

**In the app already:**
- **Fail-fast config** — with `ENVIRONMENT=production`, the service refuses to boot
  if `KAILY_API_SECRET`, the active provider's credentials, `DATABASE_URL`, the
  object store, or `REDIS_URL` are missing/placeholder. A misconfigured prod
  service crashes loudly instead of silently serving errors.
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

## 7. Durable audit store + idempotency

Every dispute decision is **persisted and idempotent** — the minimum state
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

Optional request fields: `idempotency_key`, `claim_id`. Related env vars: see
`.env.example` (`DATABASE_URL`, `OBJECT_STORE_PROVIDER`, `GCS_BUCKET`,
`GCS_REGION`, `IMAGE_RETENTION_DAYS`, `DISPUTE_PROMPT_VERSION`).

## 8. Reused-image fraud detection

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

## 9. The endpoint — Grocery dispute verification

One endpoint that resolves a grocery delivery dispute end-to-end: it takes the
customer images + ticket text + shipment data (the caller — Kaily — supplies the
Shipment Details API / Kapture fields) and returns a deterministic
approve / reject / route-to-agent decision with a refund amount and routing flags.

### `POST /api/v1/imgrecog/dispute`

**Request**

```json
{
  "images": ["data:image/jpeg;base64,/9j/..."],
  "dispute_category": "mrp_abuse",
  "is_rebuttal": false,
  "ticket": { "title": "...", "description": "...", "notes": "...", "disposition_code": "PRICE_DISPUTE" },
  "shipment": {
    "order_tracking_id": "JM-29384", "product_name": "Amul Gold Milk 500ml",
    "product_type": "dairy", "mrp": 33.0, "selling_price": 31.0,
    "invoice_amount": 62.0, "quantity": 2, "seller_type": "1P"
  }
}
```

`dispute_category` is optional — if omitted, it's inferred from the ticket
description → notes → disposition code, else the dispute is escalated with
`insufficient_data`. `product_type`: `fnv | non_fnv | dairy`. `seller_type`:
`1P | 3P`. `images` accepts 1..`DISPUTE_MAX_IMAGES` data URIs or raw base64.

**Response `200`**

```json
{
  "category": "mrp_abuse", "category_source": "provided",
  "decision": "approve", "route": "auto", "agent_flags": [],
  "refund": { "eligible": true, "amount": 4.0, "type": "price_difference",
              "assign_to_mpt": false, "seller_debit": false },
  "recommendation": "Approve ₹4 price-difference refund (1P overcharge).",
  "confidence": 0.9, "observations": { "...": "..." }
}
```

### Categories & rules

| Category | Approve when | Reject when |
|---|---|---|
| `wrong_product` | image ≠ ordered product | matches order |
| `poor_quality` | visual defect supports claim | looks normal (*internal defect → agent*) |
| `damaged` | seal/leak/tear/crush/tamper visible | packaging intact |
| `expiry` (non-FNV) | ≤ `NON_FNV_NEAR_EXPIRY_DAYS` (45) to expiry | > 45 days |
| `expiry` (dairy) | shelf-left < `DAIRY_MIN_SHELF_PCT` (30%) | ≥ 30% |
| `expiry` (FNV) | judged by `poor_quality` (variable shelf life) | |
| `smell` | detailed report + visible spoilage proxy | insufficient evidence |
| `mrp_abuse` | printed MRP **<** invoice MRP (overcharge) | printed ≥ invoice |
| `quantity_mismatch` | counted units < ordered (confident) | counts match |

**MRP refund:** 1P → price difference `(charged − printed MRP) × qty`; 3P → full
selling price + `assign_to_mpt`/`seller_debit` flags (Kaily/Kapture executes the
seller debit; ImgRec only emits the flags).

**Always routed to an agent** (with the AI recommendation attached): counterfeit,
post-rejection rebuttal (`is_rebuttal: true`), any approved refund ≥
`REFUND_AUTO_APPROVE_MAX` (₹500), reused-image fraud, AI-generated image, and
`insufficient_data`. The decision is computed deterministically in
`dispute_engine.py` — Gemini only supplies observations — so every refund is
auditable, idempotent, and persisted (with safe downgrade-to-agent on audit
failure), exactly like the other endpoints.

### Tunable env vars

```bash
REFUND_AUTO_APPROVE_MAX=500                 # >= this (₹) -> agent approval
DISPUTE_ASSIST_MODE=false                   # true = recommend-only (no auto-act)
DISPUTE_AUTONOMOUS_CATEGORIES=mrp_abuse,expiry,wrong_product,damaged
DAIRY_MIN_SHELF_PCT=30
NON_FNV_NEAR_EXPIRY_DAYS=45
DISPUTE_MAX_IMAGES=5
GEMINI_MAX_CONCURRENCY=8                     # per-instance model-call backpressure
DISPUTE_PROMPT_VERSION=dispute-v1
```

`DISPUTE_ASSIST_MODE` and `DISPUTE_AUTONOMOUS_CATEGORIES` enable a progressive
rollout — start subjective categories (poor_quality, smell, quantity) in
recommend-only and promote them to autonomous once they clear the accuracy bar,
without a redeploy.

### Accuracy eval harness

Measure dispute accuracy with `eval_harness/` — a labelled-case runner that scores
the real classifier + decision engine and reports per-category accuracy, an
approve/reject/agent confusion matrix, and approve precision/recall.

```bash
# Engine mode (offline, no API cost) — scores the deterministic logic against the
# seed labelled set; gates at 95%.
venv/bin/python -m eval_harness.run --threshold 0.95

# Custom labelled set:
venv/bin/python -m eval_harness.run --manifest path/to/cases.jsonl --threshold 0.95

# End-to-end (real Gemini over real images) — drop labelled JioMart photos into a
# manifest with "images": [...] and run:
venv/bin/python -m eval_harness.run --mode e2e --threshold 0.95
```

The bundled `eval_harness/data/seed_manifest.jsonl` covers every category and its
decision boundaries (45-day edge, dairy 30% edge, MRP equality, high-value ceiling,
counterfeit/rebuttal/AI/dedup escalations, classification fallback) and is asserted
to score **100%** in CI (`tests/test_eval_seed_golden.py`) — it doubles as a golden
regression set, so any change to a decision rule that breaks a boundary fails the
build. To get the *model-level* number, add real labelled photos and run `--mode e2e`.

> Engine mode measures the engine's **recommendation** (approve/reject/agent) — the
> business-logic accuracy. Whether a category auto-acts is a separate routing policy
> (`DISPUTE_AUTONOMOUS_CATEGORIES` / `DISPUTE_ASSIST_MODE`).

---

## 10. Deploy to Boltic (serverless)

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
>
> The demo Boltic config uses `USE_VERTEX=false` and
> `WEB_PROVENANCE_ENABLED=false` so a disabled Cloud Vision API or missing Vertex
> IAM role cannot break the demo. Keep `GOOGLE_API_KEY` as a Boltic secret/env
> var; do not commit it to `boltic.yaml`.
> If AI Studio credits are depleted, `GEMINI_VERTEX_FALLBACK_ENABLED=true` retries
> Gemini calls through Vertex as long as the Boltic runtime service account has
> `roles/aiplatform.user` on `VERTEX_PROJECT_ID`.

---

## 11. Swapping the Gemini model

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
│   ├── main.py                    # FastAPI app + lifespan + Boltic handler export
│   ├── routers/dispute.py         # POST /api/v1/imgrecog/dispute  (the one endpoint)
│   ├── services/
│   │   ├── dispute_service.py     # One multi-image Gemini observation call
│   │   ├── dispute_engine.py      # Deterministic per-category decision + refund + gates
│   │   ├── category_classifier.py # Fallback-chain category resolution
│   │   ├── gemini_service.py      # Gemini client (async + timeout + concurrency cap)
│   │   ├── image_url_fetcher.py   # SSRF-safe URL fetch + PixelBin templating
│   │   ├── ocr_parser.py          # Date/MRP/shelf-life maths
│   │   ├── damage_analyzer.py     # Damage taxonomy validation
│   │   ├── dedup_service.py       # Reused-image fraud (Redis pHash)
│   │   └── audit_service.py       # Idempotency + durable audit + safe downgrade
│   ├── models/                 # Pydantic v2 dispute request/response schemas
│   ├── middleware/             # auth · error handlers · rate limit
│   ├── config/settings.py      # Pydantic BaseSettings (all env vars)
│   ├── db/ · storage/          # Postgres audit repo · GCS/local/memory object store
│   └── utils/                  # image · date · structured logger
├── eval_harness/               # Accuracy eval (engine + e2e modes) + seed dataset
├── demo/index.html             # Standalone browser demo for /dispute
├── tests/                      # pytest suite (Gemini mocked) + fixtures
├── requirements.txt / -dev.txt
└── .env.example
```
