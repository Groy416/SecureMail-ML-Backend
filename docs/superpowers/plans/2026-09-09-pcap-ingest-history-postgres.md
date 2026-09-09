# PCAP ingest, per-session analysis, history, Postgres Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the runtime database over to Postgres, accept one authorized PCAP, convert it with TShark into `session-features.v1` JSON, let the frontend submit those sessions one-by-one to the existing analyzer, and list the resulting history.

**Architecture:** Reuse `ml.pcap.extract_sessions` (already TShark-backed) as a passive adapter. Do not parse in the browser and do not send bodies/credentials/keylogs. Persist capture jobs + extracted session JSON + analysis rows in Postgres. Frontend loop: upload PCAP → preview sessions → `POST /api/v1/analyses` once per session → read history by capture.

**Tech Stack:** FastAPI, SQLAlchemy asyncio, asyncpg, Postgres 16, TShark via `ml.pcap`, existing `predict_session` / `inference_record`.

**Spec:** `docs/superpowers/specs/2026-09-05-frontend-api-ml-dashboard-design.md` (API envelope, privacy, rules-first) and `docs/specs/spec.md` §8, §17, §20, §23. This plan **extends** the dashboard spec: that spec forbade raw PCAP upload; this plan adds a bounded, server-side extract path that still never puts payload content into ML.

## Global Constraints

- Runtime DB is Postgres only (`postgresql+asyncpg://…`). SQLite stays as a pytest convenience, not a deploy path.
- One PCAP per ingest request. Sessions are analyzed **one HTTP request each** (`POST /api/v1/analyses`).
- TShark / `extract_sessions` emit `session-features.v1` only. No email body, subject, attachment, password, cookie, token, private key, or TLS keylog in storage or ML input (`ml.product.FORBIDDEN_PAYLOAD_KEYS`).
- PCAP bytes are not stored. Hash + bounded temp file, then delete.
- Deterministic rules remain the policy floor. ML cannot downgrade them.
- Auth required on ingest, preview, analysis, and history when `SECUREMAIL_API_KEY` is set.
- API image must include `tshark` (or a documented host binary). No docker-in-docker on the VPS.
- Max PCAP size 20 MiB unless product later raises it. Reject empty / non-pcap magic.
- `source_type` for uploaded captures: `authorized_capture`. No synthetic labels from the client.

---

## Current inventory (do not reinvent)

### Already shipped HTTP API

| Method | Path | Auth today | Role |
|---|---|---|---|
| GET | `/api/v1/health` | no | readiness + bundle |
| GET | `/api/v1/bundle` | no | model metadata |
| GET | `/api/v1/rules` | no | deterministic rule catalog |
| POST | `/api/v1/analyses` | yes if key set | **keep** — one session → `ml-result.v1` |
| POST | `/api/v1/validate` | no | preflight schema/privacy |
| GET | `/api/v1/analyses` | no | flat history list (paginated) |
| GET | `/api/v1/analyses/stats` | no | aggregates |
| GET | `/api/v1/analyses/{request_id}` | no | one analysis row |
| DELETE | `/api/v1/analyses/{request_id}` | no | delete one |
| POST | `/api/v1/analyses/synthetic` | no | test fixtures (keep internal) |
| GET | `/api/v1/validations` | no | validation log |
| DELETE | `/api/v1/analyses?confirm=true` | no | wipe |

### Already shipped extraction (reuse)

- `ml.pcap.extract_sessions(pcap_path, scenario=…, environment_id=…, destination_port=…, source_type=…)` → `list[SessionFeatureRecord]`
- Host `tshark` first; Docker lab image fallback (VPS should not need the fallback)
- Groups TCP streams, STARTTLS/TLS/cert metadata, packet-range evidence
- One call is **one protocol + one dest port**. A mixed PCAP needs one call per detected mail port

### Already started Postgres wiring (finish in Phase 1)

- `compose.yaml`: `db` (postgres:16-alpine) + `api` `DATABASE_URL=postgresql+asyncpg://…@db:5432/…`
- `asyncpg` in `pyproject.toml`
- Alembic initial revision `9f2f635d439f` (SQLite-oriented `create_all` still used at boot)
- `api/config.py` still defaults to SQLite — **must change for runtime**

### Missing vs this product slice

- No capture/job table, no PCAP upload endpoint, no session preview (“what is being checked”)
- History is a flat analysis list, not grouped by PCAP
- Auth not applied to history/data routes
- API image has no `tshark`
- Boot uses `create_all`, not Alembic against Postgres

---

## File map

| File | Responsibility |
|---|---|
| `compose.yaml` | Postgres + API only (already) |
| `Dockerfile` | add `tshark`; Postgres URL already |
| `api/config.py` | Postgres default; PCAP size/dir settings |
| `api/database.py` | `CaptureJob`, `CaptureSession` models |
| `migrations/versions/*` | Postgres-safe tables for captures |
| `api/dependencies.py` | engine, auth on protected routes |
| `ml/pcap.py` | add `detect_mail_ports` + authorized-capture wrapper (thin) |
| `api/capture_routes.py` | ingest + preview + capture history |
| `api/schemas.py` | capture envelopes |
| `api/routes.py` | link `capture_id` on analysis persist |
| `api/data_routes.py` | filter history by `capture_id` |
| `tests/test_api.py` | existing analysis tests stay |
| `tests/test_capture_api.py` | ingest/preview/history with fixture PCAP |

---

## Target flow

```text
frontend
  POST /api/v1/captures          (multipart: one .pcap)
       -> auth, size/magic gate
       -> write temp file
       -> detect SMTP/IMAP/POP3 dest ports
       -> extract_sessions per port (tshark)
       -> persist job + session JSON in Postgres
       -> delete temp PCAP
       -> 201 CaptureResponse { capture_id, status, sessions: preview[] }

frontend (1 request per session)
  GET  /api/v1/captures/{id}/sessions
  POST /api/v1/analyses          existing envelope, record = session JSON
       -> persist analysis with capture_id + session_id

frontend history
  GET  /api/v1/captures                     list jobs
  GET  /api/v1/captures/{id}                job + session previews + analysis summaries
  GET  /api/v1/analyses?capture_id=…        existing list, new filter
```

Preview row (“what is being checked”) is **not** the ML result. It is the extracted feature snapshot:

```text
CaptureSessionPreview {
  session_id, protocol, src_port, dst_port,
  tls_version, cipher_suite, starttls_advertised, starttls_used,
  handshake_success, cert_present, cert_expired, hostname_mismatch,
  evidence_refs, checked_views: ["protocol_session", "tls", "certificate"]
}
```

---

## Phase 1 — Postgres mandatory (ship alone)

**Files:** `api/config.py`, `api/dependencies.py`, `compose.yaml`, `.env.example`, `deploy/vps/env.example`, `migrations/env.py`, tests stay on SQLite via env override.

**Interfaces:**

- Runtime: `DATABASE_URL` must be `postgresql+asyncpg://…` in Docker.
- Tests: `DATABASE_URL=sqlite+aiosqlite:///<tmp>` set in pytest fixture (current implicit default).
- Boot: keep `Base.metadata.create_all` **or** `alembic upgrade head` — pick **create_all + Alembic in sync**, do not run two competing schemas. Prefer Alembic on Postgres; tests can still `create_all` on SQLite.

- [ ] **Step 1:** Default `Settings.DATABASE_URL` to a Postgres URL for documented deploy; pytest fixture forces SQLite so `tests/test_api.py` stays offline.
- [ ] **Step 2:** Confirm JSON columns work on both dialects (they already use `sqlalchemy.JSON`).
- [ ] **Step 3:** Document `.env`: `POSTGRES_PASSWORD`, `DATABASE_URL` for host-run API against `127.0.0.1:5432`.
- [ ] **Step 4:** Verify: `POSTGRES_PASSWORD=… SECUREMAIL_API_KEY=… docker compose up -d` then API health; `uv run pytest tests/test_api.py`.

**Done when:** compose API talks only to Postgres; local pytest still green without Docker.

**Do not:** drop `aiosqlite` yet; do not rewrite Alembic to async.

---

## Phase 2 — TShark extract + capture preview

**Files:** Create `api/capture_routes.py`, `tests/test_capture_api.py`. Modify `ml/pcap.py`, `api/database.py`, `api/schemas.py`, `api/app.py`, `Dockerfile`.

**Interfaces:**

```python
def detect_mail_ports(pcap_path: Path) -> list[tuple[Protocol, int]]:
    """Return unique (protocol, dest_port) pairs among {25,465,587,143,993,110,995}."""

def extract_authorized_capture(
    pcap_path: Path,
    *,
    environment_id: str,
) -> list[SessionFeatureRecord]:
    """Call detect_mail_ports, then extract_sessions per pair with
    source_type=authorized_capture and a neutral ScenarioManifest
    (risk_label=informational, no client labels)."""
```

HTTP:

```text
POST /api/v1/captures
Content-Type: multipart/form-data
field: file  (application/vnd.tcpdump.pcap or application/octet-stream)

201 CaptureResponse {
  schema_version: "capture-response.v1"
  capture_id: string
  status: "extracted" | "empty" | "failed"
  pcap_sha256: string
  filename: string          # basename only
  session_count: int
  sessions: CaptureSessionPreview[]
  diagnostics: map<string, list<string>>
}

GET /api/v1/captures/{capture_id}/sessions
200 { sessions: CaptureSessionPreview[], records: session-features.v1[] }
  # records needed so the frontend can POST /analyses 1:1 without re-upload
```

ORM (Postgres):

```text
capture_jobs
  id PK, capture_id unique, created_at, filename, pcap_sha256,
  status, session_count, diagnostics JSON, error_code nullable

capture_sessions
  id PK, capture_id FK, session_id, protocol, dest_port,
  preview JSON, record JSON, analysis_request_id nullable
```

- [ ] **Step 1:** Failing test: POST tiny non-pcap → 422; POST fixture pcap (use existing lab capture if present, else a committed tiny SMTP pcap under `tests/fixtures/`) → 201 with `session_count >= 1` and preview fields populated. Mock `extract_sessions` in unit tests so CI does not require tshark; one optional `@pytest.mark.integration` hits real tshark.
- [ ] **Step 2:** Implement `detect_mail_ports` + `extract_authorized_capture`. No new parser.
- [ ] **Step 3:** Temp file under `/tmp/securemail-captures/`, hash, extract, insert rows, unlink file in `finally`.
- [ ] **Step 4:** Auth same helper as analyses. Size cap 20 MiB. Reject forbidden multipart fields.
- [ ] **Step 5:** Dockerfile: `apt-get install tshark` (noninteractive). Compose API already `read_only` + `tmpfs /tmp` — captures must use `/tmp`.
- [ ] **Step 6:** If no mail ports / no sessions: `status=empty`, 201 still (not a 500). TShark missing: 503 `extractor_unavailable`.

**Done when:** one PCAP in → session JSON + preview out; PCAP file gone; no payload keys in stored JSON.

---

## Phase 3 — One-by-one analysis + grouped history

**Files:** `api/routes.py` (persist `capture_id`), `api/data_routes.py` (filter), `api/capture_routes.py` (job history), `api/database.py` (`AnalysisRecord.capture_id`).

**Interfaces:**

```text
POST /api/v1/analyses
  AnalysisRequest stays the same.
  Optional header or body field: capture_id  (analysis-request.v1 additive, default null)

GET /api/v1/captures
  paginated job list: capture_id, filename, created_at, session_count, analyzed_count, status

GET /api/v1/captures/{capture_id}
  job + sessions with analysis summary (request_id, risk.class, action, finding_ids) or null

GET /api/v1/analyses?capture_id=
  existing list filter
```

Linking rule: after a successful analysis, if `record.provenance.capture_id` matches a job, set `capture_sessions.analysis_request_id` and `analysis_records.capture_id`. Frontend should send the record returned by GET sessions unchanged.

- [ ] **Step 1:** Test: create capture (mocked extract, 2 sessions) → POST analyses twice → GET capture shows 2/2 analyzed with distinct `request_id`s.
- [ ] **Step 2:** Add `capture_id` column + index on `analysis_records`.
- [ ] **Step 3:** Apply auth dependency to capture + analysis list/get (not health). Leave synthetic insert for tests or protect it too.
- [ ] **Step 4:** History empty state: job with `analyzed_count=0` is valid.

**Done when:** frontend can drive N sessions as N analysis calls and reload a per-PCAP history page from Postgres.

---

## Deferred (later endpoint pass — do not build now)

- Batch `POST /analyses` that infers all sessions in one call
- Storing PCAP bytes / packet viewer
- Live progress websocket
- Capture-scope ACL (403) beyond API key
- Bearer token parsing
- Dropping SQLite from pytest
- POP3/IMAP port autodetect beyond the fixed port set
- Frontend app

---

## Spec coverage

| Requirement | Phase |
|---|---|
| Postgres-only runtime | 1 |
| One PCAP → TShark JSON | 2 |
| Preview of what is checked | 2 |
| Frontend sends 1 session / analysis | 3 |
| History by capture | 3 |
| Privacy / no bodies / no keylogs | 2 (reuse `inference_record` on analyze) |
| Rules-first ML result | 3 (existing `predict_session`) |
| Dashboard spec “no PCAP upload” | **intentionally overridden**, bounded |

## Assumptions (say if wrong)

1. Server extracts; browser never runs TShark.
2. 20 MiB cap, hash-only retention.
3. pytest stays on SQLite; Docker/VPS is Postgres.
4. Existing `POST /analyses` is the scorer; we do not add a second fusion path.
