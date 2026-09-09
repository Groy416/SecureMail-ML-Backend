# SecureMail API

Base URL: `http://<host>:8000/api/v1`

Interactive OpenAPI: `/docs` · machine-readable schema: `/openapi.json`.

## Authentication

Set `SECUREMAIL_API_KEY` for every deployment. All endpoints except `GET /health`
require the raw key (not `Bearer`) in the `Authorization` header:

```http
Authorization: <SECUREMAIL_API_KEY>
```

The dashboard keeps this key server-side in `SECUREMAILSCOPE_API_KEY`; never
publish it through `NEXT_PUBLIC_*` variables.

## Readiness and model metadata

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Public process/runtime readiness. `status=healthy` only when the bundle loaded. |
| GET | `/bundle` | Active bundle version, feature names, and participating models. |
| GET | `/rules` | Deterministic finding catalog. |

## Analyze one session

`POST /analyses` accepts exactly one `analysis-request.v1` envelope containing
one `session-features.v1` record. The API validates privacy/schema constraints,
runs deterministic rules plus ML inference, persists the result, and returns
`analysis-response.v1`.

```bash
curl -X POST "$API_URL/analyses" \
  -H "Authorization: $SECUREMAIL_API_KEY" \
  -H 'Content-Type: application/json' \
  --data @analysis-request.json
```

Successful responses are `200` with `status` `complete` or `degraded`.
Rejected input is `401` or `422`; unavailable model runtime is `503`; inference
or persistence failure is `500`. Clients must treat `result.risk.class` as the
authoritative final class. Rules remain the policy floor.

`POST /validate` runs the same privacy/schema checks without inference. It
returns `validation-response.v1` and persists its validation audit row.

## PCAP extraction jobs

`POST /capture-jobs` accepts one authorized `.pcap` or `.pcapng` multipart upload
in the `file` field. The compatibility path `POST /captures` has the same
contract. The API validates PCAP/PCAPNG magic values and streams up to **20 MiB**
into a private transient Docker volume; it never stores PCAP bytes in Postgres.

```bash
curl -X POST "$API_URL/capture-jobs" \
  -H "Authorization: $SECUREMAIL_API_KEY" \
  -F 'file=@authorized-capture.pcap;type=application/vnd.tcpdump.pcap'
```

The response is `202 capture-job.v1`:

```json
{
  "schema_version": "capture-job.v1",
  "job_id": "pcap-0123abcd0123abcd",
  "capture_id": "pcap-0123abcd0123abcd",
  "status": "queued",
  "pcap_sha256": "...",
  "filename": "authorized-capture.pcap",
  "attempts": 0,
  "session_count": 0,
  "processed_sessions": 0,
  "progress": null,
  "error_code": null,
  "diagnostics": {}
}
```

The job ID is deterministic from the PCAP SHA-256. Re-uploading identical bytes
returns the existing job and does not enqueue duplicate work. A Postgres-backed
worker claims queued jobs using `FOR UPDATE SKIP LOCKED`, runs server-side
TShark, persists only safe previews and validated `session-features.v1` JSON,
and deletes the transient PCAP. The default extraction timeout is 120 seconds
and each job has at most two worker attempts. A retryable failure returns to
`queued`; terminal failures have `status=failed` and a non-secret `error_code`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/capture-jobs` | Paginated job list (`skip`, `limit`). |
| GET | `/capture-jobs/{job_id}` | Poll job status. |
| GET | `/capture-jobs/{job_id}/sessions` | Paginated extracted previews and exact session records. |
| GET | `/capture-jobs/{job_id}/results` | Paginated persisted analysis results for the job. |

Job statuses are `queued`, `running`, `complete`, `empty`, and `failed`.
`complete` means extraction produced session records; `empty` means no supported
SMTP/IMAP/POP3 sessions were found. `progress` is `null` until extraction
starts, `0` while running, and `100` after terminal extraction. It is not an
ETA.

Malformed or unsupported uploads return `422`; an upload over 20 MiB returns
`413`. A missing TShark/parser dependency is represented by a worker failure
and bounded retry, not by fabricated sessions. Mail bodies, credentials,
private keys, and TLS key logs must not be uploaded or stored.

The frontend must wait for a terminal job, retrieve `/sessions`, and submit
those records to `POST /analyses` **one record per request**. Session retrieval
does not run inference, and PCAP files must never be sent to `/analyses`.

## Persisted analysis history

| Method | Path | Purpose |
|---|---|---|
| GET | `/analyses` | Paginated persisted analysis history. |
| GET | `/analyses/stats` | Dashboard aggregates. |
| GET | `/analyses/{request_id}` | One persisted result. |
| DELETE | `/analyses/{request_id}` | Delete one persisted result. |
| DELETE | `/analyses?confirm=true` | Delete all analysis rows; `synthetic_only=true` restricts the operation. |
| GET | `/validations` | Paginated persisted validation audit rows. |

`GET /analyses` supports `skip`, `limit` (1–200), `verdict`, `session_id`,
`capture_id`, `protocol`, `posture`, `is_synthetic`, and inclusive ISO-8601
`from` / `to` filters. Example:

```text
GET /analyses?capture_id=pcap-0123abcd&protocol=SMTP&from=2026-09-01T00:00:00Z
```

Analysis responses and persisted analysis records also include nullable
`tls_details` and `certificate_details` objects when the authorized PCAP
contained the corresponding evidence. These are derived from negotiated TLS
and parsed X.509 metadata; unavailable evidence is `null`, never fixture data.
`GET /capture-jobs/{job_id}/sessions` exposes the same objects in both the
session previews and the validated `records` payloads.

`GET /analyses/stats` accepts `capture_id`, `protocol`, `from`, and `to`.
Its authoritative dashboard fields are `total_analyses`, `flagged_sessions`,
`evidence_archived`, `avg_risk_score`, `verdict_distribution`, and
`cryptographic_posture_distribution`. A missing aggregate is `null`/empty, not
a safe or zero-risk result.

## Synthetic fixture endpoint

`POST /analyses/synthetic` inserts an explicitly synthetic row for local UI
fixtures. It does not run inference. Do not use it to populate production
dashboard data.

## Current boundaries

- Only authorized captures may enter the capture-job endpoints.
- The queue reports extraction progress only; it does not invent ETAs or model
  metrics, and it does not provide a packet-payload viewer.
- The browser must not run TShark or send PCAPs to `/analyses`; the dashboard
  uses same-origin server routes so the API key remains server-side.
- Session extraction and analysis are separate: upload → poll → retrieve
  records → submit one record to `/analyses`.
