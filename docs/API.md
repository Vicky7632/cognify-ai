# API

> **Proposed contract only.** No endpoints are implemented. Paths, schemas, HTTP semantics, limits, and error format must be confirmed during implementation and documented as versioned API contracts.

## Proposed endpoints

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/health` | Basic process health | Proposed |
| `POST` | `/v1/ingestions` | Upload one supported file and start/perform ingestion | Proposed |
| `GET` | `/v1/ingestions/{ingestion_id}` | Inspect ingestion status and source identity | Proposed |
| `POST` | `/v1/query` | Retrieve evidence and answer a question with citations or abstention | Proposed |
| `POST` | `/v1/sources/{source_id}/summary` | Generate a source-grounded summary | Proposed |
| `POST` | `/v1/sources/{source_id}/study-notes` | Generate study notes from a source | Proposed |

No asynchronous job model, persistence model, or status lifecycle has been selected. The ingestion endpoint may need to return a job identifier if processing is asynchronous; settle this before implementation.

## Proposed request/response shape

### `POST /v1/ingestions`

Multipart upload of one PDF, TXT, supported audio, or supported video file. Validate actual content where practical, not only extension or client-provided MIME type. Enforce a configured maximum upload size. Return a source/ingestion identifier and a clear status; do not imply transcription or indexing succeeded until those stages do.

### `GET /v1/ingestions/{ingestion_id}`

Return the current status and, on success, the source identifier and relevant ingestion metadata. Define whether failures expose a stable error code, retryability, and stage details without exposing secrets or extracted private content.

### `POST /v1/query`

Proposed JSON fields:

```json
{
  "question": "What is the main conclusion?",
  "source_ids": ["source-id"],
  "top_k": 5,
  "similarity_threshold": 0.7
}
```

`source_ids`, `top_k`, and `similarity_threshold` are illustrative options. Validate bounds and use server-side configured defaults. Proposed response:

```json
{
  "status": "answered",
  "answer": "The retrieved evidence supports ...",
  "citations": [
    {
      "source_id": "source-id",
      "page": 4,
      "start_seconds": null,
      "end_seconds": null,
      "quote": "Relevant supporting text"
    }
  ]
}
```

The response should distinguish an answer from an abstention, such as `status: "insufficient_evidence"` with no unsupported answer. Citation fields should be optional according to source type; page/time values must be omitted or null when unavailable. The final schema and whether to include scores/snippets remain open.

### Summary and study notes

The proposed source-scoped endpoints should return generated content plus source references where feasible. Define length/format options only when needed; generation must be limited to the selected source and must not imply unsupported facts.

## Proposed errors and validation

Use explicit, stable error responses for invalid file type, upload too large, malformed/empty content, unknown source/ingestion, unavailable external provider, and internal failure. Select a common error envelope and status codes during implementation. Never include API keys, raw stack traces, or unnecessary uploaded content in responses.

## Security and compatibility

V1 is planned without authentication and is not safe for public exposure. Add suitable access controls before exposing it beyond a trusted local environment. Keep Cognify AI independently runnable; a future LectureIQ integration should call this service through LectureIQ's Express backend and establish authentication, timeout/retry, and error mapping separately.
