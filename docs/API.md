# API

> **Proposed, not implemented.** All endpoints use `/api/v1`. Paths, schemas, HTTP semantics, limits, and error format must be confirmed during implementation.

## Proposed endpoints

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/api/v1/health` | Basic process health; readiness semantics to define | Proposed |
| `POST` | `/api/v1/documents` | Upload one PDF, TXT, MP3, WAV, or MP4 | Proposed |
| `GET` | `/api/v1/documents` | List documents (pagination/filter contract to define) | Proposed |
| `GET` | `/api/v1/documents/{document_id}` | Read document metadata and processing status | Proposed |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete metadata, vectors, and original according to defined lifecycle | Proposed |
| `POST` | `/api/v1/chat/sessions/{session_id}/messages` | Ask a question; return answer/abstention and citations | Proposed |
| `POST` | `/api/v1/queries` | Optional stateless question route; whether needed remains open | Proposed |
| `POST` | `/api/v1/documents/{document_id}/summaries` | Request short or detailed source-grounded summary | Proposed |
| `POST` | `/api/v1/documents/{document_id}/study-notes` | Request topic-wise notes and key concepts | Proposed |
| `GET` | `/api/v1/documents/{document_id}/transcript` | Read transcript after successful processing | Proposed |

No asynchronous job model, persistence model, or status lifecycle has been selected. The ingestion endpoint may need to return a job identifier if processing is asynchronous; settle this before implementation.

## Proposed request/response shape

### `POST /api/v1/documents`

Multipart upload of one PDF, TXT, MP3, WAV, or MP4 file. Validate actual content where practical, not only extension or client MIME type. Enforce configured upload limits. Return a document identifier and clear processing status; do not imply transcription or indexing succeeded until those stages do.

### `GET /api/v1/documents/{document_id}`

Return document metadata and current processing status. Define whether failures expose a stable error code, retryability, and stage details without exposing secrets or extracted private content.

### `POST /api/v1/chat/sessions/{session_id}/messages`

Proposed JSON fields:

```json
{
  "question": "What is the main conclusion?",
  "document_ids": ["document-id"]
}
```

`document_ids` and all other options are illustrative. Validate bounds and use server-side configured defaults. Proposed response:

```json
{
  "status": "answered",
  "answer": "The retrieved evidence supports ...",
  "citations": [
    {
      "document_id": "document-id",
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

The proposed source-scoped endpoints should return generated content plus source references where feasible. Provide short/detailed summary choices, topic-wise notes, key concepts, and transcript segments/timestamps where available. Generation must be limited to the selected source and must not imply unsupported facts.

## Proposed errors and validation

Use explicit, stable error responses for invalid file type, upload too large, malformed/empty content, unknown source/ingestion, unavailable external provider, and internal failure. Select a common error envelope and status codes during implementation. Never include API keys, raw stack traces, or unnecessary uploaded content in responses.

## Security and compatibility

V1 is planned without authentication and is not safe for public exposure. Add suitable access controls before exposing it beyond a trusted local environment. Keep Cognify AI independently runnable; future LectureIQ integration should call this service through LectureIQ's Express backend and establish service authentication, timeout/retry, and error mapping separately.
