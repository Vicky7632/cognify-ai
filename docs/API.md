# API

> **Status:** The health endpoint and document metadata endpoints listed as implemented are available. Upload and AI endpoints remain proposals. All routes use `/api/v1`.

## Endpoints

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/api/v1/health` | Basic process health; does not check database readiness | Implemented |
| `POST` | `/api/v1/documents` | Create document metadata (no file upload) | Implemented |
| `GET` | `/api/v1/documents` | List document metadata | Implemented |
| `GET` | `/api/v1/documents/{document_id}` | Read document metadata and processing status | Implemented |
| `PATCH` | `/api/v1/documents/{document_id}/status` | Update processing status | Implemented |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete the metadata record only | Implemented |
| `POST` | `/api/v1/chat/sessions/{session_id}/messages` | Ask a question; return answer/abstention and citations | Proposed |
| `POST` | `/api/v1/queries` | Optional stateless question route; whether needed remains open | Proposed |
| `POST` | `/api/v1/documents/{document_id}/summaries` | Request short or detailed source-grounded summary | Proposed |
| `POST` | `/api/v1/documents/{document_id}/study-notes` | Request topic-wise notes and key concepts | Proposed |
| `GET` | `/api/v1/documents/{document_id}/transcript` | Read transcript after successful processing | Proposed |

No asynchronous job model or ingestion behavior has been implemented. The ingestion endpoint may need to return a job identifier if processing is asynchronous; settle this before implementation.

## Implemented document metadata endpoints

### `POST /api/v1/documents`

Accepts JSON with `original_filename` (1-255 characters) and `file_type` (`pdf`, `txt`, `mp3`, `wav`, or `mp4`). Returns `201 Created` with the document ID, filename, type, processing status, and timestamps. A server-generated relative pending storage key is reserved internally; this endpoint does not upload or create a file. Clients cannot provide a filesystem path.

### `GET /api/v1/documents/{document_id}`

Returns document metadata with `200 OK`, or `404 Not Found` when the UUID does not identify a record. The list endpoint returns the same metadata shape as an array. Responses never include `storage_path`.

### `PATCH /api/v1/documents/{document_id}/status`

Accepts JSON `{"status":"pending"}`, with status one of `pending`, `processing`, `completed`, or `failed`. Returns updated metadata with `200 OK`, `404 Not Found` for an unknown document, and `422 Unprocessable Entity` for invalid input.

### `DELETE /api/v1/documents/{document_id}`

Deletes the metadata record only and returns `204 No Content`, or `404 Not Found` if it does not exist. Original files and vectors are not managed by this endpoint.

All writes execute in an explicit SQLAlchemy transaction at the API boundary; success commits and exceptions roll back. The repository and service do not commit. The API has no authentication or document-level authorization and must remain restricted to a trusted local environment until suitable access controls are implemented.

## Proposed request/response shape

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

The current service has no authentication or document-level authorization and is not safe for public exposure. Restrict it to a trusted local environment until suitable access controls are implemented. Keep Cognify AI independently runnable; future LectureIQ integration should call this service through LectureIQ's Express backend and establish service authentication, timeout/retry, and error mapping separately.
