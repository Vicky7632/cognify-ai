# API

> **Status:** The health endpoint, document metadata endpoints, and document upload endpoint are implemented. AI endpoints remain proposals. All routes use `/api/v1`.

## Endpoints

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/api/v1/health` | Basic process health; does not check database readiness | Implemented |
| `POST` | `/api/v1/documents` | Create document metadata (no file upload) | Implemented |
| `POST` | `/api/v1/documents/upload` | Upload one supported file and create metadata | Implemented |
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

### `POST /api/v1/documents/upload`

Accepts exactly one multipart/form-data field named `file`, containing exactly one file. Supported filename extensions are `.pdf`, `.txt`, `.mp3`, `.wav`, and `.mp4`. Filenames must be 1–255 characters and cannot contain path/drive separators or control characters; Unicode is supported. Maximum size defaults to 25 MiB (`COGNIFY_MAX_UPLOAD_SIZE_BYTES`). The upload root defaults to repository-root `storage/uploads` and can be configured with `COGNIFY_UPLOAD_ROOT`.

The service checks the extension and basic file signatures; TXT is decoded as UTF-8. These checks do not fully parse or sanitize formats. There is no malware scanning. A successful upload returns `201 Created` with the document metadata shape below; internal storage paths are not returned. Invalid/empty content returns `422`, unsupported extensions return `415`, oversized files return `413`, and storage/database failures return generic `500` responses.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/upload -F "file=@lecture.pdf"
```

Example success response (`201 Created`):

```json
{
  "id": "d07962f8-c430-4b0b-b4bb-38ca63c1c083",
  "original_filename": "lecture.pdf",
  "file_type": "pdf",
  "processing_status": "uploaded",
  "processing_error_code": null,
  "created_at": "2025-01-01T12:00:00Z",
  "updated_at": "2025-01-01T12:00:00Z"
}
```

Errors use FastAPI's `{"detail":"..."}` shape: invalid filename/content or missing/multiple files return `422`, unsupported extensions return `415`, oversized files return `413`, and storage/database failures return a generic `500`.

Files are stored under a random server-generated key, never under the client filename. Writes are streamed in bounded chunks, and failed writes are cleaned up when possible. Database failure after storage triggers best-effort deletion of the new file. Because filesystem writes and PostgreSQL transactions are not atomic together, process termination or cleanup failure may leave an orphaned file.

### `POST /api/v1/documents`

Accepts JSON with `original_filename` (1-255 characters) and `file_type` (`pdf`, `txt`, `mp3`, `wav`, or `mp4`). Returns `201 Created` with the document ID, filename, type, processing status, safe processing error code, and timestamps. A server-generated relative pending storage key is reserved internally; this endpoint does not upload or create a file. Such metadata-only records retain the legacy `pending` status. Clients cannot provide a filesystem path.

### `GET /api/v1/documents/{document_id}`

Returns document metadata with `200 OK`, or `404 Not Found` when the UUID does not identify a record. The list endpoint returns the same metadata shape as an array. Responses never include `storage_path`.

### `PATCH /api/v1/documents/{document_id}/status`

Accepts JSON with only a status field, for example `{"status":"processing"}`. Supported lifecycle statuses are `uploaded`, `processing`, `completed`, and `failed`. `pending` remains readable and can transition to `processing` for compatibility with metadata-only records.

Allowed transitions are:

| Current status | Allowed next status |
|---|---|
| `pending` | `processing` |
| `uploaded` | `processing` |
| `processing` | `completed`, `failed` |
| `failed` | `processing` (retry) |
| `completed` | none |

Transitions are conditionally updated against the current database status. Repeated or disallowed transitions return `409 Conflict`; unknown IDs return `404 Not Found`; unsupported status values return `422 Unprocessable Entity`. A failure transition records only the fixed safe code `processing_failed`; exception messages and stack traces are never stored or returned. Entering `processing` on retry clears the prior code. Successful response includes `processing_error_code` (`"processing_failed"` for failed documents, otherwise `null`).

```json
{
  "status": "failed"
}
```

This endpoint manages lifecycle state only. No extraction, transcription, AI processing, worker, or queue is implemented in Day 16; callers must only report `completed` after the represented processing has actually succeeded.

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
