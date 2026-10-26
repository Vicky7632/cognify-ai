# API

> **Status:** The health endpoint, document metadata/upload endpoints, paginated document-segment retrieval, semantic vector retrieval, and grounded answer endpoint are implemented. Chat sessions, summaries, and study-note endpoints remain proposals. All routes use `/api/v1`.

## Endpoints

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/api/v1/health` | Basic process health; does not check database readiness | Implemented |
| `POST` | `/api/v1/documents` | Create document metadata (no file upload) | Implemented |
| `POST` | `/api/v1/documents/upload` | Upload one supported file and create metadata | Implemented |
| `GET` | `/api/v1/documents` | List document metadata | Implemented |
| `GET` | `/api/v1/documents/{document_id}` | Read document metadata and processing status | Implemented |
| `GET` | `/api/v1/documents/{document_id}/segments` | Retrieve ordered extracted segments with pagination | Implemented |
| `POST` | `/api/v1/retrieval/search` | Retrieve semantically similar stored segments | Implemented |
| `POST` | `/api/v1/retrieval/answer` | Generate a grounded answer with verified citations | Implemented |
| `PATCH` | `/api/v1/documents/{document_id}/status` | Update processing status | Implemented |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete the metadata record only | Implemented |
| `POST` | `/api/v1/chat/sessions/{session_id}/messages` | Ask a question; return answer/abstention and citations | Proposed |
| `POST` | `/api/v1/queries` | Optional stateless question route; whether needed remains open | Proposed |
| `POST` | `/api/v1/documents/{document_id}/summaries` | Request short or detailed source-grounded summary | Proposed |
| `POST` | `/api/v1/documents/{document_id}/study-notes` | Request topic-wise notes and key concepts | Proposed |
| `GET` | `/api/v1/documents/{document_id}/transcript` | Read transcript after successful processing | Proposed |

No asynchronous job model or ingestion behavior has been implemented. The ingestion endpoint may need to return a job identifier if processing is asynchronous; settle this before implementation.

## `POST /api/v1/retrieval/search`

Embeds a text query with the configured Gemini embedding provider and retrieves the nearest vectors from the configured, pre-existing Pinecone index. It does not generate an answer or access segment text from PostgreSQL.

Request JSON:

```json
{
  "query": "What does the material say about retrieval?",
  "top_k": 10,
  "document_id": "d07962f8-c430-4b0b-b4bb-38ca63c1c083"
}
```

`query` is required and must contain 1–10,000 non-whitespace characters. `top_k` is an integer from 1 through 100 and defaults to 10. `document_id` is an optional UUID filter against Pinecone's stored `document_id` metadata.

The `200 OK` response contains the normalized query, selected `top_k`, and a `results` array in Pinecone similarity-score order. Each result includes the deterministic vector ID, document UUID, segment UUID, zero-based ordinal, similarity score, and available stored metadata. If there are no matches, the endpoint returns `200 OK` with an empty `results` array.

Invalid input returns `422 Unprocessable Entity`. Missing or invalid provider/index configuration returns a generic `503 Service Unavailable`. Gemini or Pinecone request failures and malformed provider output return a generic `502 Bad Gateway`; provider details and credentials are not returned.

This endpoint has no authentication or ownership authorization. It is not safe to expose publicly without suitable access controls. A Gemini API key and Pinecone API key/index are required at request time; the index must already exist and match `EMBEDDING_DIMENSIONS`. Tests replace the retrieval dependency with fakes and make no network calls.

## `POST /api/v1/retrieval/answer`

Runs semantic retrieval, loads matching segment text and document metadata from PostgreSQL, then requests a grounded answer from the configured LLM provider. Pinecone metadata is not treated as segment text. Only existing stored segments whose document ID and ordinal match the retrieval result are included as context. The prompt treats retrieved text as untrusted content, and backend citations are assembled from the verified database records rather than model output.

Request JSON:

```json
{
  "query": "What does the material explain?",
  "top_k": 10,
  "document_id": "d07962f8-c430-4b0b-b4bb-38ca63c1c083"
}
```

`query` must contain 1–10,000 non-whitespace characters. `top_k` is an integer from 1 through 100 and defaults to 10. `document_id` is an optional UUID filter.

The response contains `answer`, `citations`, and `insufficient_evidence`. Each citation contains the persisted document and segment UUIDs, ordinal, original filename, file type, and retrieval score; a positive page number is included only when the stored segment has one. When retrieval returns no results, all matching segment rows are missing, or no valid stored text is available, the endpoint returns a clear insufficient-evidence answer with an empty citations list and does not call the LLM.

Invalid input returns `422`; missing provider configuration returns a generic `503`; embedding, Pinecone, or LLM provider failures return a generic `502`; PostgreSQL lookup failures return a generic `500`. Provider credentials and raw exception messages are not included in responses.

The Gemini generation adapter uses the existing `google-genai` SDK and `GEMINI_API_KEY`. No generation model is selected by default. Set `LLM_PROVIDER=gemini` and `LLM_MODEL` to an explicitly selected supported generation model before use. Tests inject fake retrieval, persistence, and generation providers and make no live API calls. The endpoint is unauthenticated and not ownership-protected; restrict it to a trusted local environment.

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

Files are stored under a random server-generated key, never under the client filename. Writes are streamed in bounded chunks, and failed writes are cleaned up when possible. The document row and a durable processing-job row are committed in one PostgreSQL transaction; extraction and provider calls happen later in the separate worker, not within the upload request. A confirmed rollback triggers best-effort deletion of the new file. If the transaction body completed but commit raises, the outcome is treated as ambiguous and the file is retained to avoid leaving committed document/job rows pointing to a deleted file. This may leave an orphaned file; filesystem/database reconciliation is future work.

Run `python -m app.document_worker` from `backend/` as a separate process after applying all Alembic migrations. The worker recovers interrupted jobs, queues pre-existing uploaded documents without jobs, retries failures up to three times, and uses a PostgreSQL advisory lock to prevent multiple worker processes from consuming the same queue concurrently. A monitor and synchronous ownership checks guard job/status writes and stage boundaries. They narrow but do not eliminate the race between checking ownership and a database write or external request; lock ownership is not atomic with those operations. Loss detected during an operation aborts further stages and suppresses old-worker failure/retry writes, but an already in-flight provider call cannot be cancelled. PDF/TXT processing is implemented; media transcription is not, so accepted audio/video uploads currently fail processing safely. Vector metadata includes a SHA-256 fingerprint of the embedded segment text. Grounded answers exclude matches whose fingerprint is absent or does not match current PostgreSQL text; pre-existing vectors need reindexing before they can support citations. The answer/retrieval APIs remain unauthenticated and must not be exposed publicly without suitable access controls.

### `POST /api/v1/documents`

Accepts JSON with `original_filename` (1-255 characters) and `file_type` (`pdf`, `txt`, `mp3`, `wav`, or `mp4`). Returns `201 Created` with the document ID, filename, type, processing status, safe processing error code, and timestamps. A server-generated relative pending storage key is reserved internally; this endpoint does not upload or create a file. Such metadata-only records retain the legacy `pending` status. Clients cannot provide a filesystem path.

### `GET /api/v1/documents/{document_id}`

Returns document metadata with `200 OK`, or `404 Not Found` when the UUID does not identify a record. The list endpoint returns the same metadata shape as an array. Responses never include `storage_path`.

### `GET /api/v1/documents/{document_id}/segments`

Returns persisted extracted segments ordered by their zero-based `ordinal`. Query parameters are `limit` (default `50`, range `1`–`100`) and `offset` (default `0`, minimum `0`). The response includes `total` (all segments for the document), `limit`, `offset`, and the page's `segments`. Each segment includes its IDs, ordinal, text, and optional page or media-time provenance. A missing document returns `404 Not Found`; an existing document with no segments returns `200 OK` with an empty list and `total: 0`. Invalid pagination values return `422 Unprocessable Entity`.

This read endpoint uses the existing database session and document/segment service layers. It does not trigger processing. The service has no authentication or document-level authorization, so segment retrieval is not ownership-protected and must remain restricted to a trusted local environment until suitable access controls are implemented.

### `PATCH /api/v1/documents/{document_id}/status`

Accepts JSON with only a status field, for example `{"status":"processing"}`. Supported lifecycle statuses are `uploaded`, `processing`, `completed`, and `failed`. `pending` remains readable and can transition to `processing` for compatibility with metadata-only records.

Allowed transitions are:

| Current status | Allowed next status |
|---|---|
| `pending` | `processing` (worker only) |
| `uploaded` | `processing` (worker only) |
| `processing` | `failed` (worker or status API); `completed` (worker only) |
| `failed` | `processing` (worker retry only) |
| `completed` | none |

Transitions are conditionally updated against the current database status. The status API rejects attempts to set `processing` or `completed` with `409 Conflict`; only the worker may begin processing or mark a document complete after extraction, segment persistence, embedding, and Pinecone upsert all succeed. A client cannot restart a failed job by changing its document status. Other repeated or disallowed transitions return `409 Conflict`; unknown IDs return `404 Not Found`; unsupported status values return `422 Unprocessable Entity`. A failure transition records only the fixed safe code `processing_failed`; exception messages and stack traces are never stored or returned. Successful response includes `processing_error_code` (`"processing_failed"` for failed documents, otherwise `null`).

```json
{
  "status": "failed"
}
```

This endpoint does not enqueue or invoke processing. Uploaded documents are queued atomically with their metadata by the upload endpoint. The status API does not accept `processing` or `completed` requests; the worker owns both transitions, preventing unqueued processing and premature completion.

### Processing service status

Day 17 added an internal synchronous processor contract; Day 18 added a local PDF/TXT processor; Day 20 Step 3 persists extracted segments. Day 32 adds durable job orchestration: upload creates a PostgreSQL job, and a separate worker extracts and commits ordered segments, then releases the database session before Gemini embedding and Pinecone upsert. Only after upsert succeeds does it mark the document `completed`. PDF extraction creates one segment per non-empty page with 1-based page numbers. TXT decoding is strict UTF-8 with optional BOM support. Storage access uses the validated server-generated storage key under the configured upload root. Empty extraction and unsupported, missing, malformed, or unreadable inputs fail with the existing safe `processing_failed` code. Jobs retry up to three attempts; interrupted jobs are recovered at worker startup. Apply all Alembic revisions before using persistence and indexing against a database.

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
