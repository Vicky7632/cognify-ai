# Backend

The backend is a standalone Python/FastAPI service using Pydantic Settings, SQLAlchemy, and PostgreSQL. Python **3.12.x** is the project target; Python 3.12.10 was available when this setup was verified. Direct runtime/test dependencies and their resolved transitive dependencies are pinned in [`requirements.txt`](requirements.txt) for repeatable pip installs. Pip manages a backend-local virtual environment; no additional dependency manager is required.

The backend has a minimal FastAPI health route, document metadata and lifecycle APIs, multipart document upload, a durable PostgreSQL-backed document-processing job queue with a separate worker, local PDF/TXT extraction with PostgreSQL segment persistence, a Gemini embedding provider, Pinecone vector upsert and retrieval, and a grounded answer service/API. PostgreSQL holds relational metadata, segment text, and job state; Pinecone contains vectors and searchable identifiers/metadata. Original files remain under the configured local upload root. Media transcription is not implemented.

## Windows PowerShell setup

Run from the repository root:

```powershell
cd E:\Cognify\backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation under the current execution policy, use the environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To verify the installed direct dependencies:

```powershell
python -c "import fastapi, pydantic_settings, pytest, uvicorn; print('FastAPI', fastapi.__version__); print('Pydantic Settings', pydantic_settings.__version__); print('pytest', pytest.__version__); print('Uvicorn', uvicorn.__version__)"
```

Run backend tests from `E:\Cognify\backend`:

```powershell
python -m pytest
```

Start the development server from `E:\Cognify\backend`:

```powershell
python -m uvicorn app.main:app --reload
```

The health check is available at `http://127.0.0.1:8000/api/v1/health`; FastAPI's interactive API documentation remains available at `http://127.0.0.1:8000/docs`.

The current tests cover settings behavior, lazy database/session behavior, `Document` model metadata, repository and service operations, upload/storage failures, PDF/TXT extraction and failure cases, lifecycle logging, and API behavior. Database tests use isolated SQLite and do not require a live PostgreSQL server.

The document repository uses the session supplied by its caller and does not commit or roll back. Callers own transaction boundaries. Unit tests exercise repository operations against an isolated in-memory SQLite database and do not require PostgreSQL.

`app/services/document_service.py` validates supported document file types and processing statuses, then delegates persistence operations to the repository. It accepts the caller-provided session and does not commit or roll back; transaction ownership remains with the caller.

## Document metadata API

`POST /api/v1/documents/upload` accepts one multipart field named `file` and uploads a PDF, TXT, MP3, WAV, or MP4. The maximum size defaults to **25 MiB** and is configurable with `COGNIFY_MAX_UPLOAD_SIZE_BYTES`. The local storage root defaults to the repository's `storage/uploads` directory and is configurable with `COGNIFY_UPLOAD_ROOT`. Stored names are random server-generated identifiers; original filenames are metadata only. Responses exclude storage paths and keys.

The request must contain exactly one multipart field, `file`, and exactly one file. Filenames must be 1–255 characters, contain no path separators/control characters, and use a supported extension. Unicode filenames are supported.

Example from the repository root:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/upload -F "file=@lecture.pdf"
```

Successful uploads return `201 Created` with document ID, original filename, file type, processing status, and timestamps. Invalid/missing files return `422`, unsupported extensions return `415`, size-limit violations return `413`, and storage/database failures return generic `500` responses without internal paths or exception details. File type checks use basic signatures (and strict UTF-8 decoding for TXT) in addition to the extension; they are not substitutes for format-specific parsers or malware scanning.

Uploads are streamed in bounded chunks to a temporary-unique server-generated path, with exclusive creation and partial-file cleanup. The API stores the document row and a durable queued processing job in the same database transaction; it does not wait for extraction, Gemini, or Pinecone. A confirmed rollback triggers best-effort deletion of the uploaded file. If the transaction body completed but commit raises, the outcome is treated as ambiguous and the file is retained so a committed document/job cannot point to a deleted file. Such uncertain outcomes can leave orphaned files; automated reconciliation/retention is future work. The filesystem and PostgreSQL do not share a distributed transaction, so abrupt process termination or cleanup failure can also leave an orphaned file. No file-serving or download route is provided.

Apply all Alembic migrations before using upload processing, then start the API and worker as separate processes. From `E:\Cognify\backend`, start the worker with:

```powershell
python -m app.document_worker
```

The worker claims durable jobs in PostgreSQL, retries failures up to three attempts, and holds a PostgreSQL advisory lock so only one worker instance processes jobs at a time. A separate monitor checks that lock ownership remains present, and synchronous ownership checks guard job/status writes and processing-stage boundaries. These checks narrow but do not eliminate the race between verifying the lock and performing a database write or starting an external request; the lock is not atomic with those operations. Loss detected during an operation aborts further stages and suppresses failure/retry writes by the old worker. An already in-flight provider request cannot be cancelled by this mechanism. At startup the worker recovers interrupted jobs and queues legacy `uploaded` documents that have no job. Processing is currently supported for PDF/TXT; uploaded audio/video files are accepted by validation but their processing is not implemented and will fail safely. Documents become `completed` only after extraction, segment persistence, embedding, and Pinecone upsert all succeed. The status API cannot set `processing` or `completed`; those transitions are managed by the worker. There is no distributed transaction across PostgreSQL and Pinecone: a partial remote upsert may leave vectors behind, and the current vector-store protocol cannot delete stale vectors after a shorter reprocessing run. Such stale vectors can occupy retrieval slots and cause abstention. Vectors carry a SHA-256 fingerprint of the exact segment text used to generate them; grounded answers exclude a vector if its fingerprint does not match the current PostgreSQL text, preventing stale vectors from being cited against changed text. Existing vectors without fingerprints must be reindexed before they can be cited.

The `/api/v1/documents` routes create, list, retrieve, update status, and delete document metadata. `GET /api/v1/documents/{document_id}/segments` returns persisted extracted segments in ordinal order with `limit` (1–100, default 50) and `offset` (minimum 0, default 0); its response includes the total segment count and each segment's source provenance. Missing documents return `404`, while documents with no segments return an empty `200` response. Metadata creation does not accept a client-provided filesystem path. Response schemas exclude `storage_path`.

Uploaded documents begin in `uploaded`; metadata-only records retain `pending`. The processing worker controls transitions to `processing` and `completed`; the status API returns `409 Conflict` if a caller requests either status. Worker-managed transitions include `pending`/`uploaded` to `processing`, `processing` to `failed`, and failed-job retries through the durable queue. Completed documents are terminal. Failed processing stores only the fixed `processing_failed` error code; retry clears it. Invalid transitions return `409 Conflict`. Changing status through this API does not enqueue or run processing.

Mutation routes own the request transaction and commit only on success; errors roll back. Repository and service methods do not commit. Status transitions use a conditional update against the current status to prevent stale requests from overwriting a newer state. The service does not have authentication or document-level authorization. Keep it restricted to a trusted local environment; do not expose it publicly until suitable access controls are implemented.

The segment retrieval endpoint also has no authentication or ownership checks; it is not ownership-protected. Restrict it to a trusted local environment until suitable access controls are implemented.

## Processing foundation

`app/services/document_processor.py` defines a synchronous processor protocol and immutable structured result segments with optional page or media-time provenance. The processor contract accepts no FastAPI request or database session objects. `app/services/document_processing_service.py` coordinates one injected processor: it claims `processing` in a short transaction, runs the processor without holding a database transaction, then conditionally records completion or a safe failure status. Stale transitions are rejected through the existing compare-and-set lifecycle service.

The local processor extracts PDF text page by page with 1-based page provenance and decodes TXT as strict UTF-8 with optional BOM support. It resolves only generated storage keys beneath the configured upload root. Missing, unreadable, malformed, unsupported, or empty inputs fail with the existing safe `processing_failed` code. The document worker persists segments in ordinal order, closes database sessions before embedding/upserting, and marks completion only after every stage succeeds. Failed attempts retain persisted segments for deterministic-ID reprocessing; the worker retries at most three times. Apply the document-segment and processing-job Alembic revisions before using this flow against a database.

## Embedding and vector storage

`app/services/embedding_service.py` defines the `EmbeddingProvider` protocol and an `EmbeddingService` wrapper. The Gemini adapter in `app/services/gemini_embedding_provider.py` uses Google's official `google-genai` Python SDK with `gemini-embedding-001`. Adapters receive non-empty text batches and a configured model, return exactly one numeric vector per input in the same order, and declare their maximum batch size. The service splits larger inputs at that limit without dropping or reordering texts. Gemini requests contain at most 100 texts. It rejects empty/invalid text, malformed or count-mismatched outputs, non-numeric or non-finite values, and inconsistent vector dimensions. Provider exceptions are replaced with safe embedding exceptions; raw provider exception messages are not exposed.

The defaults are `EMBEDDING_PROVIDER=gemini`, `EMBEDDING_MODEL=gemini-embedding-001`, and `EMBEDDING_DIMENSIONS=3072`. [Gemini's official embedding documentation](https://ai.google.dev/gemini-api/docs/embeddings) lists 3072 as the model's default output dimension and supports setting `output_dimensionality`; the adapter sends the configured dimension and validates the returned vector size. Override it only with a dimension supported by the model and keep it consistent for all vectors stored/searchable together. Configure `GEMINI_API_KEY` in the repository-root `.env` or process environment; `.env.example` contains a placeholder only. The key is stored as a Pydantic `SecretStr`, is not logged or returned, and provider errors do not expose SDK exception text. App startup does not construct the SDK client; the factory requires a configured key when creating the Gemini provider. Close the returned embedding service when finished to release SDK HTTP resources. The adapter uses Google's [official `google-genai` Python SDK](https://googleapis.github.io/python-genai/).

`app/vector_store/protocol.py` defines the provider-neutral vector storage types and upsert/search interface. `app/vector_store/pinecone_adapter.py` uses the official Pinecone Python SDK to validate records, derive deterministic UUID5 IDs from document/segment IDs, verify the configured index dimension, and upsert in batches while preserving order. Its similarity search validates query vector values/dimensions, supports positive configurable `top_k` and an optional document UUID filter, and returns typed matches with available metadata. It does not create indexes. The pre-created index dimension must match `EMBEDDING_DIMENSIONS`; no index name or dimension is hardcoded. `PINECONE_API_KEY`, `PINECONE_INDEX`, and `PINECONE_UPSERT_BATCH_SIZE` configure the adapter. The API key uses `SecretStr`; Pinecone configuration is optional at application startup and required only when the adapter is constructed. No live Pinecone index has been verified.

`app/services/document_retrieval_service.py` embeds a non-empty text query using the injected embedding service and sends the vector to the injected store. `POST /api/v1/retrieval/search` exposes this service with a bounded `top_k` (1–100, default 10) and optional document UUID filter. It returns typed segment identifiers, ordinal, similarity score, and stored metadata. Empty results are returned as an empty list. Invalid requests return `422`, missing provider/index configuration returns a generic `503`, and provider failures return a generic `502`. The endpoint does not generate answers; grounded answer generation is available separately at `/api/v1/retrieval/answer`. Deletion and embedding lifecycle synchronization are not implemented.

`app/services/grounded_answer_service.py` combines retrieval results with the matching rows loaded from PostgreSQL using their segment IDs. Only non-empty stored segment text with matching document/ordinal records enters model context. The prompt treats document content as untrusted data and requests a fixed insufficient-evidence token when context is inadequate. The model generates answer text only; citations are assembled from database segment/document rows and the retrieved scores. Missing records or unusable content produce an explicit insufficient-evidence response without calling the model.

`POST /api/v1/retrieval/answer` accepts `query`, `top_k` (1–100, default 10), and optional `document_id`. It returns the answer, verified citations (document ID, segment ID, ordinal, available page number, filename, file type, and retrieval score), and an `insufficient_evidence` flag. The Gemini generation adapter uses the already-installed `google-genai` SDK and existing `GEMINI_API_KEY`, but deliberately has no default LLM model. Configure `LLM_PROVIDER=gemini` and an explicitly selected supported `LLM_MODEL`; absent configuration returns a generic `503`. Confirm the model and applicable usage/privacy terms before use. Tests mock retrieval, database results, and generation; no live provider calls were made.

Automated tests mock SDK responses and inject fake embedding, retrieval, persistence, and generation implementations; they do not require credentials, internet access, or paid usage. No live Gemini API request or Pinecone index has been verified. Check Google's and Pinecone's current account, pricing, and usage terms before use.

## Document-to-embedding pipeline

`app/services/document_embedding_service.py` loads all stored segments for an existing document in ascending ordinal order and submits their text to the configured embedding service. It returns an ordered tuple of `EmbeddedDocumentSegment` values containing the document ID, segment ID, ordinal, and vector. Missing documents, documents with no segments, malformed segment data, and provider/output failures are surfaced as explicit errors. The embedding service performs provider-bounded batching and validates output count, finite values, and dimensions; the pipeline additionally validates result count and consistent dimensions before associating vectors with segments.

The Day 26 `DocumentVectorStorageService` uses an injected `VectorStore` and upserts generated vectors after mapping IDs and ordinals. Day 32 connects extraction, committed segment persistence, embeddings, and Pinecone upsert through a PostgreSQL-backed durable worker; upload enqueues but does not process inline. A successful indexing job means all configured batched upserts returned successfully; there is no distributed transaction or cleanup strategy for a partial remote upsert failure or stale vectors after shorter reprocessing. The Day 27 `DocumentRetrievalService` vectorizes queries and retrieves typed Pinecone matches with optional document filtering. Day 29 loads matched segment text and document metadata from PostgreSQL, builds untrusted context, generates a grounded answer through an injected provider, and assembles citations from persisted records. The complete segment set is loaded into memory for embedding; very large-document streaming is future work. A valid Gemini key, explicitly configured answer model, and pre-created dimension-compatible Pinecone index are required for live answer requests. Tests use fakes and make no external calls. Apply all Alembic migrations before running document pipelines against a database.

## Database migrations

The SQLAlchemy model defines metadata; Alembic migrations are the only supported workflow for changing database schemas. From `E:\Cognify\backend`, review the generated revision before applying it:

```powershell
python -m alembic current
python -m alembic revision --autogenerate -m "describe schema change"
python -m alembic upgrade head
python -m alembic current
```

Alembic reads the existing `DATABASE_URL` from application settings. It does not print the URL. Use the configured local development database and inspect existing tables and migration history before the first upgrade. Review every generated revision for unexpected operations. Do not use `create_all()`, `drop_all()`, or `stamp head` as a substitute for migrations. Tests verify migration configuration/model registration without requiring a database.

## Logging

The application uses Python's standard logging module for fixed startup and shutdown lifecycle messages. It does not log configuration values, request bodies, uploaded content, credentials, or API keys. Configure logging output through the server/runtime environment; no additional logging package is required.

## Configuration and safety

The minimal settings module reads `COGNIFY_APP_NAME`, `COGNIFY_API_V1_PREFIX`, `COGNIFY_UPLOAD_ROOT`, `COGNIFY_MAX_UPLOAD_SIZE_BYTES`, optional `DATABASE_URL`, embedding settings, and optional LLM settings from process environment variables or the repository-root `.env`; environment variables take precedence. Names are case-insensitive, and unrelated variables are ignored. Embedding defaults are `EMBEDDING_PROVIDER=gemini`, `EMBEDDING_MODEL=gemini-embedding-001`, and `EMBEDDING_DIMENSIONS=3072`; `GEMINI_API_KEY` is optional at application startup and required only when constructing the Gemini provider. Relative upload-root values are resolved from the repository root. Configured API prefixes must start with `/` and must not end with `/`. Runtime settings are cached once.

`DATABASE_URL` is optional for application startup and the health endpoint. When a route later requests a database session, the cached SQLAlchemy engine is created lazily from `DATABASE_URL`; if it is absent, the database dependency raises an explicit configuration error. Session cleanup is handled by the `get_db` dependency. The engine uses `pool_pre_ping`; no connection is opened merely by importing the app or constructing the engine.

The root `.env.example` contains placeholder values, including a sample local PostgreSQL URL, and is not loaded automatically. For temporary PowerShell overrides, set `$env:COGNIFY_APP_NAME`, `$env:COGNIFY_API_V1_PREFIX`, `$env:COGNIFY_UPLOAD_ROOT`, `$env:COGNIFY_MAX_UPLOAD_SIZE_BYTES`, `$env:DATABASE_URL`, `$env:EMBEDDING_PROVIDER`, `$env:EMBEDDING_MODEL`, `$env:EMBEDDING_DIMENSIONS`, `$env:GEMINI_API_KEY`, `$env:LLM_PROVIDER`, `$env:LLM_MODEL`, `$env:PINECONE_API_KEY`, `$env:PINECONE_INDEX`, or `$env:PINECONE_UPSERT_BATCH_SIZE` before starting the server; the override applies to that PowerShell session and its child processes. Alternatively, add local settings to the repository-root `.env`. Do not commit `.env`, print credentials, or put backend secrets in frontend variables. The root `.gitignore` excludes `.env` and `.venv` while allowing `.env.example` to be tracked. Pinecone and Gemini credentials are not needed to import settings, start the health endpoint, or run tests. Configure Pinecone only when live vector storage is needed; the index must already exist and match `EMBEDDING_DIMENSIONS`. Answer generation additionally requires `LLM_PROVIDER=gemini`, an explicitly selected `LLM_MODEL`, and `GEMINI_API_KEY`; the project does not choose a default generation model.

The tests do not create tables or require a live PostgreSQL server. The initial migration was applied and verified against the configured local `cognify_ai` database during this milestone.

The initial MVP is unauthenticated and must not be exposed publicly without suitable access controls. No live PostgreSQL connection, Pinecone service, paid service, or migrations are required to run the health endpoint. Pinecone index configuration has not been verified against a live service.

## Week 1 milestone review

### Implemented

- Python 3.12 backend tooling with pinned FastAPI, Uvicorn, Pydantic Settings, pytest, and TestClient dependencies.
- FastAPI application factory with a configurable application title and versioned API prefix.
- `GET /api/v1/health`, returning HTTP 200 and exactly `{"status":"ok"}`; default FastAPI `/docs` remains enabled.
- Pydantic Settings defaults, case-insensitive environment loading, unrelated-variable ignoring, repository-root `.env` path resolution, and cached runtime settings.
- API prefix validation and fixed startup/shutdown lifecycle logs that do not include configuration or request data.
- Automated tests for settings defaults/overrides/dotenv loading, valid and invalid prefixes, health behavior, and lifecycle logging.

### Not implemented

At the Week 1 review, PostgreSQL connection/session management, SQLAlchemy models, migrations, document persistence, file upload or extraction, Pinecone, embeddings, retrieval, generation, transcription, frontend screens, authentication, and deployment infrastructure were not yet implemented.

### Next milestone

The initial Alembic migration created and verified the `documents` table in the configured local database. Next, build the document persistence/use-case layer on this schema; keep relational metadata separate from Pinecone vector storage and original-file storage.
