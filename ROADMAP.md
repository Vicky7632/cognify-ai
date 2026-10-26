# Roadmap

The roadmap is ordered by dependency. Each phase has a small acceptance gate and depends on the previous phase. Implemented work includes the backend foundation, PostgreSQL document metadata/lifecycle, local upload handling, the Day 17 processor contract/orchestration foundation, and Day 18 PDF/TXT extraction. The statuses below distinguish those implemented parts from remaining work.

## 1. Repository and documentation foundation

- Keep the architecture, API contract, data flow, evaluation, contribution guidance, and scope aligned.
- Confirm the intended license, supported runtimes, and repository conventions.
- **Acceptance:** documentation is consistent, no secrets are exposed, and no features are described as implemented without code.

## 2. Backend skeleton and configuration

- Establish the Python/package/tooling conventions and minimal FastAPI application layout.
- Add typed Pydantic settings and explicit configuration validation; document local setup and health semantics.
- **Acceptance:** the backend starts locally with documented configuration, focused tests cover valid and invalid settings, and no external credentials are required for startup.

## 3. PostgreSQL integration and document metadata

- **Implemented:** SQLAlchemy document metadata, PostgreSQL session setup, Alembic-managed status/error schema, repository/service/API layers, and lifecycle transition tests.
- Keep future relational changes in reviewed Alembic migrations. Keep vectors and original files outside PostgreSQL's relational responsibility boundary.
- **Acceptance:** tests cover persistence, status/error transitions, and database unavailable behavior. Existing backend tests cover persistence and lifecycle behavior.

## 4. File upload and extraction

- **Implemented:** safe local filesystem storage and upload validation; synchronous processor contract/orchestration; PDF page-by-page text extraction with 1-based page provenance; strict UTF-8 TXT decoding with optional BOM support.
- **Implemented (Day 20 Step 3):** the internal synchronous processing service persists extracted segments with zero-based ordinal and source provenance. The Day 32 upload/worker flow now persists segments first and defers completion until indexing succeeds. Extraction runs outside database transactions; persistence failures roll back segment replacement.
- **Implemented (Day 22):** `GET /api/v1/documents/{document_id}/segments` retrieves persisted segments in ordinal order with bounded offset/limit pagination and a total count. The endpoint has no authentication or ownership authorization.
- **Implemented (Day 32):** upload atomically creates a durable PostgreSQL processing job with the uploaded document. A separate singleton worker extracts PDF/TXT content, persists segments, closes database sessions before embedding and Pinecone upsert, and marks completion only after all stages succeed. Jobs retry up to three times; interrupted jobs are recovered and legacy uploaded documents without jobs are queued on worker startup. Apply the processing-job Alembic migration and run `python -m app.document_worker` from `backend/`. Audio/video transcription remains unimplemented.
- Both the `document_segments` and `document_processing_jobs` schemas are managed by reviewed Alembic revisions and must be applied before processing and indexing.
- **Acceptance:** tests cover supported and unsupported types, size limits, malformed/empty files, extraction, provenance, storage failure, cleanup, durable enqueue/retry/recovery, and completion only after successful indexing.

## 5. Audio/video transcription

- Evaluate a local Whisper-compatible runtime against hardware, quality, latency, licensing, and resource constraints.
- Add MP3/WAV transcription and MP4 audio extraction with FFmpeg, preserving timestamps where available.
- **Acceptance:** tests cover valid media, timestamps, invalid media, missing FFmpeg/model conditions, and explicit failures. Document measured results only.

## 6. Chunking, embeddings, and Pinecone integration

- **Implemented (Day 23):** provider-neutral embedding service contract, explicit optional provider/model/dimension settings, validation of inputs and returned finite vectors, consistent dimension checks, and batching at an adapter-declared maximum. No provider/model/dimension has been selected or integrated; no API credentials or external calls are needed.
- **Implemented (Day 24):** official `google-genai` SDK adapter for `gemini-embedding-001`, secure `GEMINI_API_KEY` configuration, configurable `output_dimensionality` (default 3072, documented model default), bounded 100-text requests, safe provider errors, and mocked SDK tests. No live API call has been made or verified.
- **Implemented (Day 25):** document embedding service retrieves persisted segments in ordinal order, calls the configured embedding service, and returns typed in-memory vectors associated with document ID, segment ID, and ordinal. Empty/malformed segment data and count/dimension mismatches fail explicitly. Vectors are not persisted.
- **Implemented (Day 26):** provider-neutral `VectorStore` protocol and official Pinecone SDK adapter for validated, deterministic-ID vector upserts with document/segment/ordinal metadata. The `DocumentVectorStorageService` orchestrates Day 25 embedding and upsert; it reports success only after storage succeeds. Index presence and dimension are verified when constructing the adapter, and upserts are split into configured batches. Tests use fakes and make no network calls; no live Pinecone index was verified.
- **Implemented (Day 27):** typed similarity search through the same vector-store protocol and Pinecone adapter. Queries validate dimensions and finite values, support positive configurable `top_k` and optional document UUID filtering, and return typed IDs, ordinal, score, and available metadata. `DocumentRetrievalService` embeds text queries through the existing Gemini-compatible abstraction and searches the injected store. Tests use fake embedding/store clients and make no network calls. No retrieval endpoint, LLM generation, deletion, or live index verification was added.
- **Implemented (Day 28):** `POST /api/v1/retrieval/search` exposes the retrieval service with a validated query, `top_k` (1–100, default 10), and optional document UUID filter. Typed responses return result identifiers, ordinal, score, and available metadata. Configuration failures return safe `503` responses; provider failures return safe `502` responses; invalid requests return `422`. API tests override the service with fakes. No live Gemini or Pinecone calls were made.
- **Implemented (Day 29):** `GroundedAnswerService` loads retrieved segment IDs from PostgreSQL, verifies document/ordinal consistency, uses only non-empty stored text as untrusted context, and asks an injectable LLM provider for an answer. Citations are built from verified database records, not model-generated claims. No search results or usable sources produce an explicit abstention without calling the LLM. `POST /api/v1/retrieval/answer` exposes the service. The official existing `google-genai` SDK is wrapped behind a grounded-answer provider, but no generation-model default is chosen: configure `LLM_PROVIDER=gemini`, an explicitly selected supported `LLM_MODEL`, and `GEMINI_API_KEY`. Offline tests use fakes; no live AI service calls were made.
- **Implemented (Day 32):** the upload flow queues document processing without holding the HTTP request open for extraction or provider calls. A separate worker connects local PDF/TXT extraction and PostgreSQL segment persistence to Gemini embeddings and Pinecone upsert. Upload-to-index processing is now wired, subject to applying migrations, running the worker, and providing valid provider/index configuration.
- **Next:** evaluate retrieval relevance, citation correctness, and abstention with a versioned dataset; select/validate an LLM model and verify provider account, privacy, and usage terms before a live smoke test.
- Decide Pinecone index region/namespace, re-embedding, deletion, and reconciliation behavior for partial failures before production use.
- **Acceptance:** tests cover provider contract/input-output validation and chunk/provenance mapping, embedding compatibility, upsert/query failures, and no-result behavior. No reranking in v1.

## 7. Retrieval and grounded answers with citations

- **Implemented (Day 29):** the internal grounded-answer service retrieves matching segment records from PostgreSQL, provides non-empty text as untrusted context to an injectable LLM provider, and creates citations from verified document/segment records. Missing sources and model abstention return a clear insufficient-evidence response. `POST /api/v1/retrieval/answer` exposes this flow.
- **Configuration pending:** no generation model is selected by default. Configure `LLM_PROVIDER=gemini`, select a supported `LLM_MODEL`, and provide `GEMINI_API_KEY`; live provider/model behavior remains unverified.
- **Acceptance:** offline tests cover citations, source mismatches/missing records, prompt injection text, no-result abstention, and safe provider failures. Retrieval relevance and answer support need evaluation before production; never claim hallucinations are eliminated.

## 8. Frontend dashboard and chat UI

- Establish the React/TypeScript/Vite/Tailwind toolchain and connect to `/api/v1`.
- Implement document upload/list/status and chat experience against the documented API.
- **Acceptance:** UI checks cover primary flows and failure states; no secrets are embedded in the client bundle.

## 9. Summaries, study notes, and transcript UI

- Add short and detailed summaries, topic-wise notes, key concepts, transcript retrieval/display, and source references.
- **Acceptance:** tests verify source scoping, citations/provenance where available, validation, and generation failure behavior.

## 10. Automated testing, evaluation, and deployment preparation

- Complete automated tests and an offline evaluation set for retrieval, citations, answer support, abstention, latency, and failure cases.
- Review privacy, security, operational limits, cost, and deployment requirements. Do not add Docker/Kubernetes until a separate decision is made.
- **Acceptance:** repeatable evaluation with documented dataset/model/config versions; no invented performance claims; deployment readiness review identifies access controls.

## 11. Future integration with LectureIQ

- Integrate only through LectureIQ's existing Express backend after contract and security review.
- Keep Cognify AI independently runnable; define service authentication, network boundaries, timeout/retry behavior, and error mapping before enabling calls.
- **Acceptance:** the integration contract is agreed, success/failure tests are in place, and the service remains independently testable and runnable.
