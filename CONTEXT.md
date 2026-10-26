# Cognify AI: project context

## Scope

Cognify AI is a standalone multimodal retrieval-augmented generation (RAG) application for learners. It is planned to accept PDF, TXT, MP3, WAV, and MP4 learning materials and provide evidence-grounded Q&A, citations, summaries, topic-wise notes, key concepts, and transcripts. Future LectureIQ integration must be through its existing Express backend. LectureIQ is separate; do not access or modify its source.

The backend has a minimal FastAPI application with a health endpoint, document metadata and multipart upload APIs, lifecycle status management, a durable PostgreSQL processing-job queue with a separate worker, and a local PDF/TXT extractor. Uploads enqueue processing atomically with document metadata. The worker persists extracted segments before embedding and Pinecone upsert; only the indexing orchestration marks documents complete. Segments can be retrieved through a paginated document-segment API. No frontend functionality or media transcription has been implemented.

## Architecture and stack direction

- **Frontend:** React, TypeScript, Vite, Tailwind CSS.
- **Backend:** Python, FastAPI, Pydantic settings, SQLAlchemy, PostgreSQL, and HTTPX where useful.
- **Vector search:** Pinecone for embeddings and searchable metadata, not relational application records.
- **Files:** separate storage abstraction; local filesystem is acceptable for initial development.
- **Processing/AI:** PDF/TXT extraction, FFmpeg video audio extraction, local Whisper-compatible transcription, an embedding service/provider contract, and future LLM abstractions.
- **Testing:** pytest is planned for backend tests; frontend test tooling remains open.
- Prefer free/low-cost tools. Paid services are not mandatory. Gemini is selected for the initial embedding adapter; credentials and applicable account/usage terms must be configured and verified separately.

PostgreSQL is intended for document metadata, processing status/errors, optional ownership/course associations, chat sessions/messages, and source/processing metadata. Use reviewed Alembic migrations for schema changes. See [the architecture decision](docs/decisions/0001-initial-architecture.md).

## Constraints and behavior

- The initial service is unauthenticated, for trusted local/development use only. It must not be exposed publicly without suitable access controls.
- Validate upload size, file type, and malformed input; surface failures explicitly.
- PDF/TXT use direct extraction. MP3/WAV are transcribed; MP4 audio is extracted with FFmpeg then transcribed. All extracted content should enter a shared ingestion pipeline.
- Preserve PDF page references and media timestamps as source metadata whenever extraction supports them.
- Use semantic chunking where appropriate; tune retrieval top-k, similarity threshold, and metadata filters through evaluation.
- Answers must be grounded in retrieved evidence and abstain when evidence is insufficient. Do not claim hallucinations can be completely eliminated.
- No reranking in v1. Reconsider only if evaluation shows a measurable need.
- OCR and video-frame analysis are future scope; reranking is not required in v1.
- Manage database schema changes through reviewed Alembic migrations; keep ORM model definitions separate from schema creation.
- Never commit secrets or expose API secrets in frontend code. No paid cloud infrastructure is mandatory for development.

## Directory conventions

`frontend/` contains setup/status guidance until frontend implementation begins. `backend/` contains the FastAPI service, SQLAlchemy models, Alembic migrations, and backend tests. `tests/` and `scripts/` describe intended roles; do not add placeholder modules. `docs/` contains design, proposed API, data flow, evaluation, and decisions; `.github/` contains agent guidance, issue templates, and PR template.

When application work begins, keep a clear separation between API/transport, ingestion and extraction, retrieval, provider adapters, and configuration. Add code only when a roadmap phase calls for it; do not pre-create speculative modules. Keep tests near the project’s chosen test layout and cover failure paths as well as success paths.

## Current phase and next phase

**Current phase:** Day 32, durable document processing and indexing orchestration. The backend includes a `pypdf`-based local PDF/TXT processor implementing the typed synchronous contract. PDF text remains separated by 1-based source page; TXT uses strict UTF-8 decoding with optional BOM. Extracted segments persist in ordinal order in PostgreSQL; only indexing orchestration can mark a document complete, after embeddings and vector upsert succeed. `GET /api/v1/documents/{document_id}/segments` returns ordered segments with bounded `limit`/`offset` pagination and total count. The embedding service uses a provider-neutral typed batch contract; its Gemini adapter uses the official `google-genai` SDK and `gemini-embedding-001`. The configured default output size is 3072 dimensions, based on Gemini's documented model default, and can be overridden through `EMBEDDING_DIMENSIONS`; actual vector values and configured dimensions are validated. Gemini requests are batched at no more than 100 texts, and provider failures are exposed only as safe errors. `DocumentEmbeddingService` loads all persisted segments in ascending ordinal order and returns generated vectors associated with document ID, segment ID, and ordinal. `DocumentVectorStorageService` connects those results to a provider-neutral vector-store protocol; the Pinecone adapter validates vectors and metadata, derives stable UUID5 IDs from document/segment UUIDs, verifies an existing index's dimension, and performs ordered batched upserts. Day 27 adds typed similarity search with query-vector validation, `top_k`, optional document UUID filtering, and safe provider errors. Day 28 exposes search through `POST /api/v1/retrieval/search`. Day 29 adds `GroundedAnswerService`: it retrieves source IDs, loads segment text and document metadata from PostgreSQL by segment ID, verifies document/ordinal consistency, passes only non-empty stored content as explicitly untrusted context, and assembles citations from actual database records. Missing or unusable source rows result in a clear insufficient-evidence response without an LLM call. `POST /api/v1/retrieval/answer` returns answer, citations (document ID, segment ID, ordinal, optional page number, filename, file type, and retrieval score), and an insufficient-evidence flag. The Gemini generation adapter uses the existing `google-genai` SDK but deliberately selects no model by default; configure `LLM_PROVIDER=gemini`, an explicitly chosen supported `LLM_MODEL`, and `GEMINI_API_KEY`. Invalid requests return 422, missing provider configuration 503, and provider failures 502 with generic safe details; PostgreSQL lookup failures return 500. The answer endpoint is unauthenticated and not ownership-protected; restrict it to trusted local use. Unit/API tests use fakes; no live Gemini or Pinecone calls or live index verification were performed. Day 32 adds a PostgreSQL-backed durable job table and separate singleton worker: uploads atomically enqueue jobs; PDF/TXT extraction and segment persistence run first, then the database session closes before embedding and Pinecone upsert; only complete success marks a document completed. Jobs retry up to three attempts, interrupted jobs are recovered, and pre-existing uploaded records without jobs are queued on worker startup. The worker and API must run as separate processes after the processing-job migration is applied. Audio/video transcription remains unimplemented. The status API cannot manually set processing or completed. Because vectors and PostgreSQL cannot be committed atomically and the adapter has no delete operation, partial upserts or shorter reprocessing can leave stale vectors; these may consume retrieval slots or cause abstention. Grounded answers reject vectors with missing or mismatched fingerprints, so stale vectors are not treated as evidence against current PostgreSQL text. The document-segment and document-processing-job Alembic revisions must be applied before pipeline operation against a database. Uploaded documents start as `uploaded`; metadata-only records retain legacy `pending`.

**Next planned phase:** Evaluate retrieval relevance, citation correctness, and abstention against a versioned offline dataset; select and validate an LLM generation model before any live smoke test. Keep dimensions and model consistent between document and query embeddings; verify provider account/usage/privacy constraints before production use. Decide on deletion and retry reconciliation for partial Pinecone writes. Make future schema changes through reviewed Alembic migrations; do not create tables directly or run schema DDL outside this workflow.

**Day 32 citation-integrity update:** Vectors now include a SHA-256 fingerprint of the exact embedded segment text. Grounded answers compare it to the current PostgreSQL segment and exclude missing or mismatched fingerprints from answer context and citations; pre-existing vectors need reindexing before they can support citations. Stale vectors may still consume retrieval slots and lead to safe abstention. The worker monitors advisory-lock ownership and checks it before job/status writes and processing stages. These checks narrow but do not eliminate the race between verification and a database write or external request; loss detected during an operation prevents later stages and old-worker failure/retry writes, but an in-flight external request cannot be cancelled.

## Open decisions

- Explicit repository license selection; no `LICENSE` exists until terms are confirmed.
- Python/Node versions, package managers, project layouts, test/lint/type-check tools; PostgreSQL local setup.
- PDF parser details beyond the initial `pypdf` extractor, upload limits, file retention, and safe storage lifecycle.
- Pinecone index name, region/namespace strategy, deletion/re-embedding consistency, and account/cost approach; evaluate the selected Gemini model and task types before live use. The adapter expects the configured index to exist and match the configured embedding dimension.
- Whisper-compatible model/runtime/license/hardware and FFmpeg distribution.
- LLM provider/model, data privacy/retention terms, chunking/citation schemas, Pinecone cleanup/reconciliation, and worker deployment lifecycle.
- Deployment and access-control design before non-local exposure.

## Safe continuation instructions

Future agents should read this file and the relevant design documents before changing code, inspect the current repository and status first, preserve existing user changes, and implement only the active roadmap phase. Keep provider-specific logic behind replaceable interfaces where practical. Never add real secrets, modify LectureIQ, or claim a planned feature is implemented. Run relevant tests and report exact files changed and validation results. See [.github/copilot-instructions.md](.github/copilot-instructions.md).
