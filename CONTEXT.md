# Cognify AI: project context

## Scope

Cognify AI is a standalone multimodal retrieval-augmented generation (RAG) application for learners. It is planned to accept PDF, TXT, MP3, WAV, and MP4 learning materials and provide evidence-grounded Q&A, citations, summaries, topic-wise notes, key concepts, and transcripts. Future LectureIQ integration must be through its existing Express backend. LectureIQ is separate; do not access or modify its source.

The backend has a minimal FastAPI application with a health endpoint, document metadata and multipart upload APIs, lifecycle status management, a synchronous processor contract/orchestration foundation, and a local PDF/TXT extractor. Extracted segments are persisted in PostgreSQL by the internal processing service and can be retrieved through a paginated document-segment API. No frontend functionality, media transcription, or multimodal RAG integrations have been implemented.

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

**Current phase:** Day 28, versioned semantic retrieval API. The backend includes a `pypdf`-based local PDF/TXT processor implementing the typed synchronous contract. PDF text remains separated by 1-based source page; TXT uses strict UTF-8 decoding with optional BOM. Extracted segments persist in ordinal order in PostgreSQL, and completion is committed atomically with segment persistence. `GET /api/v1/documents/{document_id}/segments` returns ordered segments with bounded `limit`/`offset` pagination and total count. The embedding service uses a provider-neutral typed batch contract; its Gemini adapter uses the official `google-genai` SDK and `gemini-embedding-001`. The configured default output size is 3072 dimensions, based on Gemini's documented model default, and can be overridden through `EMBEDDING_DIMENSIONS`; actual vector values and configured dimensions are validated. Gemini requests are batched at no more than 100 texts, and provider failures are exposed only as safe errors. `DocumentEmbeddingService` loads all persisted segments in ascending ordinal order and returns generated vectors associated with document ID, segment ID, and ordinal. `DocumentVectorStorageService` connects those results to a provider-neutral vector-store protocol; the Pinecone adapter validates vectors and metadata, derives stable UUID5 IDs from document/segment UUIDs, verifies an existing index's dimension, and performs ordered batched upserts. Day 27 adds typed similarity search with query-vector dimension/finite-value validation, configurable positive `top_k`, optional document UUID filtering, and safe malformed-response/provider-error handling. `DocumentRetrievalService` embeds a text query then searches the injected vector store. Day 28 exposes it through `POST /api/v1/retrieval/search`, accepting a non-whitespace query, `top_k` from 1–100 (default 10), and optional `document_id`; the typed response includes vector/document/segment IDs, ordinal, score, and available metadata. Empty matches return HTTP 200 with an empty results list; validation failures return 422, configuration failures 503, and provider failures 502 with safe details. The endpoint is unauthenticated and not ownership-protected; restrict it to a trusted local environment. Configure `GEMINI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX`, and `PINECONE_UPSERT_BATCH_SIZE`; the index must be created separately with a dimension matching `EMBEDDING_DIMENSIONS`. Neither real Gemini calls nor the Pinecone index configuration have been verified. Pinecone deletion, synchronization/cleanup after partial remote failures, transcription, background workers, and queues are not implemented. The document-segment Alembic revision must be applied before segment persistence or embedding generation against a database. Uploaded documents start as `uploaded`; metadata-only records retain legacy `pending`.

**Next planned phase:** Evaluate retrieval quality and define grounded answer generation with citations and abstention. Keep dimensions and model consistent between document and query embeddings; verify provider account/usage constraints before production use. Decide on deletion and retry reconciliation for partial Pinecone writes. Make future schema changes through reviewed Alembic migrations; do not create tables directly or run schema DDL outside this workflow.

## Open decisions

- Explicit repository license selection; no `LICENSE` exists until terms are confirmed.
- Python/Node versions, package managers, project layouts, test/lint/type-check tools; PostgreSQL local setup.
- PDF parser details beyond the initial `pypdf` extractor, upload limits, file retention, and safe storage lifecycle.
- Pinecone index name, region/namespace strategy, deletion/re-embedding consistency, and account/cost approach; evaluate the selected Gemini model and task types before live use. The adapter expects the configured index to exist and match the configured embedding dimension.
- Whisper-compatible model/runtime/license/hardware and FFmpeg distribution.
- LLM provider/model, data privacy/retention terms, chunking/citation schema, API schemas, sync/job lifecycle.
- Deployment and access-control design before non-local exposure.

## Safe continuation instructions

Future agents should read this file and the relevant design documents before changing code, inspect the current repository and status first, preserve existing user changes, and implement only the active roadmap phase. Keep provider-specific logic behind replaceable interfaces where practical. Never add real secrets, modify LectureIQ, or claim a planned feature is implemented. Run relevant tests and report exact files changed and validation results. See [.github/copilot-instructions.md](.github/copilot-instructions.md).
