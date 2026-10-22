# Cognify AI: project context

## Scope

Cognify AI is a standalone multimodal retrieval-augmented generation (RAG) application for learners. It is planned to accept PDF, TXT, MP3, WAV, and MP4 learning materials and provide evidence-grounded Q&A, citations, summaries, topic-wise notes, key concepts, and transcripts. Future LectureIQ integration must be through its existing Express backend. LectureIQ is separate; do not access or modify its source.

The backend has a minimal FastAPI application with a health endpoint, Pydantic settings, lazy SQLAlchemy/PostgreSQL session management, an initial `Document` ORM model, and Alembic migration configuration. No frontend functionality, document API, or multimodal RAG integrations have been implemented.

## Architecture and stack direction

- **Frontend:** React, TypeScript, Vite, Tailwind CSS.
- **Backend:** Python, FastAPI, Pydantic settings, SQLAlchemy, PostgreSQL, and HTTPX where useful.
- **Vector search:** Pinecone for embeddings and searchable metadata, not relational application records.
- **Files:** separate storage abstraction; local filesystem is acceptable for initial development.
- **Processing/AI:** PDF/TXT extraction, FFmpeg video audio extraction, local Whisper-compatible transcription, embedding and LLM abstractions.
- **Testing:** pytest is planned for backend tests; frontend test tooling remains open.
- Prefer free/low-cost tools. Paid services are not mandatory. Provider/model/parser details remain open; do not assume services are configured.

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

**Current phase:** Week 3, Day 15. The backend includes the health endpoint, document metadata and multipart upload APIs, PostgreSQL session foundation, `Document` model, Alembic migration, document repository/service layers, and local filesystem upload storage. Document API access is unauthenticated and must remain restricted to a trusted local environment.

**Next planned phase:** Prepare uploaded files for document processing only in its designated roadmap phase, with suitable access controls before public exposure. Make future schema changes through reviewed Alembic migrations; do not create tables directly or run schema DDL outside this workflow.

## Open decisions

- Explicit repository license selection; no `LICENSE` exists until terms are confirmed.
- Python/Node versions, package managers, project layouts, test/lint/type-check tools; PostgreSQL local setup.
- PDF parser, upload limits, file retention, and safe storage lifecycle.
- Embedding model/provider/dimensions; Pinecone index, namespaces, metadata, and cost approach.
- Whisper-compatible model/runtime/license/hardware and FFmpeg distribution.
- LLM provider/model, data privacy/retention terms, chunking/citation schema, API schemas, sync/job lifecycle.
- Deployment and access-control design before non-local exposure.

## Safe continuation instructions

Future agents should read this file and the relevant design documents before changing code, inspect the current repository and status first, preserve existing user changes, and implement only the active roadmap phase. Keep provider-specific logic behind replaceable interfaces where practical. Never add real secrets, modify LectureIQ, or claim a planned feature is implemented. Run relevant tests and report exact files changed and validation results. See [.github/copilot-instructions.md](.github/copilot-instructions.md).
