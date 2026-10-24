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
- **Implemented (Day 20 Step 3):** processing persists extracted segments with zero-based ordinal and source provenance, then commits document completion in the same transaction. Extraction runs outside database transactions; persistence failures roll back both rows and completion. The processor/orchestrator are not exposed through an API.
- **Implemented (Day 22):** `GET /api/v1/documents/{document_id}/segments` retrieves persisted segments in ordinal order with bounded offset/limit pagination and a total count. The endpoint has no authentication or ownership authorization.
- The `document_segments` schema is managed by the reviewed Alembic revision and must be applied to a target database before persistence is used.
- **Acceptance:** tests cover supported and unsupported types, size limits, malformed/empty files, extraction, provenance, storage failure, and cleanup.

## 5. Audio/video transcription

- Evaluate a local Whisper-compatible runtime against hardware, quality, latency, licensing, and resource constraints.
- Add MP3/WAV transcription and MP4 audio extraction with FFmpeg, preserving timestamps where available.
- **Acceptance:** tests cover valid media, timestamps, invalid media, missing FFmpeg/model conditions, and explicit failures. Document measured results only.

## 6. Chunking, embeddings, and Pinecone integration

- **Implemented (Day 23):** provider-neutral embedding service contract, explicit optional provider/model/dimension settings, validation of inputs and returned finite vectors, consistent dimension checks, and batching at an adapter-declared maximum. No provider/model/dimension has been selected or integrated; no API credentials or external calls are needed.
- **Implemented (Day 24):** official `google-genai` SDK adapter for `gemini-embedding-001`, secure `GEMINI_API_KEY` configuration, configurable `output_dimensionality` (default 3072, documented model default), bounded 100-text requests, safe provider errors, and mocked SDK tests. No live API call has been made or verified.
- **Next:** define chunking and shared source metadata; evaluate Gemini embedding task types and account/usage constraints, then integrate Pinecone for vectors and searchable metadata. Keep the model and dimension consistent for both document and query vectors.
- Define Pinecone index/version/namespace, re-embedding, deletion, and consistency behavior before integration.
- **Acceptance:** tests cover provider contract/input-output validation and chunk/provenance mapping, embedding compatibility, upsert/query failures, and no-result behavior. No reranking in v1.

## 7. Retrieval and grounded answers with citations

- Add configurable top-k, similarity thresholds, metadata filters, and a replaceable LLM boundary.
- Return evidence-grounded answers, page/timestamp citations, and explicit abstention when evidence is insufficient.
- **Acceptance:** tests cover retrieval relevance, citation locations, unanswerable queries, provider failures, and unsupported claims; never claim hallucinations are eliminated.

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
