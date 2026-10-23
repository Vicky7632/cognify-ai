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
- Extracted segments are currently returned in memory, not persisted. The processor/orchestrator are not exposed through an API.
- **Next (Day 19):** decide whether extracted segments need relational persistence now or should remain in memory until a later ingestion/indexing design.
- **Acceptance:** tests cover supported and unsupported types, size limits, malformed/empty files, extraction, provenance, storage failure, and cleanup.

## 5. Audio/video transcription

- Evaluate a local Whisper-compatible runtime against hardware, quality, latency, licensing, and resource constraints.
- Add MP3/WAV transcription and MP4 audio extraction with FFmpeg, preserving timestamps where available.
- **Acceptance:** tests cover valid media, timestamps, invalid media, missing FFmpeg/model conditions, and explicit failures. Document measured results only.

## 6. Chunking, embeddings, and Pinecone integration

- Define chunking and shared source metadata; select an embedding approach after evaluation.
- Integrate Pinecone for vectors and searchable metadata; define index/version/namespace, re-embedding, deletion, and consistency behavior.
- **Acceptance:** tests cover chunk/provenance mapping, embedding compatibility, upsert/query failures, and no-result behavior. No reranking in v1.

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
