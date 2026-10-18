# Roadmap

Phases are ordered by dependency. Each phase has an observable acceptance gate; defer later capabilities until the preceding gate is met. Nothing below is implemented by this documentation baseline.

## 0. Repository setup

- Establish the Python project structure, supported Python version, dependency and development workflow, configuration validation, logging, and pytest setup.
- Add `.env` loading with safe defaults and fail-fast validation for required settings; keep placeholders only in `.env.example`.
- Add health/readiness behavior only after its dependencies and semantics are defined.
- **Acceptance:** clean setup instructions work in a fresh development environment; configuration errors are actionable; baseline tests and lint/type checks run; no secrets are tracked.

## 1. PDF/TXT ingestion

- Validate file type, size, and malformed uploads.
- Extract PDF text with page references and TXT text; normalize extraction output to a common internal representation.
- Implement an ingestion path that reports useful errors and does not silently accept empty/failed extraction.
- **Acceptance:** tests cover valid PDF/TXT, page/source metadata, empty or malformed files, unsupported types, and size limits.

## 2. Audio/video transcription

- Evaluate local Whisper-compatible options against target hardware, quality, latency, licensing, and resource requirements; document selected model and operational constraints.
- Evaluate FFmpeg installation/runtime behavior and extract video audio; transcribe audio and extracted tracks with timestamps where available.
- **Acceptance:** tests cover supported audio/video inputs, timestamps, missing/invalid media, FFmpeg/model failures, and resource/configuration errors. No hardware or quality claims without measurements.

## 3. Shared retrieval

- Establish the shared ingestion representation and chunking policy for extracted text, retaining document page and media time metadata.
- Evaluate and select an embedding model; define dimension/version compatibility and Pinecone index configuration.
- Implement vector storage and retrieval with configurable top-k, similarity threshold, and metadata filters. Do not add reranking in v1.
- **Acceptance:** tests verify stored text/metadata, source filters, threshold/top-k behavior, embedding/index errors, and retrieval failure handling.

## 4. Grounded Q&A, summaries, and study notes

- Define a replaceable LLM service interface and choose an initial provider after cost, quality, privacy, and operational review.
- Generate answers only from retrieved context, return citations, and abstain when evidence is insufficient.
- Add summary and study-note generation over selected source material, with provenance retained where feasible.
- **Acceptance:** tests cover grounded answers, abstention, citation mapping, provider failures, and output validation. Prompts and responses must not imply certainty beyond evidence.

## 5. Evaluation and hardening

- Build a versioned, representative evaluation set and measurable retrieval, citation, answer-grounding, abstention, and failure metrics.
- Measure actual outcomes before tuning chunking, thresholds, prompts, or model choices; keep results reproducible and label dataset/model versions.
- Add operational limits and privacy/security review appropriate to the intended deployment.
- **Acceptance:** evaluation procedure and results are repeatable; regressions can be detected against a documented baseline. See [EVALUATION.md](docs/EVALUATION.md).

## 6. LectureIQ integration

- Only after the service contract and access-control design are accepted, integrate through LectureIQ's Express backend.
- Keep Cognify AI independently runnable; define service-to-service authentication, network boundary, timeout/retry policy, and error mapping before enabling calls.
- Do not change LectureIQ during this standalone service's documentation phase.
- **Acceptance:** integration contract and security review are approved, integration tests exercise success and failure paths, and Cognify AI remains independently testable and runnable.
