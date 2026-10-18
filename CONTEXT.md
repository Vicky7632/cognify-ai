# Cognify AI: project context

## Scope

Cognify AI is a standalone, Python-based multimodal retrieval-augmented generation (RAG) microservice. It is intended to ingest PDF and TXT documents plus audio and video, retrieve relevant evidence, and produce grounded answers, summaries, and study notes with source citations where available. It may later be called by LectureIQ's Express backend. Do not modify LectureIQ as part of this project.

This repository currently contains a documentation baseline only. No application code, API endpoints, integrations, tests, or infrastructure have been implemented.

## Stack direction

- Python, FastAPI, Pydantic, and pytest are planned.
- Pinecone is the planned vector database; credential, index, region, and cost setup remain to be decided.
- Embedding model/provider is intentionally undecided pending quality, cost, latency, licensing, and deployment evaluation.
- Local Whisper-compatible transcription is a candidate for audio/video, subject to hardware, model, and licensing evaluation. FFmpeg is planned for extracting audio from video.
- LLM access must sit behind a replaceable service boundary. Provider and model are undecided.
- Prefer free or low-cost development paths where practical. Do not imply an external account, model, or tool is configured until verified.

See [the architecture decision](docs/decisions/0001-initial-architecture.md) for assumptions and decisions still open.

## Constraints and behavior

- The initial service is unauthenticated, for trusted local/development use only. It must not be exposed publicly without suitable access controls.
- Validate upload size, file type, and malformed input; surface failures explicitly.
- PDFs and TXT files use direct extraction. Audio/video are transcribed with timestamps where available. All extracted content should enter a shared ingestion pipeline.
- Preserve PDF page references and media timestamps as source metadata whenever extraction supports them.
- Use semantic chunking where appropriate; tune retrieval top-k, similarity threshold, and metadata filters through evaluation.
- Answers must be grounded in retrieved evidence and abstain when evidence is insufficient. Do not claim hallucinations can be completely eliminated.
- No reranking in v1. Reconsider only if evaluation shows a measurable need.
- No endpoints or business logic are to be implemented during the documentation-first phase.

## Directory conventions

The current repository establishes `docs/` for design, API, data flow, and evaluation documents; `docs/decisions/` for architecture decision records; and `.github/` for Copilot guidance and contribution workflow templates.

When application work begins, keep a clear separation between API/transport, ingestion and extraction, retrieval, provider adapters, and configuration. Add code only when a roadmap phase calls for it; do not pre-create speculative modules. Keep tests near the project’s chosen test layout and cover failure paths as well as success paths.

## Safe continuation instructions

Future agents should read this file and the relevant design documents before changing code, inspect the current repository and status first, preserve existing user changes, and implement only the active roadmap phase. Keep provider-specific logic behind replaceable interfaces where practical. Never add real secrets, modify LectureIQ, or claim a planned feature is implemented. Run relevant tests and report exact files changed and validation results. See [.github/copilot-instructions.md](.github/copilot-instructions.md).
