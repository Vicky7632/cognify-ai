# ADR 0001: Initial architecture direction

- **Status:** Accepted as a project direction; component selections remain open where noted
- **Date:** 2026-10-18

## Context

Cognify AI is intended to be a standalone multimodal RAG microservice that can later integrate with LectureIQ through its Express backend. The initial scope includes PDF/TXT extraction, audio/video transcription, semantic retrieval, grounded answers, summaries, study notes, and source citations. The repository begins as documentation only; no provider or external service is assumed configured.

## Decision

- Use Python, FastAPI, Pydantic, and pytest as the planned service foundation.
- Plan Pinecone as the vector database, pending credential, index, region, cost, and operational setup.
- Use a shared ingestion representation for extracted/transcribed text, with source metadata for PDF pages and media timestamps when available.
- Keep embedding, transcription, vector database, and LLM provider-specific behavior behind replaceable boundaries where practical.
- Evaluate a Whisper-compatible local transcription option and FFmpeg for video audio extraction before committing to a model/runtime.
- Ground generation in retrieved evidence and abstain when evidence is insufficient; do not claim hallucinations can be eliminated.
- Exclude reranking from v1; revisit only when evaluation data demonstrates a need.
- Keep the v1 service unauthenticated, with explicit documentation that it must not be publicly exposed without suitable access controls.
- Do not implement endpoints or business logic in this documentation-first phase.

## Consequences

- The service can be developed and run independently of LectureIQ, with integration deferred until the service contract and security approach are ready.
- Multiple provider and infrastructure decisions must be validated before selecting dependencies or promising operational characteristics.
- Embedding/index changes may require re-embedding and migration planning.
- Local transcription may impose hardware and model-distribution constraints; measured evaluation is required.
- An unauthenticated API is restricted to trusted local/development use unless access controls are subsequently designed and implemented.
- Provenance must flow through extraction, chunking, retrieval, and generation to support useful citations.

## Open decisions

Embedding model/provider and dimensions; LLM provider/model; PDF parser; transcription model/runtime and license; FFmpeg distribution; Pinecone index settings and metadata schema; file size/type limits; chunking policy; synchronous versus asynchronous ingestion; API error envelope; supported Python version and package workflow; deployment, retention, authentication, and privacy requirements.

## Alternatives considered

- **Self-hosted vector database:** not selected for the initial direction because Pinecone is explicitly in the planned technology direction; a local alternative may be considered for low-cost development if it does not obscure production behavior.
- **Provider-specific implementation in API handlers:** rejected because it would make model and service choices harder to evaluate and replace.
- **Reranking at launch:** deferred to avoid complexity before a measured retrieval baseline exists.
