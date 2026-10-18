# ADR 0001: Initial architecture direction

- **Status:** Accepted as initial project direction; selected details remain open where noted
- **Date:** 2026-10-18

## Context

Cognify AI is a standalone multimodal learning-material RAG application, later callable by LectureIQ through its existing Express backend. V1 inputs are PDF, TXT, MP3, WAV, and MP4. Planned outputs include grounded answers with citations, summaries, topic-wise notes, key concepts, and transcripts. This repository currently contains documentation and directory guides only.

## Decision

- **React + TypeScript + Vite + Tailwind CSS** for the planned browser interface. No frontend code/tooling is implemented.
- **Python + FastAPI + Pydantic settings** for the standalone service and typed runtime configuration. No routes/configuration code is implemented.
- **SQLAlchemy + PostgreSQL** for relational document/processing/chat data, optional ownership/course associations, and source metadata. Tables and migrations are deferred from the current phase.
- **Pinecone** for vector embeddings and searchable metadata, separate from relational records and original files. Index, region, cost, and operational setup remain open; no credentials or index are assumed.
- **Separate file-storage abstraction**, with local filesystem acceptable initially, to keep original files independent of relational and vector stores. Define safe paths, lifecycle, and retention before implementation.
- Use a shared ingestion representation and preserve PDF page and media timestamps when available.
- Keep embedding, transcription, file-storage, and LLM provider details behind replaceable boundaries. Evaluate local Whisper-compatible model/runtime and FFmpeg requirements before selecting exact components.
- Ground answers in retrieved evidence and abstain when insufficient; do not claim hallucinations can be eliminated.
- OCR, video-frame analysis, MVP authentication, and reranking are not initial requirements. Never expose the unauthenticated service publicly without appropriate access controls.
- Prefer free/low-cost development choices; paid cloud infrastructure is not mandatory. Use HTTPX where useful for outbound integrations.
- Do not implement application logic, routes, schema, or migrations during this documentation/structure phase.

## Consequences

- Cognify AI remains independently runnable; future LectureIQ integration is through its Express backend.
- PostgreSQL, Pinecone, and file storage have distinct responsibilities and require cross-store consistency/deletion decisions.
- Provider/model, dependencies, licensing, privacy, hardware, and operational choices require evaluation before selection.
- Embedding/index changes may require re-embedding and migration planning.
- Unauthenticated operation is restricted to trusted development unless suitable controls are added.
- Provenance must survive extraction, chunking, retrieval, and generation for citations.

## Open decisions

Explicit software license; Python/Node versions and package tooling; layouts and test/lint tools; PostgreSQL local setup and future migration conventions; embedding model/provider/dimensions; LLM provider/model; PDF parser and limits; transcription runtime/license/hardware; FFmpeg distribution; Pinecone index/cost and metadata; file retention; chunking/citations; sync/job lifecycle; API error envelope; deployment/privacy/access-control requirements.

## Alternatives considered

- **Use PostgreSQL for vectors and relational data together:** not selected; PostgreSQL handles relational state while Pinecone handles vector search metadata.
- **Store original binaries in PostgreSQL/Pinecone:** not selected; originals belong to the separate file-storage abstraction.
- **Provider-specific implementation in API handlers:** rejected because it makes model/service choices harder to evaluate and replace.
- **Reranking at launch:** deferred until a measured retrieval baseline demonstrates need.
