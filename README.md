# Cognify AI

Cognify AI is a planned, standalone multimodal RAG service for turning documents, recordings, and videos into searchable, cited study material. It is designed to run independently and later integrate with the LectureIQ Express backend.

> **Status:** Documentation baseline only. No API, ingestion pipeline, vector database, transcription, or LLM integration is implemented yet.

## Planned capabilities

- Extract text from PDF and TXT files; transcribe audio and video with source timing where available.
- Index content for semantic retrieval and answer questions using retrieved evidence.
- Provide summaries and study notes, with citations to document pages or media timestamps where available.
- Abstain when retrieved evidence is insufficient. Retrieval and citations can reduce unsupported answers, but cannot eliminate hallucinations.

## Planned stack

Python, FastAPI, Pydantic, pytest, Pinecone, an embedding model selected through evaluation, FFmpeg for video audio extraction, and a local Whisper-compatible transcription option subject to hardware and model evaluation. LLM provider logic will be replaceable. These components are proposed, not installed or configured.

## Start here

- [Architecture](docs/ARCHITECTURE.md) and [data flow](docs/DATA_FLOW.md)
- [Proposed API](docs/API.md) (endpoints are not implemented)
- [Roadmap](ROADMAP.md) and [evaluation plan](docs/EVALUATION.md)
- [Project context for contributors and agents](CONTEXT.md)

Before implementation, review the open decisions in [the initial architecture decision](docs/decisions/0001-initial-architecture.md). Copy `.env.example` to `.env` only when local configuration is needed; never commit secrets.

## Security note

The v1 service is planned without authentication. Do not expose it publicly without suitable access controls, transport security, upload limits, and operational safeguards. See [architecture and security considerations](docs/ARCHITECTURE.md#security-and-privacy).
