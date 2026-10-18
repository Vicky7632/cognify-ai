# Backend application layout

This package layout is reserved for the planned Python/FastAPI service. No Python modules, routes, models, or business logic are implemented. Add `main.py` as the application entry point only when the backend skeleton phase begins.

| Module | Intended responsibility |
|---|---|
| `api/v1/` | Versioned HTTP endpoints and API dependency wiring. |
| `core/` | Pydantic Settings configuration and shared application concerns. |
| `db/` | PostgreSQL engine, connection, and session management. |
| `models/` | SQLAlchemy relational models only; no models in this structure phase. |
| `schemas/` | Request/response validation schemas. |
| `services/` | Application-level orchestration across boundaries. |
| `repositories/` | Relational PostgreSQL data access. |
| `ingestion/` | Shared ingestion pipeline coordinating stages. |
| `extraction/` | PDF/TXT text extraction and source location capture. |
| `chunking/` | Semantic or structure-aware chunking with provenance. |
| `embeddings/` | Embedding generation behind a replaceable interface. |
| `vector_store/` | Pinecone integration for vectors and searchable metadata. |
| `retrieval/` | Top-K search, similarity thresholds, and metadata filters. |
| `generation/` | Evidence-grounded LLM responses and source citations. |
| `transcription/` | Local Whisper-compatible transcription and timestamp metadata. |

PostgreSQL, Pinecone, and local filesystem storage remain separate responsibilities. These README files preserve the intended directories without adding empty code modules.
