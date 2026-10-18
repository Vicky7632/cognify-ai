# Data flow

> **Status:** Proposed pipelines only. No endpoints, processing jobs, or persistence are implemented.

## Ingestion

```mermaid
flowchart TD
    Upload[Frontend multipart upload] --> API[FastAPI /api/v1/documents]
    API --> Validate{Allowed type, size, and content?}
    Validate -->|No| Reject[Return explicit validation error]
    Validate -->|Yes| Create[Create document and processing state in PostgreSQL]
    Create --> File[Store original via file storage abstraction]
    File --> Route{PDF, TXT, MP3, WAV, MP4}
    Route -->|PDF| PDF[Extract text with page references]
    Route -->|TXT| TXT[Decode and extract text]
    Route -->|MP3 / WAV| Audio[Local Whisper-compatible transcription]
    Route -->|MP4| ExtractAudio[FFmpeg extracts audio]
    ExtractAudio --> Audio
    PDF --> Normalize[Normalize text, source identity, and locations]
    TXT --> Normalize
    Audio --> Normalize
    Normalize --> Nonempty{Usable extracted content?}
    Nonempty -->|No| Fail[Return explicit extraction/transcription error]
    Nonempty -->|Yes| Chunk[Chunk content and retain provenance]
    Chunk --> Embed[Create embeddings]
    Embed --> Persist[Pinecone vector upsert]
    Persist --> Complete[Record completion and metadata in PostgreSQL]
    Complete --> Result[Return ingestion status and source identifier]
    Normalize -->|Extraction/transcription failure| Failed[Record processing failure in PostgreSQL]
    Failed --> Result
```

### Ingestion requirements

1. Validate supported formats (PDF, TXT, MP3, WAV, MP4), actual content where practical, and configured size limits; do not trust only filename or client MIME type.
2. Create document/processing metadata in PostgreSQL and persist the original through the separate file storage abstraction.
3. Extract PDF/TXT directly. Transcribe MP3/WAV with a selected local model; use FFmpeg to extract MP4 audio first. Timestamp availability depends on the implementation.
4. Normalize output into a shared representation, retain source ID/type/location, and chunk while preserving page/time ranges where available.
5. Generate embeddings through an abstraction and upsert vectors plus minimal searchable metadata into Pinecone. Keep relational state in PostgreSQL and originals in file storage.
6. Make retry, partial failure, cleanup, idempotency, and cross-store consistency explicit. Do not silently index empty content or claim full success when a required step failed.

Temporary-file lifecycle, deduplication, idempotency, and retry behavior must be defined during implementation.

## Retrieval and answer generation

```mermaid
flowchart TD
    Query[Frontend question and filters] --> ValidateQuery[Validate request]
    ValidateQuery --> Metadata[Load document and chat metadata from PostgreSQL]
    ValidateQuery --> EmbedQuery[Embed query]
    EmbedQuery --> Search[Vector search with top-k and metadata filters]
    Search --> Threshold{Evidence supports an answer?}
    Threshold -->|No| Abstain[Return abstention with no unsupported answer]
    Threshold -->|Yes| Context[Build evidence context with source locations]
    Context --> Generate[Generate answer using replaceable LLM service]
    Generate --> Citations[Map supported claims to retrieved source citations]
    Citations --> Persist[Persist chat and source references in PostgreSQL]
    Persist --> Response[Return answer, citations, and evidence metadata]
    Metadata --> EmbedQuery
```

### Retrieval requirements

- Apply configurable top-k, similarity threshold, and metadata filters; document score semantics for the selected vector backend.
- No reranking in v1.
- Treat retrieved text as untrusted content, not system instructions.
- Abstain when evidence is absent, below threshold, or does not support the requested answer. Threshold alone is not proof that a generated answer is correct.
- Preserve PDF page and audio/video timestamp references in results and citations whenever the source extraction provides them.
- Store chat/session and relational source records in PostgreSQL; keep vectors/searchable metadata in Pinecone and original bytes in file storage.
- Tests should include relevant and irrelevant queries, metadata filtering, threshold boundaries, citation mapping, unanswerable questions, and vector/embedding/LLM errors.

Short/detailed summaries, topic-wise notes, key concepts, and transcripts are planned outputs scoped to source material. Preserve source references where practical and do not treat generated text as authoritative source evidence by default.
