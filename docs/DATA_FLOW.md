# Data flow

> **Status:** Proposed flow. No ingestion or retrieval implementation exists yet.

## Ingestion

```mermaid
flowchart TD
    Upload[Upload request] --> Validate{Type and size allowed?}
    Validate -->|No| Reject[Return explicit validation error]
    Validate -->|Yes| Route{Input type}
    Route -->|PDF| PDF[Extract text and page references]
    Route -->|TXT| TXT[Decode and extract text]
    Route -->|Audio| Audio[Transcribe with timestamps where available]
    Route -->|Video| ExtractAudio[Extract audio using FFmpeg]
    ExtractAudio --> Audio
    PDF --> Normalize[Normalize text, source identity, and locations]
    TXT --> Normalize
    Audio --> Normalize
    Normalize --> Nonempty{Usable extracted content?}
    Nonempty -->|No| Fail[Return explicit extraction/transcription error]
    Nonempty -->|Yes| Chunk[Chunk content and retain provenance]
    Chunk --> Embed[Create embeddings]
    Embed --> Persist[Store vectors, text, and metadata]
    Persist --> Result[Return ingestion status and source identifier]
```

### Ingestion requirements

1. Validate supported media type and configured byte-size limit; do not trust a filename extension alone.
2. Extract PDF/TXT directly. For audio/video, transcribe speech; video audio is extracted with FFmpeg. Timestamp availability and quality depend on the selected extractor/model.
3. Normalize all extractor output to a shared representation while retaining source ID, content type, and location (page or time range where possible).
4. Apply an evaluated chunking strategy; create embeddings and persist the chunk text and metadata alongside vectors.
5. Make failure and partial-success semantics explicit. Do not silently index empty content or claim full ingestion when a required step failed.

Temporary-file lifecycle, deduplication, idempotency, and retry behavior must be defined during implementation.

## Retrieval and answer generation

```mermaid
flowchart TD
    Query[Question and optional filters] --> ValidateQuery[Validate query and options]
    ValidateQuery --> EmbedQuery[Embed query]
    EmbedQuery --> Search[Vector search with top-k and metadata filters]
    Search --> Threshold{Evidence meets configured threshold?}
    Threshold -->|No| Abstain[Return abstention with no unsupported answer]
    Threshold -->|Yes| Context[Build evidence context with source locations]
    Context --> Generate[Generate answer using replaceable LLM service]
    Generate --> Citations[Map supported claims to retrieved source citations]
    Citations --> Response[Return answer, citations, and evidence metadata]
```

### Retrieval requirements

- Apply configurable top-k, similarity threshold, and metadata filters; document score semantics for the selected vector backend.
- No reranking in v1.
- Treat retrieved text as untrusted content, not system instructions.
- Abstain when evidence is absent, below threshold, or does not support the requested answer. Threshold alone is not proof that a generated answer is correct.
- Preserve PDF page and audio/video timestamp references in results and citations whenever the source extraction provides them.
- Tests should include relevant and irrelevant queries, metadata filtering, threshold boundaries, citation mapping, unanswerable questions, and vector/embedding/LLM errors.
