# Vector storage

`protocol.py` defines the provider-neutral `VectorStore` contract and typed
records. The `PineconeVectorStore` adapter supports validated, batched upserts
and similarity search. Deletion is not implemented.

`DocumentVectorStorageService` combines the Day 25 document embedding pipeline
with an injected vector store. It returns generated results only after all
upserts succeed. The Pinecone adapter uses stable UUID5 IDs derived from the
document and segment UUIDs, stores those UUIDs plus the ordinal as metadata,
validates vectors against the configured embedding dimension, and batches
upserts. It verifies that the configured index exists and reports the expected
dimension; it never creates an index.

The Pinecone service does not provide a transaction across adapter batches. If
a later batch fails, earlier batches may already have been accepted; the
service raises rather than reporting success. Retry/deletion reconciliation is
not implemented.

Similarity search accepts an embedding vector, a positive `top_k`, and an
optional document UUID filter. The adapter validates finite values and
configured dimensions, requests metadata, and converts ordered matches into
typed results with the stable vector ID, document/segment UUIDs, ordinal,
score, and available metadata. Malformed responses and SDK failures become
safe application errors. `DocumentRetrievalService` embeds a non-empty text
query with the injected embedding service before searching. Its default
`top_k` is 10 and callers can provide another positive integer. Retrieval is
internal; no HTTP endpoint or answer generation is provided.

Configure `PINECONE_API_KEY`, `PINECONE_INDEX`, and
`PINECONE_UPSERT_BATCH_SIZE` in the backend environment. The index must be
created separately with a dimension matching `EMBEDDING_DIMENSIONS`. Pinecone
configuration is optional at application startup, but required when creating
the adapter. Tests use fake clients and make no network calls.
