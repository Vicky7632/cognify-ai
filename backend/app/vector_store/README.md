# Vector storage

`protocol.py` defines the provider-neutral `VectorStore` contract and typed
records. The initial `PineconeVectorStore` adapter only upserts vectors and
metadata; query and deletion operations are not implemented.

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

Configure `PINECONE_API_KEY`, `PINECONE_INDEX`, and
`PINECONE_UPSERT_BATCH_SIZE` in the backend environment. The index must be
created separately with a dimension matching `EMBEDDING_DIMENSIONS`. Pinecone
configuration is optional at application startup, but required when creating
the adapter. Tests use fake clients and make no network calls.
