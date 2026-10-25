from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.services.embedding_service import (
    EmbeddingInputError,
    EmbeddingVector,
)
from app.vector_store.protocol import (
    VectorSearchResult,
    VectorStore,
    VectorStoreInputError,
)

DEFAULT_TOP_K = 10


class QueryEmbeddingService(Protocol):
    def embed_text(self, text: str) -> EmbeddingVector:
        """Return a finite vector for one non-empty query."""
        ...


class DocumentRetrievalService:
    """Embed a text query and retrieve matching stored document segments."""

    def __init__(
        self,
        embedding_service: QueryEmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        document_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Return Pinecone matches in descending similarity score order."""
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingInputError("Embedding texts must be non-empty strings.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise VectorStoreInputError("top_k must be a positive integer.")
        if document_id is not None and not isinstance(document_id, UUID):
            raise VectorStoreInputError("Document filter must be a UUID.")

        query_vector: Sequence[float] = self._embedding_service.embed_text(query)
        return self._vector_store.search(
            query_vector,
            top_k=top_k,
            document_id=document_id,
        )
