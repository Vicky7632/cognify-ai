from collections.abc import Sequence
from uuid import UUID

import pytest

from app.services.document_retrieval_service import (
    DEFAULT_TOP_K,
    DocumentRetrievalService,
)
from app.services.embedding_service import (
    EmbeddingInputError,
    EmbeddingProviderError,
)
from app.vector_store.protocol import (
    VectorSearchResult,
    VectorStoreError,
    VectorStoreInputError,
)

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class FakeEmbeddingService:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[str] = []

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        if self.failure is not None:
            raise self.failure
        return (0.1, 0.2, 0.3)


class FakeVectorStore:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[
            tuple[tuple[float, ...], int, UUID | None]
        ] = []

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        document_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self.calls.append((tuple(query_vector), top_k, document_id))
        if self.failure is not None:
            raise self.failure
        return (
            VectorSearchResult(
                vector_id="stable-vector-id",
                document_id=DOCUMENT_ID,
                segment_id=SEGMENT_ID,
                ordinal=3,
                score=0.87,
                metadata={
                    "document_id": str(DOCUMENT_ID),
                    "segment_id": str(SEGMENT_ID),
                    "ordinal": 3,
                    "source": "lesson.pdf",
                },
            ),
        )


def make_service(
    *,
    embedding_failure: Exception | None = None,
    store_failure: Exception | None = None,
) -> tuple[DocumentRetrievalService, FakeEmbeddingService, FakeVectorStore]:
    embedder = FakeEmbeddingService(embedding_failure)
    vector_store = FakeVectorStore(store_failure)
    return DocumentRetrievalService(embedder, vector_store), embedder, vector_store


def test_retrieves_with_query_embedding_top_k_and_document_filter() -> None:
    service, embedder, vector_store = make_service()

    results = service.retrieve(
        "what is retrieval?",
        top_k=7,
        document_id=DOCUMENT_ID,
    )

    assert embedder.calls == ["what is retrieval?"]
    assert vector_store.calls == [((0.1, 0.2, 0.3), 7, DOCUMENT_ID)]
    assert len(results) == 1
    assert results[0].document_id == DOCUMENT_ID
    assert results[0].segment_id == SEGMENT_ID
    assert results[0].ordinal == 3
    assert results[0].score == 0.87
    assert results[0].metadata["source"] == "lesson.pdf"


def test_uses_default_top_k() -> None:
    service, _, vector_store = make_service()

    service.retrieve("query")

    assert vector_store.calls == [((0.1, 0.2, 0.3), DEFAULT_TOP_K, None)]


@pytest.mark.parametrize("query", ("", "   ", None, 42))
def test_empty_or_invalid_query_fails_before_embedding(query: object) -> None:
    service, embedder, vector_store = make_service()

    with pytest.raises(EmbeddingInputError):
        service.retrieve(query)

    assert embedder.calls == []
    assert vector_store.calls == []


@pytest.mark.parametrize("top_k", (0, -1, True, 1.5))
def test_invalid_top_k_fails_before_embedding(top_k: int) -> None:
    service, embedder, vector_store = make_service()

    with pytest.raises(VectorStoreInputError, match="top_k"):
        service.retrieve("query", top_k=top_k)

    assert embedder.calls == []
    assert vector_store.calls == []


def test_embedding_provider_failure_propagates_without_search() -> None:
    failure = EmbeddingProviderError("Safe embedding failure.")
    service, embedder, vector_store = make_service(embedding_failure=failure)

    with pytest.raises(EmbeddingProviderError, match="Safe embedding failure"):
        service.retrieve("query")

    assert embedder.calls == ["query"]
    assert vector_store.calls == []


def test_vector_store_failure_propagates_without_returning_success() -> None:
    failure = VectorStoreError("Safe vector store failure.")
    service, embedder, vector_store = make_service(store_failure=failure)

    with pytest.raises(VectorStoreError, match="Safe vector store failure"):
        service.retrieve("query")

    assert embedder.calls == ["query"]
    assert vector_store.calls == [((0.1, 0.2, 0.3), DEFAULT_TOP_K, None)]
