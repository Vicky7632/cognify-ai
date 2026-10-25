from collections.abc import Generator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.v1.retrieval import get_document_retrieval_service
from app.application import create_app
from app.core.config import Settings
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingInputError,
    EmbeddingOutputError,
    EmbeddingProviderError,
)
from app.vector_store.protocol import (
    VectorSearchResult,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreInputError,
)

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class FakeRetrievalService:
    def __init__(
        self,
        *,
        results: tuple[VectorSearchResult, ...] = (),
        failure: Exception | None = None,
    ) -> None:
        self.results = results
        self.failure = failure
        self.calls: list[tuple[str, int, UUID | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self.calls.append((query, top_k, document_id))
        if self.failure is not None:
            raise self.failure
        return self.results


def make_result() -> VectorSearchResult:
    return VectorSearchResult(
        vector_id="stable-vector-id",
        document_id=DOCUMENT_ID,
        segment_id=SEGMENT_ID,
        ordinal=4,
        score=0.875,
        metadata={
            "document_id": str(DOCUMENT_ID),
            "segment_id": str(SEGMENT_ID),
            "ordinal": 4,
            "source": "lesson.pdf",
            "tags": ("lecture", "pdf"),
        },
    )


@pytest.fixture
def api_client() -> Generator[tuple[TestClient, FakeRetrievalService], None, None]:
    service = FakeRetrievalService(results=(make_result(),))
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_document_retrieval_service] = lambda: service
    with TestClient(app) as client:
        yield client, service
    app.dependency_overrides.clear()


def test_retrieval_endpoint_returns_typed_match_and_forwards_filter(
    api_client: tuple[TestClient, FakeRetrievalService],
) -> None:
    client, service = api_client

    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "Explain this lesson",
            "top_k": 8,
            "document_id": str(DOCUMENT_ID),
        },
    )

    assert response.status_code == 200
    assert service.calls == [("Explain this lesson", 8, DOCUMENT_ID)]
    assert response.json() == {
        "query": "Explain this lesson",
        "top_k": 8,
        "results": [
            {
                "vector_id": "stable-vector-id",
                "document_id": str(DOCUMENT_ID),
                "segment_id": str(SEGMENT_ID),
                "ordinal": 4,
                "score": 0.875,
                "metadata": {
                    "document_id": str(DOCUMENT_ID),
                    "segment_id": str(SEGMENT_ID),
                    "ordinal": 4,
                    "source": "lesson.pdf",
                    "tags": ["lecture", "pdf"],
                },
            }
        ],
    }


def test_retrieval_endpoint_returns_empty_results(
    api_client: tuple[TestClient, FakeRetrievalService],
) -> None:
    client, service = api_client
    service.results = ()

    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "query"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "query",
        "top_k": 10,
        "results": [],
    }


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": " \t\n "},
        {"query": "query", "top_k": 0},
        {"query": "query", "top_k": 101},
        {"query": "query", "top_k": True},
        {"query": "query", "document_id": "not-a-uuid"},
        {"query": "query", "unexpected": "field"},
    ],
)
def test_invalid_retrieval_request_returns_422_without_service_call(
    api_client: tuple[TestClient, FakeRetrievalService],
    body: dict[str, object],
) -> None:
    client, service = api_client

    response = client.post("/api/v1/retrieval/search", json=body)

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    ("failure", "status_code", "detail"),
    [
        (EmbeddingInputError("Query must not be empty."), 422, "Query must not be empty."),
        (VectorStoreInputError("top_k must be positive."), 422, "top_k must be positive."),
        (EmbeddingConfigurationError("secret config detail"), 503, "Retrieval service is not configured."),
        (VectorStoreConfigurationError("secret index detail"), 503, "Retrieval service is not configured."),
        (EmbeddingProviderError("secret API key detail"), 502, "Document retrieval failed."),
        (EmbeddingOutputError("secret provider detail"), 502, "Document retrieval failed."),
        (VectorStoreError("secret Pinecone detail"), 502, "Document retrieval failed."),
    ],
)
def test_retrieval_service_failures_map_to_safe_http_responses(
    failure: Exception,
    status_code: int,
    detail: str,
) -> None:
    service = FakeRetrievalService(failure=failure)
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_document_retrieval_service] = lambda: service

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "query"},
        )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "secret" not in response.text
    app.dependency_overrides.clear()


def test_missing_provider_configuration_returns_safe_503_without_network() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "query"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Retrieval service is not configured."
    }
