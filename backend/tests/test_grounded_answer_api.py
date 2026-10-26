from collections.abc import Generator
from hashlib import sha256
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.answers import get_grounded_answer_provider
from app.api.v1.retrieval import get_document_retrieval_service
from app.application import create_app
from app.core.config import Settings
from app.db.base import Base
from app.db.session import get_db
from app.models.document_segment import DocumentSegment
from app.repositories.document_repository import create_document
from app.services.grounded_answer_service import GroundedAnswerPrompt
from app.vector_store.protocol import VectorSearchResult, VectorStoreError

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class FakeRetrievalService:
    def __init__(self) -> None:
        self.matches: tuple[VectorSearchResult, ...] = ()
        self.failure: Exception | None = None
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
        return self.matches


class FakeAnswerProvider:
    def __init__(self) -> None:
        self.answer = "The stored source describes retrieval."
        self.failure: Exception | None = None
        self.prompts: list[GroundedAnswerPrompt] = []

    def generate_answer(self, prompt: GroundedAnswerPrompt) -> str:
        self.prompts.append(prompt)
        if self.failure is not None:
            raise self.failure
        return self.answer


@pytest.fixture
def answer_client() -> Generator[
    tuple[TestClient, sessionmaker[Session], FakeRetrievalService, FakeAnswerProvider],
    None,
    None,
]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    retrieval = FakeRetrievalService()
    provider = FakeAnswerProvider()

    def override_get_db() -> Generator[Session, None, None]:
        with session_factory() as session:
            yield session

    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_document_retrieval_service] = lambda: retrieval
    app.dependency_overrides[get_grounded_answer_provider] = lambda: provider
    with TestClient(app) as client:
        yield client, session_factory, retrieval, provider
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


def add_source(
    session_factory: sessionmaker[Session],
    *,
    text_content: str = "The stored source describes retrieval.",
) -> tuple[UUID, UUID]:
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename="lesson.txt",
                file_type="txt",
                storage_path="random-storage-key.txt",
            )
            segment = DocumentSegment(
                id=SEGMENT_ID,
                document_id=document.id,
                ordinal=0,
                text_content=text_content,
            )
            session.add(segment)
            session.flush()
            return document.id, segment.id


def make_match(
    document_id: UUID,
    segment_id: UUID,
    *,
    text_content: str = "The stored source describes retrieval.",
) -> VectorSearchResult:
    return VectorSearchResult(
        vector_id="stable-vector-id",
        document_id=document_id,
        segment_id=segment_id,
        ordinal=0,
        score=0.92,
        metadata={
            "document_id": str(document_id),
            "ordinal": 0,
            "content_hash": sha256(text_content.encode("utf-8")).hexdigest(),
        },
    )


def test_answer_endpoint_returns_verified_citations_and_filter(
    answer_client: tuple[
        TestClient, sessionmaker[Session], FakeRetrievalService, FakeAnswerProvider
    ],
) -> None:
    client, session_factory, retrieval, provider = answer_client
    document_id, segment_id = add_source(session_factory)
    retrieval.matches = (make_match(document_id, segment_id),)

    response = client.post(
        "/api/v1/retrieval/answer",
        json={
            "query": "What does the source say?",
            "top_k": 6,
            "document_id": str(document_id),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == provider.answer
    assert body["insufficient_evidence"] is False
    assert body["citations"] == [
        {
            "document_id": str(document_id),
            "segment_id": str(segment_id),
            "ordinal": 0,
            "original_filename": "lesson.txt",
            "file_type": "txt",
            "score": 0.92,
        }
    ]
    assert retrieval.calls == [
        ("What does the source say?", 6, document_id)
    ]
    assert provider.prompts[0].sources[0].text_content == (
        "The stored source describes retrieval."
    )


def test_answer_endpoint_abstains_for_no_results(
    answer_client: tuple[
        TestClient, sessionmaker[Session], FakeRetrievalService, FakeAnswerProvider
    ],
) -> None:
    client, _, retrieval, provider = answer_client

    response = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "No evidence?"},
    )

    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert response.json()["citations"] == []
    assert provider.prompts == []
    assert retrieval.calls == [("No evidence?", 10, None)]


def test_answer_endpoint_abstains_if_retrieved_segment_is_missing(
    answer_client: tuple[
        TestClient, sessionmaker[Session], FakeRetrievalService, FakeAnswerProvider
    ],
) -> None:
    client, _, retrieval, provider = answer_client
    retrieval.matches = (
        make_match(DOCUMENT_ID, UUID("00000000-0000-0000-0000-000000000099")),
    )

    response = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "Question?"},
    )

    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert response.json()["citations"] == []
    assert provider.prompts == []


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": " \t\n "},
        {"query": "query", "top_k": 0},
        {"query": "query", "top_k": 101},
        {"query": "query", "top_k": True},
        {"query": "query", "document_id": "not-a-uuid"},
        {"query": "query", "unexpected": "value"},
    ],
)
def test_invalid_answer_request_returns_422(
    answer_client: tuple[
        TestClient, sessionmaker[Session], FakeRetrievalService, FakeAnswerProvider
    ],
    body: dict[str, object],
) -> None:
    client, _, retrieval, provider = answer_client

    response = client.post("/api/v1/retrieval/answer", json=body)

    assert response.status_code == 422
    assert retrieval.calls == []
    assert provider.prompts == []


def test_provider_failure_is_safe_at_api_boundary(
    answer_client: tuple[
        TestClient, sessionmaker[Session], FakeRetrievalService, FakeAnswerProvider
    ],
) -> None:
    client, session_factory, retrieval, provider = answer_client
    query_secret = "sensitive-query-sentinel"
    document_secret = "sensitive-document-sentinel"
    document_id, segment_id = add_source(
        session_factory,
        text_content=document_secret,
    )
    retrieval.matches = (make_match(
        document_id,
        segment_id,
        text_content=document_secret,
    ),)
    provider.failure = RuntimeError(
        "provider error with secret API key, "
        f"{query_secret}, {document_secret}"
    )

    response = client.post(
        "/api/v1/retrieval/answer",
        json={"query": query_secret},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Grounded answer generation failed."
    }
    assert "secret" not in response.text
    assert query_secret not in response.text
    assert document_secret not in response.text


def test_retrieval_failure_is_safe_at_api_boundary(
    answer_client: tuple[
        TestClient, sessionmaker[Session], FakeRetrievalService, FakeAnswerProvider
    ],
) -> None:
    client, _, retrieval, provider = answer_client
    retrieval.failure = VectorStoreError("Pinecone error with secret key")

    response = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "Question?"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Grounded answer generation failed."
    }
    assert "secret" not in response.text
    assert provider.prompts == []


def test_endpoint_reports_unconfigured_llm_without_connecting_to_pinecone() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/retrieval/answer",
            json={"query": "Question?"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Grounded answer generation is not configured."
    }
