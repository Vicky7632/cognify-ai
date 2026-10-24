from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document_segment import DocumentSegment
from app.repositories.document_repository import create_document
from app.repositories.document_segment_repository import list_document_segments
from app.services.embedding_service import (
    EmbeddingInputError,
    EmbeddingOutputError,
    EmbeddingProviderError,
)
from app.services.document_embedding_service import (
    DocumentEmbeddingNotFoundError,
    DocumentEmbeddingService,
)


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'document-embedding-tests.db'}",
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_test_document(session_factory: Callable[[], Session]) -> UUID:
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename="lesson.txt",
                file_type="txt",
                storage_path="random-storage-key.txt",
            )
            return document.id


def add_segments(
    session_factory: Callable[[], Session],
    document_id: UUID,
    texts: Sequence[str],
) -> dict[int, UUID]:
    segments = [
        DocumentSegment(
            document_id=document_id,
            ordinal=ordinal,
            text_content=text,
        )
        for ordinal, text in reversed(tuple(enumerate(texts)))
    ]
    with session_factory() as session:
        with session.begin():
            session.add_all(segments)
            session.flush()
            segment_ids = {
                segment.ordinal: segment.id
                for segment in segments
            }
    return segment_ids


class FakeEmbeddingService:
    def __init__(
        self,
        *,
        max_batch_size: int = 2,
        failure: Exception | None = None,
        vectors: tuple[tuple[float, ...], ...] | None = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.failure = failure
        self.vectors = vectors
        self.batches: list[tuple[str, ...]] = []

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        self.batches.extend(
            tuple(texts[start : start + self.max_batch_size])
            for start in range(0, len(texts), self.max_batch_size)
        )
        if self.failure is not None:
            raise self.failure
        if self.vectors is not None:
            return self.vectors
        return tuple(
            (float(index), float(index + 1))
            for index in range(len(texts))
        )


def test_embeds_segments_in_stable_ordinal_order_and_preserves_association(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_test_document(session_factory)
    persisted = add_segments(
        session_factory,
        document_id,
        ("first text", "second text", "third text"),
    )
    embedder = FakeEmbeddingService()
    service = DocumentEmbeddingService(embedder)

    with session_factory() as session:
        results = service.embed_document(session, document_id)

    assert embedder.batches == [("first text", "second text"), ("third text",)]
    assert [
        (result.document_id, result.segment_id, result.ordinal)
        for result in results
    ] == [
        (document_id, persisted[0], 0),
        (document_id, persisted[1], 1),
        (document_id, persisted[2], 2),
    ]
    assert [result.vector for result in results] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
    ]


def test_segment_repository_returns_all_segments_in_ordinal_order(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_test_document(session_factory)
    with session_factory() as session:
        with session.begin():
            session.add_all(
                [
                    DocumentSegment(
                        document_id=document_id,
                        ordinal=2,
                        text_content="last",
                    ),
                    DocumentSegment(
                        document_id=document_id,
                        ordinal=0,
                        text_content="first",
                    ),
                    DocumentSegment(
                        document_id=document_id,
                        ordinal=1,
                        text_content="middle",
                    ),
                ]
            )
        segments = list_document_segments(session, document_id)

    assert [segment.ordinal for segment in segments] == [0, 1, 2]
    assert [segment.text_content for segment in segments] == [
        "first",
        "middle",
        "last",
    ]


def test_document_without_segments_raises_without_calling_embedder(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_test_document(session_factory)
    embedder = FakeEmbeddingService()
    service = DocumentEmbeddingService(embedder)

    with session_factory() as session:
        with pytest.raises(EmbeddingInputError, match="no persisted segments"):
            service.embed_document(session, document_id)

    assert embedder.batches == []


def test_missing_document_raises_not_found(
    session_factory: sessionmaker[Session],
) -> None:
    service = DocumentEmbeddingService(FakeEmbeddingService())

    with session_factory() as session:
        with pytest.raises(DocumentEmbeddingNotFoundError, match="not found"):
            service.embed_document(
                session,
                UUID("00000000-0000-0000-0000-000000000001"),
            )


@pytest.mark.parametrize(
    ("ordinal", "text", "page_number", "start_seconds", "end_seconds"),
    [
        (-1, "malformed", None, None, None),
        (0, "   ", None, None, None),
        (0, "malformed", 0, None, None),
        (0, "malformed", None, 1.0, None),
        (0, "malformed", None, 2.0, 1.0),
    ],
)
def test_malformed_persisted_segment_is_rejected_before_embedding(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    ordinal: int,
    text: str,
    page_number: int | None,
    start_seconds: float | None,
    end_seconds: float | None,
) -> None:
    document_id = create_test_document(session_factory)
    embedder = FakeEmbeddingService()
    malformed_segment = SimpleNamespace(
        id=UUID("00000000-0000-0000-0000-000000000002"),
        document_id=document_id,
        ordinal=ordinal,
        text_content=text,
        page_number=page_number,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    monkeypatch.setattr(
        "app.services.document_embedding_service."
        "document_segment_repository.list_document_segments",
        lambda _session, _document_id: [malformed_segment],
    )
    service = DocumentEmbeddingService(embedder)

    with session_factory() as session:
        with pytest.raises(EmbeddingInputError, match="malformed segment"):
            service.embed_document(session, document_id)

    assert embedder.batches == []


def test_embedding_provider_failure_propagates_without_rewriting(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_test_document(session_factory)
    add_segments(session_factory, document_id, ("segment text",))
    provider_error = EmbeddingProviderError("Safe provider failure.")
    service = DocumentEmbeddingService(
        FakeEmbeddingService(failure=provider_error)
    )

    with session_factory() as session:
        with pytest.raises(EmbeddingProviderError) as error:
            service.embed_document(session, document_id)

    assert error.value is provider_error


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (((0.1, 0.2),), "vector count"),
        (((0.1, 0.2), (0.3,)), "inconsistent vector dimensions"),
    ],
)
def test_embedding_count_and_dimension_mismatches_are_rejected(
    session_factory: sessionmaker[Session],
    vectors: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    document_id = create_test_document(session_factory)
    add_segments(session_factory, document_id, ("one", "two"))
    service = DocumentEmbeddingService(
        FakeEmbeddingService(vectors=vectors)
    )

    with session_factory() as session:
        with pytest.raises(EmbeddingOutputError, match=message):
            service.embed_document(session, document_id)
