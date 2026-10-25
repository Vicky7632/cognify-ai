from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document_segment import DocumentSegment
from app.repositories.document_repository import create_document
from app.services.document_embedding_service import (
    DocumentEmbeddingNotFoundError,
    DocumentEmbeddingService,
)
from app.services.document_vector_storage_service import (
    DocumentVectorStorageService,
)
from app.services.embedding_service import EmbeddingInputError
from app.vector_store.protocol import (
    VectorRecord,
    VectorStoreError,
)


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'document-vector-storage-tests.db'}",
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_document_with_segments(
    session_factory: Callable[[], Session],
    texts: Sequence[str],
) -> tuple[UUID, dict[int, UUID]]:
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename="lesson.txt",
                file_type="txt",
                storage_path="random-storage-key.txt",
            )
            segments = [
                DocumentSegment(
                    document_id=document.id,
                    ordinal=ordinal,
                    text_content=text,
                )
                for ordinal, text in reversed(tuple(enumerate(texts)))
            ]
            session.add_all(segments)
            session.flush()
            ids = {segment.ordinal: segment.id for segment in segments}
            document_id = document.id
    return document_id, ids


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return tuple(
            (float(index), float(index + 1), float(index + 2))
            for index in range(len(texts))
        )


class FakeVectorStore:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[VectorRecord, ...]] = []

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        self.calls.append(tuple(records))
        if self.failure is not None:
            raise self.failure

    def close(self) -> None:
        pass


def test_pipeline_embeddings_are_mapped_and_stored_in_ordinal_order(
    session_factory: sessionmaker[Session],
) -> None:
    document_id, segment_ids = create_document_with_segments(
        session_factory,
        ("first text", "second text", "third text"),
    )
    embedder = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    service = DocumentVectorStorageService(
        DocumentEmbeddingService(embedder),
        vector_store,
    )

    with session_factory() as session:
        results = service.embed_and_store(session, document_id)

    assert embedder.calls == [("first text", "second text", "third text")]
    assert [
        (record.metadata.document_id, record.metadata.segment_id, record.metadata.ordinal)
        for record in vector_store.calls[0]
    ] == [
        (document_id, segment_ids[0], 0),
        (document_id, segment_ids[1], 1),
        (document_id, segment_ids[2], 2),
    ]
    assert [result.vector for result in results] == [
        (0.0, 1.0, 2.0),
        (1.0, 2.0, 3.0),
        (2.0, 3.0, 4.0),
    ]


def test_missing_document_is_not_sent_to_vector_store(
    session_factory: sessionmaker[Session],
) -> None:
    embedder = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    service = DocumentVectorStorageService(
        DocumentEmbeddingService(embedder),
        vector_store,
    )

    with session_factory() as session:
        with pytest.raises(DocumentEmbeddingNotFoundError):
            service.embed_and_store(
                session,
                UUID("00000000-0000-0000-0000-000000000099"),
            )

    assert embedder.calls == []
    assert vector_store.calls == []


def test_document_without_segments_is_not_sent_to_vector_store(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename="empty.txt",
                file_type="txt",
                storage_path="random-storage-key.txt",
            )
            document_id = document.id
    embedder = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    service = DocumentVectorStorageService(
        DocumentEmbeddingService(embedder),
        vector_store,
    )

    with session_factory() as session:
        with pytest.raises(EmbeddingInputError, match="no persisted segments"):
            service.embed_and_store(session, document_id)

    assert embedder.calls == []
    assert vector_store.calls == []


def test_storage_failure_propagates_instead_of_reporting_success(
    session_factory: sessionmaker[Session],
) -> None:
    document_id, _ = create_document_with_segments(
        session_factory,
        ("segment text",),
    )
    embedder = FakeEmbeddingService()
    vector_store = FakeVectorStore(failure=VectorStoreError("safe failure"))
    service = DocumentVectorStorageService(
        DocumentEmbeddingService(embedder),
        vector_store,
    )

    with session_factory() as session:
        with pytest.raises(VectorStoreError, match="safe failure"):
            service.embed_and_store(session, document_id)

    assert embedder.calls == [("segment text",)]
    assert len(vector_store.calls) == 1
