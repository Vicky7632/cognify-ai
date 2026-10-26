"""Service-layer integration tests using a manually composed pipeline.

These tests do not prove document upload automatically triggers processing or indexing.
"""

from collections.abc import Generator, Sequence
from hashlib import sha256
from numbers import Real
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document_segment import DocumentSegment
from app.repositories.document_repository import create_document
from app.services.document_embedding_service import DocumentEmbeddingService
from app.services.document_retrieval_service import DocumentRetrievalService
from app.services.document_vector_storage_service import (
    DocumentVectorStorageService,
)
from app.services.grounded_answer_service import (
    GroundedAnswerProviderError,
    GroundedAnswerPrompt,
    GroundedAnswerService,
)
from app.vector_store.protocol import (
    VectorRecord,
    VectorSearchResult,
    VectorStoreError,
)

FIRST_SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000002")
SECOND_SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000003")
MISSING_SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000004")


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'rag-pipeline-integration.db'}",
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def add_document_with_segments(
    session_factory: sessionmaker[Session],
) -> UUID:
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename="integration-guide.txt",
                file_type="txt",
                storage_path="random-storage-key.txt",
            )
            session.add_all(
                [
                    DocumentSegment(
                        id=SECOND_SEGMENT_ID,
                        document_id=document.id,
                        ordinal=1,
                        text_content="Second persisted source.",
                    ),
                    DocumentSegment(
                        id=FIRST_SEGMENT_ID,
                        document_id=document.id,
                        ordinal=0,
                        text_content="First persisted source.",
                    ),
                ]
            )
            document_id = document.id
    return document_id


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        self.batch_calls.append(tuple(texts))
        return tuple(
            (float(index + 1), float(index + 2), float(index + 3))
            for index, _text in enumerate(texts)
        )

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.query_calls.append(text)
        return (0.5, 0.6, 0.7)


class FakeVectorStore:
    def __init__(
        self,
        *,
        fail_upsert: bool = False,
        persist_before_failure: int = 0,
    ) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.upsert_calls: list[tuple[VectorRecord, ...]] = []
        self.search_calls: list[tuple[tuple[float, ...], int, UUID | None]] = []
        self.search_override: tuple[VectorSearchResult, ...] | None = None
        self.fail_upsert = fail_upsert
        self.persist_before_failure = persist_before_failure

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        batch = tuple(records)
        self.upsert_calls.append(batch)
        if self.fail_upsert:
            self.records.update(
                (record.id, record)
                for record in batch[: self.persist_before_failure]
            )
            raise VectorStoreError("Fake vector storage failure.")
        self.records.update((record.id, record) for record in batch)

    def search(
        self,
        query_vector: Sequence[Real],
        *,
        top_k: int,
        document_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self.search_calls.append((tuple(query_vector), top_k, document_id))
        if self.search_override is not None:
            return self.search_override[:top_k]
        matches = [
            _search_result(record, score=0.9 if record.metadata.ordinal == 1 else 0.8)
            for record in self.records.values()
            if document_id is None or record.metadata.document_id == document_id
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return tuple(matches[:top_k])

    def close(self) -> None:
        pass


class FakeGeminiAnswerProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.prompts: list[GroundedAnswerPrompt] = []
        self.fail = fail

    def generate_answer(self, prompt: GroundedAnswerPrompt) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("Fake Gemini generation failure.")
        return "The retrieved segments describe both sources."

    def close(self) -> None:
        pass


def _search_result(record: VectorRecord, *, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        vector_id=record.id,
        document_id=record.metadata.document_id,
        segment_id=record.metadata.segment_id,
        ordinal=record.metadata.ordinal,
        score=score,
        metadata=record.metadata.as_dict(),
    )


def _make_pipeline(
    embedding: FakeEmbeddingService,
    vector_store: FakeVectorStore,
    answer_provider: FakeGeminiAnswerProvider,
) -> tuple[DocumentVectorStorageService, GroundedAnswerService]:
    storage_service = DocumentVectorStorageService(
        DocumentEmbeddingService(embedding),
        vector_store,
    )
    retrieval_service = DocumentRetrievalService(embedding, vector_store)
    answer_service = GroundedAnswerService(retrieval_service, answer_provider)
    return storage_service, answer_service


def test_stored_segments_flow_through_upsert_retrieval_and_grounded_answer(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = add_document_with_segments(session_factory)
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    answer_provider = FakeGeminiAnswerProvider()
    storage_service, answer_service = _make_pipeline(
        embedding,
        vector_store,
        answer_provider,
    )

    with session_factory() as session:
        embedded_segments = storage_service.embed_and_store(session, document_id)
        answer = answer_service.answer(
            session,
            "What do these sources explain?",
            top_k=2,
            document_id=document_id,
        )

    assert embedding.batch_calls == [
        ("First persisted source.", "Second persisted source.")
    ]
    assert [
        (segment.document_id, segment.segment_id, segment.ordinal, segment.vector)
        for segment in embedded_segments
    ] == [
        (document_id, FIRST_SEGMENT_ID, 0, (1.0, 2.0, 3.0)),
        (document_id, SECOND_SEGMENT_ID, 1, (2.0, 3.0, 4.0)),
    ]
    assert [
        (
            record.metadata.document_id,
            record.metadata.segment_id,
            record.metadata.ordinal,
            record.values,
        )
        for record in vector_store.upsert_calls[0]
    ] == [
        (document_id, FIRST_SEGMENT_ID, 0, (1.0, 2.0, 3.0)),
        (document_id, SECOND_SEGMENT_ID, 1, (2.0, 3.0, 4.0)),
    ]
    assert embedding.query_calls == ["What do these sources explain?"]
    assert vector_store.search_calls == [((0.5, 0.6, 0.7), 2, document_id)]
    assert answer.answer == "The retrieved segments describe both sources."
    assert answer.insufficient_evidence is False
    assert [
        (
            citation.document_id,
            citation.segment_id,
            citation.ordinal,
            citation.score,
            citation.original_filename,
        )
        for citation in answer.citations
    ] == [
        (document_id, SECOND_SEGMENT_ID, 1, 0.9, "integration-guide.txt"),
        (document_id, FIRST_SEGMENT_ID, 0, 0.8, "integration-guide.txt"),
    ]
    assert [
        (source.segment_id, source.text_content)
        for source in answer_provider.prompts[0].sources
    ] == [
        (SECOND_SEGMENT_ID, "Second persisted source."),
        (FIRST_SEGMENT_ID, "First persisted source."),
    ]


def test_empty_search_results_abstain_without_calling_answer_provider(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = add_document_with_segments(session_factory)
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    answer_provider = FakeGeminiAnswerProvider()
    storage_service, answer_service = _make_pipeline(
        embedding,
        vector_store,
        answer_provider,
    )

    with session_factory() as session:
        storage_service.embed_and_store(session, document_id)
        vector_store.search_override = ()
        answer = answer_service.answer(session, "No matching source?")

    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer_provider.prompts == []


def test_missing_retrieved_source_does_not_create_a_citation(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = add_document_with_segments(session_factory)
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    answer_provider = FakeGeminiAnswerProvider()
    storage_service, answer_service = _make_pipeline(
        embedding,
        vector_store,
        answer_provider,
    )

    with session_factory() as session:
        storage_service.embed_and_store(session, document_id)
        vector_store.search_override = (
            VectorSearchResult(
                vector_id="unknown-segment-vector",
                document_id=document_id,
                segment_id=MISSING_SEGMENT_ID,
                ordinal=9,
                score=0.99,
                metadata={
                    "document_id": str(document_id),
                    "segment_id": str(MISSING_SEGMENT_ID),
                    "ordinal": 9,
                },
            ),
        )
        answer = answer_service.answer(session, "Question?")

    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer_provider.prompts == []


def test_vector_store_failure_does_not_report_pipeline_success(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = add_document_with_segments(session_factory)
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore(fail_upsert=True)
    answer_provider = FakeGeminiAnswerProvider()
    storage_service, _ = _make_pipeline(
        embedding,
        vector_store,
        answer_provider,
    )

    with session_factory() as session:
        with pytest.raises(VectorStoreError, match="Fake vector storage failure"):
            storage_service.embed_and_store(session, document_id)

    assert len(vector_store.upsert_calls) == 1


def test_failed_reindex_abstains_for_stale_vectors_after_shorter_reprocessing(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = add_document_with_segments(session_factory)
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    answer_provider = FakeGeminiAnswerProvider()
    storage_service, answer_service = _make_pipeline(
        embedding,
        vector_store,
        answer_provider,
    )

    with session_factory() as session:
        storage_service.embed_and_store(session, document_id)

    with session_factory() as session:
        with session.begin():
            first_segment = session.get(DocumentSegment, FIRST_SEGMENT_ID)
            second_segment = session.get(DocumentSegment, SECOND_SEGMENT_ID)
            assert first_segment is not None
            assert second_segment is not None
            first_segment.text_content = "Changed source after reprocessing."
            session.delete(second_segment)

    vector_store.fail_upsert = True
    vector_store.persist_before_failure = 0
    with session_factory() as session:
        with pytest.raises(VectorStoreError):
            storage_service.embed_and_store(session, document_id)

    vector_store.fail_upsert = False
    with session_factory() as session:
        answer = answer_service.answer(
            session,
            "What does the source explain?",
            top_k=2,
            document_id=document_id,
        )

    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer_provider.prompts == []
    assert len(vector_store.records) == 2
    assert sha256(
        b"First persisted source."
    ).hexdigest() == vector_store.records[
        next(
            record_id
            for record_id, record in vector_store.records.items()
            if record.metadata.segment_id == FIRST_SEGMENT_ID
        )
    ].metadata.content_hash


def test_partial_upsert_excludes_stale_vector_from_grounded_answer(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = add_document_with_segments(session_factory)
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    answer_provider = FakeGeminiAnswerProvider()
    storage_service, answer_service = _make_pipeline(
        embedding,
        vector_store,
        answer_provider,
    )

    with session_factory() as session:
        storage_service.embed_and_store(session, document_id)

    with session_factory() as session:
        with session.begin():
            first_segment = session.get(DocumentSegment, FIRST_SEGMENT_ID)
            second_segment = session.get(DocumentSegment, SECOND_SEGMENT_ID)
            assert first_segment is not None
            assert second_segment is not None
            first_segment.text_content = "Updated first source."
            second_segment.text_content = "Updated second source."

    vector_store.fail_upsert = True
    vector_store.persist_before_failure = 1
    with session_factory() as session:
        with pytest.raises(VectorStoreError):
            storage_service.embed_and_store(session, document_id)

    vector_store.fail_upsert = False
    with session_factory() as session:
        answer = answer_service.answer(
            session,
            "What does the source explain?",
            top_k=2,
            document_id=document_id,
        )

    assert answer.insufficient_evidence is False
    assert [citation.segment_id for citation in answer.citations] == [
        FIRST_SEGMENT_ID
    ]
    assert len(answer_provider.prompts) == 1
    assert [
        source.text_content for source in answer_provider.prompts[0].sources
    ] == ["Updated first source."]


def test_answer_provider_failure_is_replaced_with_safe_error(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = add_document_with_segments(session_factory)
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    answer_provider = FakeGeminiAnswerProvider(fail=True)
    storage_service, answer_service = _make_pipeline(
        embedding,
        vector_store,
        answer_provider,
    )

    with session_factory() as session:
        storage_service.embed_and_store(session, document_id)
        with pytest.raises(
            GroundedAnswerProviderError,
            match="^The grounded answer provider request failed\\.$",
        ):
            answer_service.answer(session, "Question?")
