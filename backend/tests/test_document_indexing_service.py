from collections.abc import Callable, Generator, Sequence
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document import (
    Document,
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.models.document_processing_job import (
    DocumentProcessingJob,
    DocumentProcessingJobStatus,
)
from app.models.document_segment import DocumentSegment
from app.repositories import document_processing_job_repository
from app.repositories.document_repository import create_document
from app.services.document_embedding_service import DocumentEmbeddingService
from app.services.document_indexing_service import (
    DocumentIndexingFailedError,
    DocumentIndexingService,
    DocumentWorkerLockLostError,
)
from app.services.document_processing_worker import process_next_document_job
from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessingInput,
    ProcessedSegment,
    ProcessingResult,
)
from app.services.document_vector_storage_service import (
    DocumentVectorStorageService,
)
from app.services.embedding_service import EmbeddingProviderError
from app.vector_store.protocol import (
    VectorRecord,
    VectorSearchResult,
    VectorStoreError,
)


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'document-indexing-tests.db'}",
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_uploaded_document(session_factory: sessionmaker[Session]) -> UUID:
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename="lesson.txt",
                file_type=DocumentFileType.TXT.value,
                storage_path="generated-key.txt",
                processing_status=DocumentStatus.UPLOADED,
            )
            return document.id


class FakeProcessor:
    def __init__(
        self,
        result: ProcessingResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.result = result or ProcessingResult(
            segments=(
                ProcessedSegment("First persisted section."),
                ProcessedSegment("Second persisted section."),
            )
        )
        self.failure = failure

    def process(self, document: DocumentProcessingInput) -> ProcessingResult:
        if self.failure is not None:
            raise self.failure
        return self.result


class FakeEmbeddingService:
    def __init__(
        self,
        failure: Exception | None = None,
        before_embed: Callable[[], None] | None = None,
    ) -> None:
        self.failure = failure
        self.before_embed = before_embed
        self.calls: list[tuple[str, ...]] = []
        self.closed = False

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        if self.before_embed is not None:
            self.before_embed()
        self.calls.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        return tuple((float(index), 1.0) for index in range(len(texts)))

    def close(self) -> None:
        self.closed = True


class FakeVectorStore:
    def __init__(
        self,
        failure: Exception | None = None,
        before_upsert: Callable[[], None] | None = None,
    ) -> None:
        self.failure = failure
        self.before_upsert = before_upsert
        self.calls: list[tuple[VectorRecord, ...]] = []
        self.closed = False

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if self.before_upsert is not None:
            self.before_upsert()
        self.calls.append(tuple(records))
        if self.failure is not None:
            raise self.failure

    def search(
        self,
        query_vector: Sequence[Real],
        *,
        top_k: int,
        document_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


def make_indexing_service(
    session_factory: sessionmaker[Session],
    *,
    processor: FakeProcessor | None = None,
    embedding_failure: Exception | None = None,
    vector_failure: Exception | None = None,
) -> tuple[DocumentIndexingService, FakeEmbeddingService, FakeVectorStore]:
    embedding = FakeEmbeddingService(embedding_failure)
    vector_store = FakeVectorStore(vector_failure)
    service = DocumentIndexingService(
        session_factory,
        processor or FakeProcessor(),
        lambda: DocumentVectorStorageService(
            DocumentEmbeddingService(embedding),
            vector_store,
        ),
    )
    return service, embedding, vector_store


def get_document(
    session_factory: sessionmaker[Session],
    document_id: UUID,
) -> Document:
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document is not None
        return document


def get_segments(
    session_factory: sessionmaker[Session],
    document_id: UUID,
) -> list[DocumentSegment]:
    with session_factory() as session:
        return list(
            session.scalars(
                select(DocumentSegment)
                .where(DocumentSegment.document_id == document_id)
                .order_by(DocumentSegment.ordinal)
            )
        )


def assert_no_active_transaction(active_transactions: int) -> None:
    assert active_transactions == 0


def test_processing_embedding_and_upsert_complete_document_only_after_success(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    service, embedding, vector_store = make_indexing_service(session_factory)

    service.process_and_index(document_id)

    document = get_document(session_factory, document_id)
    segments = get_segments(session_factory, document_id)
    assert document.processing_status == DocumentStatus.COMPLETED.value
    assert document.processing_error_code is None
    assert [segment.text_content for segment in segments] == [
        "First persisted section.",
        "Second persisted section.",
    ]
    assert embedding.calls == [
        ("First persisted section.", "Second persisted section.")
    ]
    assert [
        (record.metadata.document_id, record.metadata.segment_id, record.metadata.ordinal)
        for record in vector_store.calls[0]
    ] == [
        (document_id, segments[0].id, 0),
        (document_id, segments[1].id, 1),
    ]
    assert embedding.closed is True
    assert vector_store.closed is True


def test_external_embedding_and_upsert_run_without_open_database_transaction(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    engine = session_factory.kw["bind"]
    assert engine is not None
    active_transactions = 0

    def transaction_started(_connection) -> None:
        nonlocal active_transactions
        active_transactions += 1

    def transaction_ended(_connection) -> None:
        nonlocal active_transactions
        active_transactions -= 1

    event.listen(engine, "begin", transaction_started)
    event.listen(engine, "commit", transaction_ended)
    event.listen(engine, "rollback", transaction_ended)
    embedding = FakeEmbeddingService(
        before_embed=lambda: assert_no_active_transaction(active_transactions)
    )
    vector_store = FakeVectorStore(
        before_upsert=lambda: assert_no_active_transaction(active_transactions)
    )
    service = DocumentIndexingService(
        session_factory,
        FakeProcessor(),
        lambda: DocumentVectorStorageService(
            DocumentEmbeddingService(embedding),
            vector_store,
        ),
    )

    try:
        service.process_and_index(document_id)
    finally:
        event.remove(engine, "begin", transaction_started)
        event.remove(engine, "commit", transaction_ended)
        event.remove(engine, "rollback", transaction_ended)

    assert active_transactions == 0


def test_extraction_failure_marks_document_failed_without_embedding(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    processor = FakeProcessor(
        failure=DocumentProcessingError(DocumentProcessingErrorCode.PROCESSING_FAILED)
    )
    service, embedding, vector_store = make_indexing_service(
        session_factory,
        processor=processor,
    )

    with pytest.raises(
        DocumentIndexingFailedError,
        match="Document processing and indexing failed",
    ):
        service.process_and_index(document_id)

    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
    assert get_segments(session_factory, document_id) == []
    assert embedding.calls == []
    assert vector_store.calls == []


@pytest.mark.parametrize(
    ("embedding_failure", "vector_failure"),
    [
        (EmbeddingProviderError("provider secret detail"), None),
        (None, VectorStoreError("vector provider secret detail")),
    ],
)
def test_embedding_or_upsert_failure_never_marks_document_completed(
    session_factory: sessionmaker[Session],
    embedding_failure: Exception | None,
    vector_failure: Exception | None,
) -> None:
    document_id = create_uploaded_document(session_factory)
    service, embedding, vector_store = make_indexing_service(
        session_factory,
        embedding_failure=embedding_failure,
        vector_failure=vector_failure,
    )

    with pytest.raises(DocumentIndexingFailedError) as error:
        service.process_and_index(document_id)

    assert str(error.value) == "Document processing and indexing failed."
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
    assert len(get_segments(session_factory, document_id)) == 2
    assert "secret detail" not in str(error.value)
    if embedding_failure is not None:
        assert vector_store.calls == []
    else:
        assert len(vector_store.calls) == 1


def test_reprocessing_reuses_segment_ids_and_vector_ids(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    first_service, _, first_store = make_indexing_service(
        session_factory,
        vector_failure=VectorStoreError("temporary failure"),
    )
    with pytest.raises(DocumentIndexingFailedError):
        first_service.process_and_index(document_id)
    original_segment_ids = [segment.id for segment in get_segments(
        session_factory,
        document_id,
    )]

    retry_service, _, retry_store = make_indexing_service(session_factory)
    retry_service.process_and_index(document_id)
    retried_segment_ids = [segment.id for segment in get_segments(
        session_factory,
        document_id,
    )]

    assert retried_segment_ids == original_segment_ids
    assert [
        record.id for record in first_store.calls[0]
    ] == [record.id for record in retry_store.calls[0]]
    assert get_document(session_factory, document_id).processing_status == (
        DocumentStatus.COMPLETED.value
    )


def test_durable_job_worker_marks_job_complete_after_indexing(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id
    service, _, _ = make_indexing_service(session_factory)

    processed = process_next_document_job(
        session_factory,
        lambda _document_id: service,
    )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.COMPLETED.value
        assert job.attempts == 1
    assert processed is True
    assert get_document(session_factory, document_id).processing_status == (
        DocumentStatus.COMPLETED.value
    )


def test_lock_loss_during_failed_embedding_skips_failure_and_retry_writes(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id

    lock_lost = False

    def ensure_worker_lock() -> None:
        if lock_lost:
            raise DocumentWorkerLockLostError("Worker lock was lost.")

    def lose_lock_during_embedding() -> None:
        nonlocal lock_lost
        lock_lost = True

    embedding = FakeEmbeddingService(
        failure=EmbeddingProviderError("provider operation failed"),
        before_embed=lose_lock_during_embedding,
    )
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        session_factory,
        FakeProcessor(),
        lambda: DocumentVectorStorageService(
            DocumentEmbeddingService(embedding),
            vector_store,
        ),
        ensure_worker_lock=ensure_worker_lock,
    )

    with pytest.raises(DocumentWorkerLockLostError, match="Worker lock was lost"):
        process_next_document_job(
            session_factory,
            lambda _document_id: service,
            ensure_worker_lock=ensure_worker_lock,
        )

    document = get_document(session_factory, document_id)
    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None

    assert document.processing_status == DocumentStatus.PROCESSING.value
    assert document.processing_error_code is None
    assert job.status == DocumentProcessingJobStatus.PROCESSING.value
    assert job.attempts == 1
    assert embedding.calls == [("First persisted section.", "Second persisted section.")]
    assert vector_store.calls == []
    assert embedding.closed is True
    assert vector_store.closed is True


def test_worker_startup_queues_legacy_uploaded_documents_only_once(
    session_factory: sessionmaker[Session],
) -> None:
    queued_document_id = create_uploaded_document(session_factory)
    unqueued_document_id = create_uploaded_document(session_factory)

    with session_factory() as session:
        with session.begin():
            document_processing_job_repository.enqueue_document_processing(
                session,
                queued_document_id,
            )
    with session_factory() as session:
        with session.begin():
            queued_count = (
                document_processing_job_repository
                .enqueue_unqueued_uploaded_documents(session)
            )
    with session_factory() as session:
        jobs = list(session.scalars(select(DocumentProcessingJob)))

    assert queued_count == 1
    assert {job.document_id for job in jobs} == {
        queued_document_id,
        unqueued_document_id,
    }


def test_worker_failure_queues_retry_and_keeps_document_failed(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id
    service, _, _ = make_indexing_service(
        session_factory,
        vector_failure=VectorStoreError("provider details"),
    )

    assert process_next_document_job(
        session_factory,
        lambda _document_id: service,
    )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.QUEUED.value
        assert job.attempts == 1
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"


def test_worker_stops_retrying_after_maximum_attempts(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id
    service, _, _ = make_indexing_service(
        session_factory,
        vector_failure=VectorStoreError("provider details"),
    )

    for _ in range(
        document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
    ):
        assert process_next_document_job(
            session_factory,
            lambda _document_id: service,
        )
        with session_factory() as session:
            with session.begin():
                job = session.get(DocumentProcessingJob, job_id)
                assert job is not None
                job.available_at = datetime.now(UTC)

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.FAILED.value
        assert job.attempts == (
            document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
        )


def test_interrupted_job_recovery_requeues_work_and_fails_stale_document(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            document_service_result = session.get(Document, document_id)
            assert document_service_result is not None
            document_service_result.processing_status = DocumentStatus.PROCESSING.value
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job.status = DocumentProcessingJobStatus.PROCESSING.value
            job.attempts = 1
            job_id = job.id

    with session_factory() as session:
        with session.begin():
            recovered = (
                document_processing_job_repository
                .recover_interrupted_document_processing_jobs(session)
            )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.QUEUED.value
    document = get_document(session_factory, document_id)
    assert recovered == 1
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"


@pytest.mark.parametrize(
    "document_status",
    [DocumentStatus.UPLOADED, DocumentStatus.PROCESSING],
)
def test_exhausted_interrupted_job_fails_job_and_document_safely(
    session_factory: sessionmaker[Session],
    document_status: DocumentStatus,
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            document = session.get(Document, document_id)
            assert document is not None
            document.processing_status = document_status.value
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job.status = DocumentProcessingJobStatus.PROCESSING.value
            job.attempts = (
                document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
            )
            job_id = job.id

    with session_factory() as session:
        with session.begin():
            recovered = (
                document_processing_job_repository
                .recover_interrupted_document_processing_jobs(session)
            )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.FAILED.value
        assert job.attempts == (
            document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
        )
    document = get_document(session_factory, document_id)
    assert recovered == 1
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
