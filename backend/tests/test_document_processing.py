from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document import (
    Document,
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.models.document_segment import DocumentSegment
from app.repositories import document_segment_repository
from app.services import document_service
from app.services.document_processing_service import (
    DocumentNotFoundError,
    DocumentProcessingConflictError,
    DocumentProcessingFailedError,
    DocumentProcessingPersistenceError,
    process_document,
)
from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessingInput,
    DocumentProcessor,
    ProcessedSegment,
    ProcessingResult,
)


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'processing-tests.db'}",
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_document(
    session_factory: Callable[[], Session],
    *,
    status: DocumentStatus = DocumentStatus.UPLOADED,
    error_code: DocumentProcessingErrorCode | None = None,
    file_type: DocumentFileType = DocumentFileType.PDF,
) -> UUID:
    with session_factory() as session:
        with session.begin():
            document = document_service.create_document(
                session,
                original_filename="lesson.pdf",
                file_type=file_type,
                storage_path=f"random-storage-key.{file_type.value}",
                processing_status=status,
            )
            document.processing_error_code = (
                error_code.value if error_code is not None else None
            )
            session.flush()
            return document.id


class ReturningProcessor:
    def __init__(self, result: ProcessingResult) -> None:
        self.result = result
        self.input: DocumentProcessingInput | None = None

    def process(self, document: DocumentProcessingInput) -> ProcessingResult:
        self.input = document
        return self.result


def get_document(
    session_factory: Callable[[], Session],
    document_id: UUID,
) -> Document:
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document is not None
        return document


def test_processing_result_contains_valid_text_and_provenance() -> None:
    result = ProcessingResult(
        segments=(
            ProcessedSegment("Page text", page_number=2),
            ProcessedSegment("Timed text", start_seconds=1.5, end_seconds=3.0),
        )
    )

    assert result.segments[0].page_number == 2
    assert result.segments[1].start_seconds == 1.5


def test_processing_result_rejects_empty_segments() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        ProcessingResult(segments=())


def test_processor_contract_completes_only_with_valid_result(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)
    result = ProcessingResult(
        segments=(
            ProcessedSegment("Later page", page_number=3),
            ProcessedSegment("Earlier page", page_number=1),
        )
    )
    processor = ReturningProcessor(result)

    returned = process_document(session_factory, document_id, processor)

    assert returned == result
    assert processor.input is not None
    assert processor.input.document_id == document_id
    assert processor.input.storage_key == "random-storage-key.pdf"
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.COMPLETED.value
    assert document.processing_error_code is None
    with session_factory() as session:
        segments = session.scalars(
            select(DocumentSegment)
            .where(DocumentSegment.document_id == document_id)
            .order_by(DocumentSegment.ordinal)
        ).all()
    assert [
        (segment.ordinal, segment.text_content, segment.page_number)
        for segment in segments
    ] == [
        (0, "Later page", 3),
        (1, "Earlier page", 1),
    ]


def test_processor_runs_without_an_open_database_transaction(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)
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

    class TransactionCheckingProcessor:
        def process(self, document: DocumentProcessingInput) -> ProcessingResult:
            assert active_transactions == 0
            return ProcessingResult(
                segments=(ProcessedSegment("Processed text", page_number=1),)
            )

    try:
        process_document(
            session_factory,
            document_id,
            TransactionCheckingProcessor(),
        )
    finally:
        event.remove(engine, "begin", transaction_started)
        event.remove(engine, "commit", transaction_ended)
        event.remove(engine, "rollback", transaction_ended)

    assert active_transactions == 0


def test_txt_segments_persist_without_source_locations(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(
        session_factory,
        file_type=DocumentFileType.TXT,
    )
    result = ProcessingResult(
        segments=(
            ProcessedSegment("First text segment"),
            ProcessedSegment("Second text segment"),
        )
    )

    process_document(session_factory, document_id, ReturningProcessor(result))

    with session_factory() as session:
        segments = session.scalars(
            select(DocumentSegment)
            .where(DocumentSegment.document_id == document_id)
            .order_by(DocumentSegment.ordinal)
        ).all()
    assert [(segment.ordinal, segment.page_number) for segment in segments] == [
        (0, None),
        (1, None),
    ]
    assert all(
        segment.start_seconds is None and segment.end_seconds is None
        for segment in segments
    )


def test_invalid_pdf_provenance_fails_without_persisting_segments(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)
    processor = ReturningProcessor(
        ProcessingResult(segments=(ProcessedSegment("Missing page reference"),))
    )

    with pytest.raises(DocumentProcessingFailedError):
        process_document(session_factory, document_id, processor)

    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
    with session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []


def test_expected_processor_failure_is_persisted_with_safe_code(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)

    class FailingProcessor:
        def process(self, document: DocumentProcessingInput) -> ProcessingResult:
            raise DocumentProcessingError(
                DocumentProcessingErrorCode.PROCESSING_FAILED
            )

    with pytest.raises(DocumentProcessingFailedError) as raised:
        process_document(session_factory, document_id, FailingProcessor())

    assert str(raised.value) == "Document processing failed."
    assert raised.value.code is DocumentProcessingErrorCode.PROCESSING_FAILED
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
    with session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []


def test_unexpected_processor_exception_is_sanitized_and_marks_failed(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)
    sensitive_detail = r"secret C:\private\storage file contents"

    class CrashingProcessor:
        def process(self, document: DocumentProcessingInput) -> ProcessingResult:
            raise RuntimeError(sensitive_detail)

    with pytest.raises(DocumentProcessingFailedError) as raised:
        process_document(session_factory, document_id, CrashingProcessor())

    assert sensitive_detail not in str(raised.value)
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    with session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []
    assert document.processing_error_code == "processing_failed"


def test_invalid_processor_result_does_not_mark_document_completed(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)

    class InvalidResultProcessor:
        def process(self, document: DocumentProcessingInput) -> object:
            return None

    with pytest.raises(DocumentProcessingFailedError) as raised:
        process_document(
            session_factory,
            document_id,
            cast(DocumentProcessor, InvalidResultProcessor()),
        )

    assert str(raised.value) == "Document processing failed."
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value


def test_missing_document_does_not_call_processor(
    session_factory: sessionmaker[Session],
) -> None:
    processor = ReturningProcessor(
        ProcessingResult(segments=(ProcessedSegment("Processed text"),))
    )

    with pytest.raises(DocumentNotFoundError, match="Document not found"):
        process_document(
            session_factory,
            UUID("00000000-0000-0000-0000-000000000001"),
            processor,
        )

    assert processor.input is None


def test_ineligible_lifecycle_state_is_rejected(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(
        session_factory,
        status=DocumentStatus.COMPLETED,
    )
    processor = ReturningProcessor(
        ProcessingResult(
            segments=(ProcessedSegment("Processed text", page_number=1),)
        )
    )

    with pytest.raises(DocumentProcessingConflictError):
        process_document(session_factory, document_id, processor)

    assert processor.input is None
    assert get_document(session_factory, document_id).processing_status == "completed"


def test_failed_document_can_be_retried_and_old_error_is_cleared(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(
        session_factory,
        status=DocumentStatus.FAILED,
        error_code=DocumentProcessingErrorCode.PROCESSING_FAILED,
    )
    processor = ReturningProcessor(
        ProcessingResult(
            segments=(ProcessedSegment("Retried text", page_number=1),)
        )
    )

    process_document(session_factory, document_id, processor)

    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.COMPLETED.value
    assert document.processing_error_code is None
    with session_factory() as session:
        segments = session.scalars(
            select(DocumentSegment).where(DocumentSegment.document_id == document_id)
        ).all()
    assert [(segment.ordinal, segment.text_content) for segment in segments] == [
        (0, "Retried text")
    ]


def test_stale_completion_does_not_overwrite_newer_failure(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)
    result = ProcessingResult(
        segments=(ProcessedSegment("Processed text", page_number=1),)
    )

    class ConcurrentFailureProcessor:
        def process(self, document: DocumentProcessingInput) -> ProcessingResult:
            with session_factory() as session:
                with session.begin():
                    document_service.update_document_status(
                        session,
                        document_id,
                        DocumentStatus.FAILED,
                    )
            return result

    with pytest.raises(DocumentProcessingConflictError):
        process_document(session_factory, document_id, ConcurrentFailureProcessor())

    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
    with session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []


def test_segment_persistence_failure_rolls_back_completion_and_partial_rows(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = create_document(session_factory)
    result = ProcessingResult(
        segments=(
            ProcessedSegment("First persisted row", page_number=1),
            ProcessedSegment("Second persisted row", page_number=2),
        )
    )
    replace_segments = document_segment_repository.replace_document_segments

    def fail_after_first_segment(
        session: Session,
        target_document_id: UUID,
        segments: tuple[DocumentSegment, ...],
    ) -> None:
        replace_segments(session, target_document_id, segments[:1])
        session.add(
            DocumentSegment(
                document_id=target_document_id,
                ordinal=1,
                text_content=" \t ",
                page_number=2,
            )
        )
        session.flush()

    monkeypatch.setattr(
        document_segment_repository,
        "replace_document_segments",
        fail_after_first_segment,
    )

    with pytest.raises(DocumentProcessingPersistenceError):
        process_document(session_factory, document_id, ReturningProcessor(result))

    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.PROCESSING.value
    assert document.processing_error_code is None
    with session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []
