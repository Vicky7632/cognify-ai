from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document import (
    Document,
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.services import document_service
from app.services.document_processing_service import (
    DocumentNotFoundError,
    DocumentProcessingConflictError,
    DocumentProcessingFailedError,
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
) -> UUID:
    with session_factory() as session:
        with session.begin():
            document = document_service.create_document(
                session,
                original_filename="lesson.pdf",
                file_type=DocumentFileType.PDF,
                storage_path="random-storage-key.pdf",
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
    result = ProcessingResult(segments=(ProcessedSegment("Processed text"),))
    processor = ReturningProcessor(result)

    returned = process_document(session_factory, document_id, processor)

    assert returned == result
    assert processor.input is not None
    assert processor.input.document_id == document_id
    assert processor.input.storage_key == "random-storage-key.pdf"
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.COMPLETED.value
    assert document.processing_error_code is None


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
        ProcessingResult(segments=(ProcessedSegment("Processed text"),))
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
        ProcessingResult(segments=(ProcessedSegment("Retried text"),))
    )

    process_document(session_factory, document_id, processor)

    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.COMPLETED.value
    assert document.processing_error_code is None


def test_stale_completion_does_not_overwrite_newer_failure(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_document(session_factory)
    result = ProcessingResult(segments=(ProcessedSegment("Processed text"),))

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
