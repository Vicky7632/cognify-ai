from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.models.document import (
    Document,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.repositories.document_repository import (
    create_document,
    delete_document,
    get_document,
    list_documents,
    update_document_status,
)


@pytest.fixture
def database_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_create_and_retrieve_document(database_session: Session) -> None:
    created = create_document(
        database_session,
        original_filename="lesson.pdf",
        file_type="pdf",
        storage_path="storage/documents/lesson.pdf",
    )

    assert isinstance(created.id, UUID)
    assert created.processing_status == DocumentStatus.PENDING.value
    assert get_document(database_session, created.id) is created
    assert created.created_at is not None
    assert created.updated_at is not None


def test_list_documents_uses_newest_first_and_uuid_tie_breaker(
    database_session: Session,
) -> None:
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    earlier = Document(
        original_filename="earlier.txt",
        file_type="txt",
        storage_path="storage/earlier.txt",
        created_at=base_time,
    )
    tied_first = Document(
        original_filename="tied-first.txt",
        file_type="txt",
        storage_path="storage/tied-first.txt",
        created_at=base_time + timedelta(days=1),
    )
    tied_second = Document(
        original_filename="tied-second.txt",
        file_type="txt",
        storage_path="storage/tied-second.txt",
        created_at=base_time + timedelta(days=1),
    )
    database_session.add_all([tied_second, earlier, tied_first])
    database_session.flush()
    expected_tie_order = sorted((tied_first, tied_second), key=lambda doc: doc.id)

    assert list_documents(database_session) == [*expected_tie_order, earlier]


def test_update_document_status_compares_current_status_and_persists_error_code(
    database_session: Session,
) -> None:
    document = create_document(
        database_session,
        original_filename="lesson.pdf",
        file_type="pdf",
        storage_path="storage/lesson.pdf",
        processing_status=DocumentStatus.PROCESSING,
    )

    updated, transitioned = update_document_status(
        database_session,
        document.id,
        DocumentStatus.FAILED,
        expected_statuses=(DocumentStatus.PROCESSING,),
        error_code=DocumentProcessingErrorCode.PROCESSING_FAILED,
    )

    assert transitioned is True
    assert updated is document
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"


def test_update_document_status_clears_error_code_on_retry(
    database_session: Session,
) -> None:
    document = create_document(
        database_session,
        original_filename="lesson.pdf",
        file_type="pdf",
        storage_path="storage/lesson.pdf",
        processing_status=DocumentStatus.FAILED,
    )
    document.processing_error_code = DocumentProcessingErrorCode.PROCESSING_FAILED.value
    database_session.flush()

    updated, transitioned = update_document_status(
        database_session,
        document.id,
        DocumentStatus.PROCESSING,
        expected_statuses=(DocumentStatus.FAILED,),
        error_code=None,
    )

    assert transitioned is True
    assert updated is document
    assert document.processing_status == DocumentStatus.PROCESSING.value
    assert document.processing_error_code is None


def test_delete_existing_document(database_session: Session) -> None:
    document = create_document(
        database_session,
        original_filename="lesson.pdf",
        file_type="pdf",
        storage_path="storage/lesson.pdf",
    )

    assert delete_document(database_session, document.id) is True
    assert get_document(database_session, document.id) is None


def test_nonexistent_document_operations_return_consistent_results(
    database_session: Session,
) -> None:
    missing_id = UUID("00000000-0000-0000-0000-000000000001")

    assert get_document(database_session, missing_id) is None
    assert (
        update_document_status(
            database_session,
            missing_id,
            DocumentStatus.COMPLETED,
            expected_statuses=(DocumentStatus.PROCESSING,),
            error_code=None,
        )
        == (None, False)
    )
    assert delete_document(database_session, missing_id) is False


def test_invalid_status_is_rejected_without_changing_document(
    database_session: Session,
) -> None:
    document = create_document(
        database_session,
        original_filename="lesson.pdf",
        file_type="pdf",
        storage_path="storage/lesson.pdf",
    )

    updated, transitioned = update_document_status(
        database_session,
        document.id,
        DocumentStatus.COMPLETED,
        expected_statuses=(DocumentStatus.PROCESSING,),
        error_code=None,
    )

    assert transitioned is False
    assert updated is document
    assert document.processing_status == DocumentStatus.PENDING.value


def test_repository_does_not_commit_caller_transaction(
    database_session: Session,
) -> None:
    created = create_document(
        database_session,
        original_filename="lesson.pdf",
        file_type="pdf",
        storage_path="storage/lesson.pdf",
    )

    assert database_session.in_transaction()
    database_session.rollback()
    assert get_document(database_session, created.id) is None
