from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from app.models.document import (
    Document,
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.services import document_service
from app.services.document_service import InvalidDocumentTransitionError


@pytest.fixture
def session() -> MagicMock:
    return MagicMock(spec=Session)


def test_create_document_delegates_to_repository(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    document = MagicMock(spec=Document)
    create = MagicMock(return_value=document)
    monkeypatch.setattr(document_service.document_repository, "create_document", create)

    result = document_service.create_document(
        session,
        original_filename="lecture.pdf",
        file_type=DocumentFileType.PDF,
        storage_path="storage/lecture.pdf",
    )

    assert result is document
    create.assert_called_once_with(
        session,
        original_filename="lecture.pdf",
        file_type="pdf",
        storage_path="storage/lecture.pdf",
        processing_status=DocumentStatus.PENDING,
    )


def test_create_document_rejects_unsupported_file_type(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    create = MagicMock()
    monkeypatch.setattr(document_service.document_repository, "create_document", create)

    with pytest.raises(ValueError, match="Unsupported document file type"):
        document_service.create_document(
            session,
            original_filename="lecture.docx",
            file_type="docx",
            storage_path="storage/lecture.docx",
        )

    create.assert_not_called()


def test_get_document_delegates_and_preserves_missing_result(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    document_id = UUID("00000000-0000-0000-0000-000000000001")
    get = MagicMock(return_value=None)
    monkeypatch.setattr(document_service.document_repository, "get_document", get)

    result = document_service.get_document(session, document_id)

    assert result is None
    get.assert_called_once_with(session, document_id)


def test_list_documents_delegates_to_repository(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    documents = [MagicMock(spec=Document)]
    list_all = MagicMock(return_value=documents)
    monkeypatch.setattr(
        document_service.document_repository,
        "list_documents",
        list_all,
    )

    assert document_service.list_documents(session) is documents
    list_all.assert_called_once_with(session)


@pytest.mark.parametrize(
    ("status", "expected_statuses", "error_code"),
    [
        (
            DocumentStatus.PROCESSING,
            (
                DocumentStatus.PENDING,
                DocumentStatus.UPLOADED,
                DocumentStatus.FAILED,
            ),
            None,
        ),
        (
            DocumentStatus.COMPLETED,
            (DocumentStatus.PROCESSING,),
            None,
        ),
        (
            DocumentStatus.FAILED,
            (DocumentStatus.PROCESSING,),
            DocumentProcessingErrorCode.PROCESSING_FAILED,
        ),
    ],
)
def test_update_document_status_delegates_allowed_transition(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
    status: DocumentStatus,
    expected_statuses: tuple[DocumentStatus, ...],
    error_code: DocumentProcessingErrorCode | None,
) -> None:
    document_id = UUID("00000000-0000-0000-0000-000000000001")
    document = MagicMock(spec=Document)
    update = MagicMock(return_value=(document, True))
    monkeypatch.setattr(
        document_service.document_repository,
        "update_document_status",
        update,
    )

    result = document_service.update_document_status(session, document_id, status.value)

    assert result is document
    update.assert_called_once_with(
        session,
        document_id,
        status,
        expected_statuses=expected_statuses,
        error_code=error_code,
    )


def test_update_document_status_rejects_unsupported_status(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    update = MagicMock()
    monkeypatch.setattr(
        document_service.document_repository,
        "update_document_status",
        update,
    )

    with pytest.raises(ValueError, match="Unsupported document status"):
        document_service.update_document_status(
            session,
            UUID("00000000-0000-0000-0000-000000000001"),
            "archived",
        )

    update.assert_not_called()


def test_update_document_status_preserves_missing_result(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    document_id = UUID("00000000-0000-0000-0000-000000000001")
    update = MagicMock(return_value=(None, False))
    monkeypatch.setattr(
        document_service.document_repository,
        "update_document_status",
        update,
    )

    assert (
        document_service.update_document_status(
            session,
            document_id,
            DocumentStatus.COMPLETED,
        )
        is None
    )

def test_update_document_status_rejects_invalid_transition(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    update = MagicMock()
    monkeypatch.setattr(
        document_service.document_repository,
        "update_document_status",
        update,
    )

    with pytest.raises(InvalidDocumentTransitionError):
        document_service.update_document_status(
            session,
            UUID("00000000-0000-0000-0000-000000000001"),
            DocumentStatus.PENDING,
        )

    update.assert_not_called()


def test_delete_document_delegates_and_preserves_missing_result(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    document_id = UUID("00000000-0000-0000-0000-000000000001")
    delete = MagicMock(return_value=False)
    monkeypatch.setattr(document_service.document_repository, "delete_document", delete)

    assert document_service.delete_document(session, document_id) is False
    delete.assert_called_once_with(session, document_id)


def test_service_does_not_commit_or_roll_back(
    monkeypatch: pytest.MonkeyPatch,
    session: MagicMock,
) -> None:
    create = MagicMock(return_value=MagicMock(spec=Document))
    monkeypatch.setattr(document_service.document_repository, "create_document", create)

    document_service.create_document(
        session,
        original_filename="lecture.pdf",
        file_type="pdf",
        storage_path="storage/lecture.pdf",
    )

    session.commit.assert_not_called()
    session.rollback.assert_not_called()
