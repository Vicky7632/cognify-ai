from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document import (
    Document,
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.models.document_segment import DocumentSegment
from app.repositories import document_repository, document_segment_repository


class InvalidDocumentTransitionError(ValueError):
    pass


ALLOWED_PREVIOUS_STATUSES = {
    DocumentStatus.PROCESSING: (
        DocumentStatus.PENDING,
        DocumentStatus.UPLOADED,
        DocumentStatus.FAILED,
    ),
    DocumentStatus.COMPLETED: (DocumentStatus.PROCESSING,),
    DocumentStatus.FAILED: (DocumentStatus.PROCESSING,),
}


def create_document(
    session: Session,
    *,
    original_filename: str,
    file_type: str,
    storage_path: str,
    processing_status: DocumentStatus = DocumentStatus.PENDING,
) -> Document:
    """Create a document record; transaction ownership stays with the caller."""
    try:
        valid_file_type = DocumentFileType(file_type)
    except ValueError as error:
        raise ValueError(f"Unsupported document file type: {file_type!r}") from error

    return document_repository.create_document(
        session,
        original_filename=original_filename,
        file_type=valid_file_type.value,
        storage_path=storage_path,
        processing_status=processing_status,
    )


def get_document(session: Session, document_id: UUID) -> Document | None:
    """Retrieve a document by ID, or None if it does not exist."""
    return document_repository.get_document(session, document_id)


def list_documents(session: Session) -> list[Document]:
    """List documents in the repository's deterministic order."""
    return document_repository.list_documents(session)


def list_document_segments(
    session: Session,
    document_id: UUID,
    *,
    offset: int,
    limit: int,
) -> tuple[list[DocumentSegment], int]:
    """Return one ordered segment page and the document's total segment count."""
    total = document_segment_repository.count_document_segments(session, document_id)
    segments = document_segment_repository.get_document_segments(
        session,
        document_id,
        offset=offset,
        limit=limit,
    )
    return segments, total


def update_document_status(
    session: Session,
    document_id: UUID,
    status: DocumentStatus | str,
) -> Document | None:
    """Apply one allowed status transition without committing the caller's transaction."""
    try:
        valid_status = DocumentStatus(status)
    except ValueError as error:
        raise ValueError(f"Unsupported document status: {status!r}") from error

    expected_statuses = ALLOWED_PREVIOUS_STATUSES.get(valid_status)
    if expected_statuses is None:
        raise InvalidDocumentTransitionError(
            f"Transition to {valid_status.value!r} is not allowed."
        )

    error_code = (
        DocumentProcessingErrorCode.PROCESSING_FAILED
        if valid_status is DocumentStatus.FAILED
        else None
    )
    document, transitioned = document_repository.update_document_status(
        session,
        document_id,
        valid_status,
        expected_statuses=expected_statuses,
        error_code=error_code,
    )
    if not transitioned and document is not None:
        raise InvalidDocumentTransitionError(
            f"Transition from {document.processing_status!r} "
            f"to {valid_status.value!r} is not allowed."
        )
    return document


def delete_document(session: Session, document_id: UUID) -> bool:
    """Delete a document by ID and indicate whether it existed."""
    return document_repository.delete_document(session, document_id)
