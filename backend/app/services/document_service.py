from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document import Document, DocumentFileType, DocumentStatus
from app.repositories import document_repository


def create_document(
    session: Session,
    *,
    original_filename: str,
    file_type: str,
    storage_path: str,
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
    )


def get_document(session: Session, document_id: UUID) -> Document | None:
    """Retrieve a document by ID, or None if it does not exist."""
    return document_repository.get_document(session, document_id)


def list_documents(session: Session) -> list[Document]:
    """List documents in the repository's deterministic order."""
    return document_repository.list_documents(session)


def update_document_status(
    session: Session,
    document_id: UUID,
    status: DocumentStatus | str,
) -> Document | None:
    """Update a supported status without committing the caller's transaction."""
    try:
        valid_status = DocumentStatus(status)
    except ValueError as error:
        raise ValueError(f"Unsupported document status: {status!r}") from error

    return document_repository.update_document_status(
        session,
        document_id,
        valid_status,
    )


def delete_document(session: Session, document_id: UUID) -> bool:
    """Delete a document by ID and indicate whether it existed."""
    return document_repository.delete_document(session, document_id)
