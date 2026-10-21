from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus


def create_document(
    session: Session,
    *,
    original_filename: str,
    file_type: str,
    storage_path: str,
) -> Document:
    """Add and flush a document; the caller owns the transaction."""
    document = Document(
        original_filename=original_filename,
        file_type=file_type,
        storage_path=storage_path,
    )
    session.add(document)
    session.flush()
    return document


def get_document(session: Session, document_id: UUID) -> Document | None:
    """Return a document by ID, or None if it does not exist."""
    return session.get(Document, document_id)


def list_documents(session: Session) -> list[Document]:
    """Return documents newest-first, with UUID as a stable tie-breaker."""
    statement = select(Document).order_by(
        Document.created_at.desc(),
        Document.id.asc(),
    )
    return list(session.scalars(statement).all())


def update_document_status(
    session: Session,
    document_id: UUID,
    status: DocumentStatus | str,
) -> Document | None:
    """Update a supported status; the caller owns the transaction."""
    try:
        valid_status = DocumentStatus(status)
    except ValueError as error:
        raise ValueError(f"Unsupported document status: {status!r}") from error

    document = session.get(Document, document_id)
    if document is None:
        return None

    document.processing_status = valid_status.value
    session.flush()
    return document


def delete_document(session: Session, document_id: UUID) -> bool:
    """Delete a document and return whether it existed."""
    document = session.get(Document, document_id)
    if document is None:
        return False

    session.delete(document)
    session.flush()
    return True
