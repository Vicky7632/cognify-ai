from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentProcessingErrorCode, DocumentStatus


def create_document(
    session: Session,
    *,
    original_filename: str,
    file_type: str,
    storage_path: str,
    processing_status: DocumentStatus = DocumentStatus.PENDING,
) -> Document:
    """Add and flush a document; the caller owns the transaction."""
    document = Document(
        original_filename=original_filename,
        file_type=file_type,
        storage_path=storage_path,
        processing_status=processing_status.value,
    )
    session.add(document)
    session.flush()
    return document


def get_document(session: Session, document_id: UUID) -> Document | None:
    """Return a document by ID, or None if it does not exist."""
    statement = (
        select(Document)
        .where(Document.id == document_id)
        .execution_options(populate_existing=True)
    )
    return session.scalar(statement)


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
    *,
    expected_statuses: tuple[DocumentStatus, ...],
    error_code: DocumentProcessingErrorCode | None,
) -> tuple[Document | None, bool]:
    """Compare-and-set a lifecycle transition; the caller owns the transaction."""
    valid_status = DocumentStatus(status)
    statement = (
        update(Document)
        .where(
            Document.id == document_id,
            Document.processing_status.in_(
                tuple(previous.value for previous in expected_statuses)
            ),
        )
        .values(
            processing_status=valid_status.value,
            processing_error_code=error_code.value if error_code else None,
        )
        .execution_options(synchronize_session="fetch")
    )
    result = session.execute(statement)
    document = get_document(session, document_id)
    return document, result.rowcount == 1


def delete_document(session: Session, document_id: UUID) -> bool:
    """Delete a document and return whether it existed."""
    document = session.get(Document, document_id)
    if document is None:
        return False

    session.delete(document)
    session.flush()
    return True
