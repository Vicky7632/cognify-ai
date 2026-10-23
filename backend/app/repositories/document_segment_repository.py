from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models.document_segment import DocumentSegment


def replace_document_segments(
    session: Session,
    document_id: UUID,
    segments: tuple[DocumentSegment, ...],
) -> None:
    """Replace a document's segments; the caller owns the transaction."""
    session.execute(
        delete(DocumentSegment).where(DocumentSegment.document_id == document_id)
    )
    session.add_all(segments)
    session.flush()
