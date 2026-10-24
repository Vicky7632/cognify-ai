from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.document_segment import DocumentSegment


def get_document_segments(
    session: Session,
    document_id: UUID,
    *,
    offset: int,
    limit: int,
) -> list[DocumentSegment]:
    """Return a page of a document's segments in ordinal order."""
    statement = (
        select(DocumentSegment)
        .where(DocumentSegment.document_id == document_id)
        .order_by(DocumentSegment.ordinal.asc())
        .offset(offset)
        .limit(limit)
    )
    return list(session.scalars(statement).all())


def count_document_segments(session: Session, document_id: UUID) -> int:
    """Return the total number of segments belonging to a document."""
    statement = (
        select(func.count())
        .select_from(DocumentSegment)
        .where(DocumentSegment.document_id == document_id)
    )
    return session.execute(statement).scalar_one()


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
