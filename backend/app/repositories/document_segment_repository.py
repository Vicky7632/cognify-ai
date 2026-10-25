from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, joinedload

from app.models.document_segment import DocumentSegment


def list_document_segments(
    session: Session,
    document_id: UUID,
) -> list[DocumentSegment]:
    """Return all segments for a document in ascending ordinal order."""
    statement = (
        select(DocumentSegment)
        .where(DocumentSegment.document_id == document_id)
        .order_by(DocumentSegment.ordinal.asc())
    )
    return list(session.scalars(statement).all())


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


def get_document_segments_by_ids(
    session: Session,
    segment_ids: Sequence[UUID],
) -> list[DocumentSegment]:
    """Load existing segments and their document metadata by segment ID."""
    if not segment_ids:
        return []
    statement = (
        select(DocumentSegment)
        .options(joinedload(DocumentSegment.document))
        .where(DocumentSegment.id.in_(set(segment_ids)))
    )
    return list(session.scalars(statement).all())


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
