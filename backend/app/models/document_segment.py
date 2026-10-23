from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.expression import ColumnElement

from app.db.base import Base


class _ContainsNonWhitespace(ColumnElement[bool]):
    type = Boolean()
    inherit_cache = True


@compiles(_ContainsNonWhitespace, "postgresql")
def _compile_postgresql_non_whitespace(
    _element: _ContainsNonWhitespace,
    _compiler,
    **_kwargs: object,
) -> str:
    return "text_content ~ '[^[:space:]]'"


@compiles(_ContainsNonWhitespace, "sqlite")
def _compile_sqlite_non_whitespace(
    _element: _ContainsNonWhitespace,
    _compiler,
    **_kwargs: object,
) -> str:
    return (
        "length(trim(replace(replace(replace(text_content, char(9), ''), "
        "char(10), ''), char(13), ''))) > 0"
    )


class DocumentSegment(Base):
    __tablename__ = "document_segments"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "ordinal",
            name="uq_document_segments_document_id_ordinal",
        ),
        CheckConstraint(
            "ordinal >= 0",
            name="ck_document_segments_ordinal_nonnegative",
        ),
        CheckConstraint(
            _ContainsNonWhitespace(),
            name="ck_document_segments_text_nonempty",
        ),
        CheckConstraint(
            "page_number IS NULL OR page_number > 0",
            name="ck_document_segments_page_number_positive",
        ),
        CheckConstraint(
            "(start_seconds IS NULL AND end_seconds IS NULL) OR "
            "(start_seconds IS NOT NULL AND end_seconds IS NOT NULL AND "
            "start_seconds >= 0 AND end_seconds >= start_seconds)",
            name="ck_document_segments_timestamps_valid",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
            name="fk_document_segments_document_id_documents",
        ),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="segments",
    )
