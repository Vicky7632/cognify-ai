from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import (
    CheckConstraint,
    delete,
    event,
    Float,
    ForeignKey,
    func,
    Integer,
    select,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    Session,
    configure_mappers,
    sessionmaker,
)
from sqlalchemy.dialects import postgresql, sqlite

from app.db.base import Base
from app.models import Document, DocumentSegment


@pytest.fixture
def database_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(connection, _record) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_document(session: Session) -> Document:
    document = Document(
        original_filename="lesson.pdf",
        file_type="pdf",
        storage_path="random-storage-key.pdf",
    )
    session.add(document)
    session.flush()
    return document


def test_document_segment_is_registered_with_shared_metadata() -> None:
    assert DocumentSegment.__table__ is Base.metadata.tables["document_segments"]


def test_document_segment_columns_have_expected_types_and_nullability() -> None:
    columns = DocumentSegment.__table__.columns

    assert columns.id.type.python_type is UUID
    assert columns.id.primary_key is True
    assert columns.id.default.arg.__name__ == uuid4.__name__
    assert columns.document_id.type.python_type is UUID
    assert columns.document_id.nullable is False
    assert isinstance(columns.ordinal.type, Integer)
    assert columns.ordinal.nullable is False
    assert isinstance(columns.text_content.type, Text)
    assert columns.text_content.nullable is False
    assert isinstance(columns.page_number.type, Integer)
    assert columns.page_number.nullable is True
    assert isinstance(columns.start_seconds.type, Float)
    assert columns.start_seconds.nullable is True
    assert isinstance(columns.end_seconds.type, Float)
    assert columns.end_seconds.nullable is True


def test_document_segment_has_cascading_document_foreign_key() -> None:
    foreign_key = next(iter(DocumentSegment.__table__.foreign_keys))

    assert isinstance(foreign_key, ForeignKey)
    assert foreign_key.target_fullname == "documents.id"
    assert foreign_key.ondelete == "CASCADE"
    assert foreign_key.constraint.name == (
        "fk_document_segments_document_id_documents"
    )


def test_document_segment_constraints_cover_order_text_and_provenance() -> None:
    table = DocumentSegment.__table__
    unique_constraints = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    check_constraints = {
        constraint.name: " ".join(
            str(constraint.sqltext.compile(dialect=postgresql.dialect())).split()
        )
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert unique_constraints == {
        "uq_document_segments_document_id_ordinal": ("document_id", "ordinal")
    }
    assert check_constraints == {
        "ck_document_segments_ordinal_nonnegative": "ordinal >= 0",
        "ck_document_segments_text_nonempty": "text_content ~ '[^[:space:]]'",
        "ck_document_segments_page_number_positive": (
            "page_number IS NULL OR page_number > 0"
        ),
        "ck_document_segments_timestamps_valid": (
            "(start_seconds IS NULL AND end_seconds IS NULL) OR "
            "(start_seconds IS NOT NULL AND end_seconds IS NOT NULL AND "
            "start_seconds >= 0 AND end_seconds >= start_seconds)"
        ),
    }
    text_constraint = next(
        constraint
        for constraint in table.constraints
        if constraint.name == "ck_document_segments_text_nonempty"
    )
    assert str(
        text_constraint.sqltext.compile(dialect=postgresql.dialect())
    ) == "text_content ~ '[^[:space:]]'"
    assert str(text_constraint.sqltext.compile(dialect=sqlite.dialect())) == (
        "length(trim(replace(replace(replace(text_content, char(9), ''), "
        "char(10), ''), char(13), ''))) > 0"
    )


def test_document_segments_relationship_is_ordered_and_bidirectional() -> None:
    configure_mappers()
    relationship = Document.segments.property

    assert relationship.back_populates == "document"
    assert relationship.order_by == (DocumentSegment.ordinal,)
    assert DocumentSegment.document.property.back_populates == "segments"


def test_database_rejects_duplicate_document_ordinals(
    database_session: Session,
) -> None:
    document = create_document(database_session)
    database_session.add_all(
        [
            DocumentSegment(
                document_id=document.id,
                ordinal=0,
                text_content="First segment",
            ),
            DocumentSegment(
                document_id=document.id,
                ordinal=0,
                text_content="Duplicate ordinal",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        database_session.flush()


@pytest.mark.parametrize(
    ("ordinal", "text_content", "page_number", "start_seconds", "end_seconds"),
    [
        (-1, "Valid segment text", None, None, None),
        (0, "", None, None, None),
        (0, "   ", None, None, None),
        (0, "\t", None, None, None),
        (0, "\n", None, None, None),
        (0, "Valid segment text", 0, None, None),
        (0, "Valid segment text", -1, None, None),
        (0, "Valid segment text", None, 1.0, None),
        (0, "Valid segment text", None, None, 1.0),
        (0, "Valid segment text", None, -1.0, 2.0),
        (0, "Valid segment text", None, 1.0, -1.0),
        (0, "Valid segment text", None, 2.0, 1.0),
    ],
)
def test_database_rejects_invalid_segment_values(
    database_session: Session,
    ordinal: int,
    text_content: str,
    page_number: int | None,
    start_seconds: float | None,
    end_seconds: float | None,
) -> None:
    document = create_document(database_session)
    database_session.add(
        DocumentSegment(
            document_id=document.id,
            ordinal=ordinal,
            text_content=text_content,
            page_number=page_number,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
    )

    with pytest.raises(IntegrityError):
        database_session.flush()


def test_database_cascades_document_deletion_to_segments(
    database_session: Session,
) -> None:
    document = create_document(database_session)
    database_session.add(
        DocumentSegment(
            document_id=document.id,
            ordinal=0,
            text_content="Segment to delete",
        )
    )
    database_session.commit()

    assert (
        database_session.connection()
        .exec_driver_sql("PRAGMA foreign_keys")
        .scalar_one()
        == 1
    )
    assert (
        database_session.scalar(select(func.count()).select_from(DocumentSegment))
        == 1
    )

    database_session.execute(delete(Document).where(Document.id == document.id))
    database_session.commit()

    assert database_session.scalar(select(func.count()).select_from(DocumentSegment)) == 0
