import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, String, Text

from app.db.base import Base
from app.models import Document


def test_document_is_registered_with_shared_metadata() -> None:
    assert Document.__table__ is Base.metadata.tables["documents"]


def test_document_columns_have_expected_types_and_nullability() -> None:
    columns = Document.__table__.columns

    assert columns.id.type.python_type is UUID
    assert isinstance(columns.original_filename.type, String)
    assert columns.original_filename.type.length == 255
    assert isinstance(columns.file_type.type, String)
    assert columns.file_type.type.length == 10
    assert isinstance(columns.storage_path.type, Text)
    assert isinstance(columns.processing_status.type, String)
    assert columns.processing_status.type.length == 20
    assert isinstance(columns.created_at.type, DateTime)
    assert columns.created_at.type.timezone is True
    assert isinstance(columns.updated_at.type, DateTime)
    assert columns.updated_at.type.timezone is True
    assert all(column.nullable is False for column in columns)


def test_document_has_uuid_and_pending_status_defaults() -> None:
    columns = Document.__table__.columns

    assert columns.id.default.arg.__name__ == uuid4.__name__
    assert columns.processing_status.default.arg == "pending"
    assert str(columns.processing_status.server_default.arg) == "'pending'"
    assert columns.created_at.server_default is not None
    assert columns.updated_at.server_default is not None
    assert columns.updated_at.onupdate is not None


def test_document_constraints_limit_file_types_and_processing_statuses() -> None:
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in Document.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert "ck_documents_file_type" in constraints
    assert all(
        file_type in constraints["ck_documents_file_type"]
        for file_type in ("pdf", "txt", "mp3", "wav", "mp4")
    )
    assert "ck_documents_processing_status" in constraints
    assert all(
        status in constraints["ck_documents_processing_status"]
        for status in ("pending", "processing", "completed", "failed")
    )


def test_importing_document_model_does_not_create_engine() -> None:
    environment = {
        "PATH": os.environ["PATH"],
        "SYSTEMROOT": os.environ["SYSTEMROOT"],
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sqlalchemy\n"
            "def fail_if_engine_created(*args, **kwargs):\n"
            "    raise AssertionError('Engine creation attempted')\n"
            "sqlalchemy.create_engine = fail_if_engine_created\n"
            "import app.models.document\n",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
