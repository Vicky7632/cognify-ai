import importlib.util
from io import StringIO
from pathlib import Path

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from app.db.base import Base
from app.models import Document, DocumentProcessingJob


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_alembic_config_uses_backend_migration_directory() -> None:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    scripts = ScriptDirectory.from_config(config)

    assert Path(scripts.dir) == BACKEND_ROOT / "alembic"


def test_document_model_is_in_alembic_target_metadata() -> None:
    assert Document.__table__ is Base.metadata.tables["documents"]
    assert DocumentProcessingJob.__table__ is Base.metadata.tables[
        "document_processing_jobs"
    ]


def test_processing_job_revision_creates_durable_queue_table() -> None:
    revision_path = (
        BACKEND_ROOT
        / "alembic"
        / "versions"
        / "b31d20261026_add_document_processing_jobs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "document_processing_jobs_revision",
        revision_path,
    )
    assert spec is not None
    assert spec.loader is not None
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)

    output = StringIO()
    migration_context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(migration_context):
        revision.upgrade()

    sql = output.getvalue()
    assert "CREATE TABLE document_processing_jobs" in sql
    assert "document_id UUID NOT NULL" in sql
    assert "UNIQUE (document_id)" in sql
    assert "status IN ('queued', 'processing', 'completed', 'failed')" in sql
    assert "available_at TIMESTAMP WITH TIME ZONE" in sql
    assert "FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE" in sql


def test_initial_revision_generates_only_expected_documents_ddl() -> None:
    revision_path = (
        BACKEND_ROOT
        / "alembic"
        / "versions"
        / "c5ede9494309_create_documents_table.py"
    )
    spec = importlib.util.spec_from_file_location("initial_revision", revision_path)
    assert spec is not None
    assert spec.loader is not None
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)

    output = StringIO()
    migration_context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(migration_context):
        revision.upgrade()

    sql = output.getvalue()
    assert sql.count("CREATE TABLE") == 1
    assert "CREATE TABLE documents" in sql
    assert all(
        column in sql
        for column in (
            "id UUID NOT NULL",
            "original_filename VARCHAR(255) NOT NULL",
            "file_type VARCHAR(10) NOT NULL",
            "storage_path TEXT NOT NULL",
            "processing_status VARCHAR(20) DEFAULT 'pending' NOT NULL",
            "created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL",
            "updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL",
        )
    )
    assert "ck_documents_file_type" in sql
    assert "ck_documents_processing_status" in sql


def test_lifecycle_revision_adds_safe_error_code_and_uploaded_status() -> None:
    revision_path = (
        BACKEND_ROOT
        / "alembic"
        / "versions"
        / "d16a20261022_document_processing_lifecycle.py"
    )
    spec = importlib.util.spec_from_file_location("lifecycle_revision", revision_path)
    assert spec is not None
    assert spec.loader is not None
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)

    output = StringIO()
    migration_context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(migration_context):
        revision.upgrade()

    sql = output.getvalue()
    assert "ADD COLUMN processing_error_code VARCHAR(50)" in sql
    normalized_sql = " ".join(sql.split())
    assert (
        "CHECK (processing_status IN "
        "('pending', 'uploaded', 'processing', 'completed', 'failed'))"
        in normalized_sql
    )
    assert (
        "CHECK (processing_error_code IS NULL OR "
        "(processing_status = 'failed' AND "
        "processing_error_code = 'processing_failed'))"
        in normalized_sql
    )


def test_lifecycle_downgrade_converts_uploaded_before_restoring_constraint() -> None:
    revision_path = (
        BACKEND_ROOT
        / "alembic"
        / "versions"
        / "d16a20261022_document_processing_lifecycle.py"
    )
    spec = importlib.util.spec_from_file_location("lifecycle_downgrade", revision_path)
    assert spec is not None
    assert spec.loader is not None
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)

    output = StringIO()
    migration_context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(migration_context):
        revision.downgrade()

    normalized_sql = " ".join(output.getvalue().split())
    data_conversion = (
        "UPDATE documents SET processing_status = 'pending' "
        "WHERE processing_status = 'uploaded'"
    )
    old_constraint = (
        "CHECK (processing_status IN "
        "('pending', 'processing', 'completed', 'failed'))"
    )
    assert data_conversion in normalized_sql
    assert normalized_sql.index(data_conversion) < normalized_sql.index(old_constraint)
