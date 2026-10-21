import importlib.util
from io import StringIO
from pathlib import Path

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from app.db.base import Base
from app.models import Document


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_alembic_config_uses_backend_migration_directory() -> None:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    scripts = ScriptDirectory.from_config(config)

    assert Path(scripts.dir) == BACKEND_ROOT / "alembic"


def test_document_model_is_in_alembic_target_metadata() -> None:
    assert Document.__table__ is Base.metadata.tables["documents"]


def test_initial_revision_generates_only_expected_documents_ddl() -> None:
    revision_path = next((BACKEND_ROOT / "alembic" / "versions").glob("*.py"))
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
