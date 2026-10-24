from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import ENV_FILE, PROJECT_ROOT, Settings


@pytest.fixture(autouse=True)
def clear_application_environment(monkeypatch) -> None:
    for variable in (
        "COGNIFY_APP_NAME",
        "COGNIFY_API_V1_PREFIX",
        "COGNIFY_UPLOAD_ROOT",
        "COGNIFY_MAX_UPLOAD_SIZE_BYTES",
        "DATABASE_URL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(variable, raising=False)


def test_defaults_without_environment_or_dotenv(monkeypatch) -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "Cognify AI"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.database_url is None
    assert settings.embedding_provider is None
    assert settings.embedding_model is None
    assert settings.embedding_dimensions is None
    assert settings.max_upload_size_bytes == 25 * 1024 * 1024
    assert settings.upload_root == PROJECT_ROOT / "storage" / "uploads"


def test_rejects_nonpositive_maximum_upload_size() -> None:
    with pytest.raises(ValidationError, match="Maximum upload size"):
        Settings(max_upload_size_bytes=0, _env_file=None)


def test_rejects_nonpositive_embedding_dimensions() -> None:
    with pytest.raises(ValidationError, match="Embedding dimensions"):
        Settings(embedding_dimensions=0, _env_file=None)


def test_resolves_relative_upload_root_from_repository_root() -> None:
    settings = Settings(upload_root=Path("storage/uploads"), _env_file=None)

    assert settings.upload_root == PROJECT_ROOT / "storage" / "uploads"


def test_environment_variables_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("COGNIFY_APP_NAME", "Test Cognify")
    monkeypatch.setenv("COGNIFY_API_V1_PREFIX", "/test/v1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://test:test@localhost/test_db"
    )

    settings = Settings(_env_file=None)

    assert settings.app_name == "Test Cognify"
    assert settings.api_v1_prefix == "/test/v1"
    assert settings.embedding_provider == "fake"
    assert settings.embedding_model == "test-model"
    assert settings.embedding_dimensions == 3
    assert settings.database_url == "postgresql+psycopg://test:test@localhost/test_db"


def test_loads_values_from_explicit_temporary_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    test_env_file = tmp_path / "test.env"
    test_env_file.write_text(
        "COGNIFY_APP_NAME=Temporary Cognify\n"
        "COGNIFY_API_V1_PREFIX=/temporary/v1\n"
        "DATABASE_URL=postgresql+psycopg://test:test@localhost/test_db\n"
        "UNRELATED_TEST_SETTING=ignored\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=test_env_file)

    assert settings.app_name == "Temporary Cognify"
    assert settings.api_v1_prefix == "/temporary/v1"
    assert settings.database_url == "postgresql+psycopg://test:test@localhost/test_db"


def test_dotenv_path_is_repository_root_relative() -> None:
    assert ENV_FILE == PROJECT_ROOT / ".env"
    assert PROJECT_ROOT == Path(__file__).resolve().parents[2]


def test_environment_names_are_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("cognify_app_name", "Case Insensitive Cognify")

    settings = Settings(_env_file=None)

    assert settings.app_name == "Case Insensitive Cognify"


@pytest.mark.parametrize("api_v1_prefix", ("api/v1", "/api/v1/"))
def test_rejects_invalid_api_v1_prefix(api_v1_prefix: str) -> None:
    with pytest.raises(ValidationError, match="API v1 prefix"):
        Settings(api_v1_prefix=api_v1_prefix, _env_file=None)
