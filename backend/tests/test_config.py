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
        "TRANSCRIPTION_MODEL_PATH",
        "TRANSCRIPTION_CPU_THREADS",
        "TRANSCRIPTION_COMPUTE_TYPE",
        "TRANSCRIPTION_MAX_AUDIO_SECONDS",
        "DATABASE_URL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "GEMINI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX",
        "PINECONE_HOST",
        "PINECONE_NAMESPACE",
        "PINECONE_UPSERT_BATCH_SIZE",
    ):
        monkeypatch.delenv(variable, raising=False)


def test_defaults_without_environment_or_dotenv(monkeypatch) -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "Cognify AI"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.database_url is None
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.embedding_dimensions == 1024
    assert settings.llm_provider is None
    assert settings.llm_model is None
    assert settings.gemini_api_key is None
    assert settings.pinecone_api_key is None
    assert settings.pinecone_index is None
    assert settings.pinecone_host is None
    assert settings.pinecone_namespace == "cognify"
    assert settings.pinecone_upsert_batch_size == 100
    assert settings.max_upload_size_bytes == 25 * 1024 * 1024
    assert settings.upload_root == PROJECT_ROOT / "storage" / "uploads"
    assert settings.transcription_model_path == (
        PROJECT_ROOT / "models" / "faster-whisper" / "base"
    )
    assert settings.transcription_cpu_threads == 2
    assert settings.transcription_compute_type == "int8"
    assert settings.transcription_max_audio_seconds == 300


def test_rejects_nonpositive_maximum_upload_size() -> None:
    with pytest.raises(ValidationError, match="Maximum upload size"):
        Settings(max_upload_size_bytes=0, _env_file=None)


def test_rejects_nonpositive_transcription_cpu_threads() -> None:
    with pytest.raises(ValidationError, match="Transcription CPU threads"):
        Settings(transcription_cpu_threads=0, _env_file=None)


@pytest.mark.parametrize("value", ["", "cuda", "int7"])
def test_rejects_unsupported_transcription_compute_type(value: str) -> None:
    with pytest.raises(ValidationError, match="Transcription compute type"):
        Settings(transcription_compute_type=value, _env_file=None)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_transcription_audio_duration(value: float) -> None:
    with pytest.raises(ValidationError, match="Maximum transcription audio duration"):
        Settings(transcription_max_audio_seconds=value, _env_file=None)


def test_rejects_nonpositive_embedding_dimensions() -> None:
    with pytest.raises(ValidationError, match="Embedding dimensions"):
        Settings(embedding_dimensions=0, _env_file=None)


def test_rejects_nonpositive_pinecone_batch_size() -> None:
    with pytest.raises(ValidationError, match="Pinecone upsert batch size"):
        Settings(pinecone_upsert_batch_size=0, _env_file=None)


def test_rejects_empty_pinecone_namespace() -> None:
    with pytest.raises(ValidationError, match="Pinecone namespace"):
        Settings(pinecone_namespace=" ", _env_file=None)


def test_empty_optional_pinecone_host_is_treated_as_unconfigured() -> None:
    settings = Settings(pinecone_host=" ", _env_file=None)

    assert settings.pinecone_host is None


def test_resolves_relative_upload_root_from_repository_root() -> None:
    settings = Settings(upload_root=Path("storage/uploads"), _env_file=None)

    assert settings.upload_root == PROJECT_ROOT / "storage" / "uploads"


def test_resolves_relative_transcription_model_path_from_repository_root() -> None:
    settings = Settings(
        transcription_model_path=Path("models/faster-whisper/base"),
        _env_file=None,
    )

    assert settings.transcription_model_path == (
        PROJECT_ROOT / "models" / "faster-whisper" / "base"
    )


def test_environment_variables_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("COGNIFY_APP_NAME", "Test Cognify")
    monkeypatch.setenv("COGNIFY_API_V1_PREFIX", "/test/v1")
    monkeypatch.setenv(
        "TRANSCRIPTION_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "custom"),
    )
    monkeypatch.setenv("TRANSCRIPTION_CPU_THREADS", "3")
    monkeypatch.setenv("TRANSCRIPTION_COMPUTE_TYPE", "int8_float32")
    monkeypatch.setenv("TRANSCRIPTION_MAX_AUDIO_SECONDS", "45")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-test-key")
    monkeypatch.setenv("PINECONE_INDEX", "cognify-test-index")
    monkeypatch.setenv(
        "PINECONE_HOST",
        "https://cognify-test-index.svc.pinecone.io",
    )
    monkeypatch.setenv("PINECONE_NAMESPACE", "test-namespace")
    monkeypatch.setenv("PINECONE_UPSERT_BATCH_SIZE", "25")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://test:test@localhost/test_db"
    )

    settings = Settings(_env_file=None)

    assert settings.app_name == "Test Cognify"
    assert settings.api_v1_prefix == "/test/v1"
    assert settings.transcription_model_path == PROJECT_ROOT / "models" / "custom"
    assert settings.transcription_cpu_threads == 3
    assert settings.transcription_compute_type == "int8_float32"
    assert settings.transcription_max_audio_seconds == 45
    assert settings.embedding_provider == "fake"
    assert settings.embedding_model == "test-model"
    assert settings.embedding_dimensions == 3
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "configured-model"
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "test-key"
    assert settings.pinecone_api_key is not None
    assert settings.pinecone_api_key.get_secret_value() == "pinecone-test-key"
    assert settings.pinecone_index == "cognify-test-index"
    assert settings.pinecone_host == "https://cognify-test-index.svc.pinecone.io"
    assert settings.pinecone_namespace == "test-namespace"
    assert settings.pinecone_upsert_batch_size == 25
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
