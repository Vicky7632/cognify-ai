from pathlib import Path

from app.core.config import ENV_FILE, PROJECT_ROOT, Settings


def test_defaults_without_environment_or_dotenv(monkeypatch) -> None:
    monkeypatch.delenv("COGNIFY_APP_NAME", raising=False)
    monkeypatch.delenv("COGNIFY_API_V1_PREFIX", raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_name == "Cognify AI"
    assert settings.api_v1_prefix == "/api/v1"


def test_environment_variables_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("COGNIFY_APP_NAME", "Test Cognify")
    monkeypatch.setenv("COGNIFY_API_V1_PREFIX", "/test/v1")

    settings = Settings(_env_file=None)

    assert settings.app_name == "Test Cognify"
    assert settings.api_v1_prefix == "/test/v1"


def test_loads_values_from_explicit_temporary_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("COGNIFY_APP_NAME", raising=False)
    monkeypatch.delenv("COGNIFY_API_V1_PREFIX", raising=False)
    test_env_file = tmp_path / "test.env"
    test_env_file.write_text(
        "COGNIFY_APP_NAME=Temporary Cognify\n"
        "COGNIFY_API_V1_PREFIX=/temporary/v1\n"
        "UNRELATED_TEST_SETTING=ignored\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=test_env_file)

    assert settings.app_name == "Temporary Cognify"
    assert settings.api_v1_prefix == "/temporary/v1"


def test_dotenv_path_is_repository_root_relative() -> None:
    assert ENV_FILE == PROJECT_ROOT / ".env"
    assert PROJECT_ROOT == Path(__file__).resolve().parents[2]


def test_environment_names_are_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("cognify_app_name", "Case Insensitive Cognify")

    settings = Settings(_env_file=None)

    assert settings.app_name == "Case Insensitive Cognify"
