from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    app_name: str = "Cognify AI"
    api_v1_prefix: str = "/api/v1"
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    upload_root: Path = PROJECT_ROOT / "storage" / "uploads"
    max_upload_size_bytes: int = 25 * 1024 * 1024
    embedding_provider: str = Field(
        default="gemini",
        validation_alias="EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(
        default=3072,
        validation_alias="EMBEDDING_DIMENSIONS",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        repr=False,
    )

    @field_validator("upload_root")
    @classmethod
    def resolve_upload_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            return PROJECT_ROOT / value
        return value

    @field_validator("api_v1_prefix")
    @classmethod
    def validate_api_v1_prefix(cls, value: str) -> str:
        if not value.startswith("/") or value.endswith("/"):
            raise ValueError(
                "API v1 prefix must start with '/' and must not end with '/'."
            )
        return value

    @field_validator("max_upload_size_bytes")
    @classmethod
    def validate_max_upload_size_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Maximum upload size must be greater than zero.")
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")
        return value

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="COGNIFY_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
