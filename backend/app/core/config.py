from functools import lru_cache
from math import isfinite
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"
SUPPORTED_TRANSCRIPTION_COMPUTE_TYPES = frozenset(
    {
        "auto",
        "default",
        "int8",
        "int8_float32",
        "int8_float16",
        "int8_bfloat16",
        "int16",
        "float16",
        "float32",
        "bfloat16",
    }
)


class Settings(BaseSettings):
    app_name: str = "Cognify AI"
    api_v1_prefix: str = "/api/v1"
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    upload_root: Path = PROJECT_ROOT / "storage" / "uploads"
    max_upload_size_bytes: int = 25 * 1024 * 1024
    transcription_model_path: Path = Field(
        default=PROJECT_ROOT / "models" / "faster-whisper" / "base",
        validation_alias="TRANSCRIPTION_MODEL_PATH",
    )
    transcription_cpu_threads: int = Field(
        default=2,
        validation_alias="TRANSCRIPTION_CPU_THREADS",
    )
    transcription_compute_type: str = Field(
        default="int8",
        validation_alias="TRANSCRIPTION_COMPUTE_TYPE",
    )
    transcription_max_audio_seconds: float = Field(
        default=300,
        validation_alias="TRANSCRIPTION_MAX_AUDIO_SECONDS",
    )
    embedding_provider: str = Field(
        default="gemini",
        validation_alias="EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(
        default=1024,
        validation_alias="EMBEDDING_DIMENSIONS",
    )
    llm_provider: str | None = Field(
        default=None,
        validation_alias="LLM_PROVIDER",
    )
    llm_model: str | None = Field(
        default=None,
        validation_alias="LLM_MODEL",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        repr=False,
    )
    pinecone_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="PINECONE_API_KEY",
        repr=False,
    )
    pinecone_index: str | None = Field(
        default=None,
        validation_alias="PINECONE_INDEX",
    )
    pinecone_host: str | None = Field(
        default=None,
        validation_alias="PINECONE_HOST",
    )
    pinecone_namespace: str = Field(
        default="cognify",
        validation_alias="PINECONE_NAMESPACE",
    )
    pinecone_upsert_batch_size: int = Field(
        default=100,
        validation_alias="PINECONE_UPSERT_BATCH_SIZE",
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

    @field_validator("transcription_model_path")
    @classmethod
    def resolve_transcription_model_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            return PROJECT_ROOT / value
        return value

    @field_validator("transcription_cpu_threads")
    @classmethod
    def validate_transcription_cpu_threads(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Transcription CPU threads must be greater than zero.")
        return value

    @field_validator("transcription_compute_type", mode="before")
    @classmethod
    def validate_transcription_compute_type(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Transcription compute type is not supported.")
        compute_type = value.strip().lower()
        if compute_type not in SUPPORTED_TRANSCRIPTION_COMPUTE_TYPES:
            raise ValueError("Transcription compute type is not supported.")
        return compute_type

    @field_validator("transcription_max_audio_seconds")
    @classmethod
    def validate_transcription_max_audio_seconds(cls, value: float) -> float:
        if not isfinite(value) or value <= 0:
            raise ValueError(
                "Maximum transcription audio duration must be greater than zero."
            )
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")
        return value

    @field_validator("pinecone_upsert_batch_size")
    @classmethod
    def validate_pinecone_upsert_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Pinecone upsert batch size must be greater than zero.")
        return value

    @field_validator("pinecone_namespace")
    @classmethod
    def validate_pinecone_namespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pinecone namespace must not be empty.")
        return value

    @field_validator("pinecone_host", mode="before")
    @classmethod
    def normalize_optional_pinecone_host(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
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
