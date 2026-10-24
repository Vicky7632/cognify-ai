from collections.abc import Sequence
from numbers import Real
from typing import Protocol

from google import genai
from google.genai import types

from app.core.config import Settings
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingInputError,
    EmbeddingOutputError,
    EmbeddingProviderError,
)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_MAX_BATCH_SIZE = 100


class _EmbeddingModelClient(Protocol):
    def embed_content(
        self,
        *,
        model: str,
        contents: Sequence[str],
        config: types.EmbedContentConfig | None = None,
    ) -> object:
        ...


class _GeminiClient(Protocol):
    models: _EmbeddingModelClient

    def close(self) -> None:
        ...


class GeminiEmbeddingProvider:
    """Google Gemini Developer API adapter for text embeddings."""

    max_batch_size = GEMINI_MAX_BATCH_SIZE

    def __init__(
        self,
        *,
        api_key: str,
        output_dimensions: int,
        client: _GeminiClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise EmbeddingConfigurationError("Gemini API key is not configured.")
        if output_dimensions <= 0:
            raise EmbeddingConfigurationError(
                "Embedding dimensions must be greater than zero."
            )
        try:
            self._client = (
                client if client is not None else genai.Client(api_key=api_key)
            )
        except Exception:
            raise EmbeddingConfigurationError(
                "Unable to initialize the Gemini embedding provider."
            ) from None
        self._output_dimensions = output_dimensions

    def embed_batch(
        self,
        texts: tuple[str, ...],
        *,
        model: str,
    ) -> Sequence[Sequence[Real]]:
        if not texts or len(texts) > self.max_batch_size:
            raise EmbeddingInputError(
                f"Gemini embedding batches must contain 1 to "
                f"{self.max_batch_size} texts."
            )
        config = types.EmbedContentConfig(
            output_dimensionality=self._output_dimensions,
        )
        try:
            response = self._client.models.embed_content(
                model=model,
                contents=list(texts),
                config=config,
            )
        except Exception:
            raise EmbeddingProviderError(
                "The Gemini embedding request failed."
            ) from None

        embeddings = getattr(response, "embeddings", None)
        if not isinstance(embeddings, Sequence):
            raise EmbeddingOutputError(
                "The Gemini embedding provider returned a malformed response."
            )

        vectors: list[Sequence[Real]] = []
        for embedding in embeddings:
            values = getattr(embedding, "values", None)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise EmbeddingOutputError(
                    "The Gemini embedding provider returned a malformed response."
                )
            vectors.append(values)
        return tuple(vectors)

    def close(self) -> None:
        """Release the SDK client's underlying HTTP resources."""
        self._client.close()


def create_gemini_embedding_provider(settings: Settings) -> GeminiEmbeddingProvider:
    """Create a Gemini adapter from application settings without exposing the key."""
    if settings.gemini_api_key is None:
        raise EmbeddingConfigurationError("Gemini API key is not configured.")
    if settings.embedding_model != GEMINI_EMBEDDING_MODEL:
        raise EmbeddingConfigurationError(
            "The Gemini provider requires gemini-embedding-001."
        )
    return GeminiEmbeddingProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        output_dimensions=settings.embedding_dimensions,
    )
