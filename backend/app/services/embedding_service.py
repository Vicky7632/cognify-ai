from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Protocol

from app.core.config import Settings

EmbeddingVector = tuple[float, ...]
EmbeddingBatch = tuple[EmbeddingVector, ...]


class EmbeddingConfigurationError(RuntimeError):
    """Raised when embedding provider configuration is incomplete or unavailable."""


class EmbeddingInputError(ValueError):
    """Raised when input text does not meet the embedding service contract."""


class EmbeddingProviderError(RuntimeError):
    """Raised when the configured embedding provider fails."""


class EmbeddingOutputError(RuntimeError):
    """Raised when a provider response violates the embedding output contract."""


class EmbeddingProvider(Protocol):
    """Provider adapter contract; each call must preserve input order."""

    max_batch_size: int

    def embed_batch(
        self,
        texts: tuple[str, ...],
        *,
        model: str,
    ) -> Sequence[Sequence[Real]]:
        """Return one vector per input text, in the same order."""
        ...


class EmbeddingService:
    """Validate and batch texts through an injected embedding provider."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        model: str,
        expected_dimensions: int | None = None,
    ) -> None:
        if not model.strip():
            raise EmbeddingConfigurationError("An embedding model must be configured.")
        max_batch_size = provider.max_batch_size
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise EmbeddingConfigurationError(
                "The embedding provider has an invalid batch limit."
            )
        if max_batch_size < 1:
            raise EmbeddingConfigurationError(
                "The embedding provider has an invalid batch limit."
            )
        if expected_dimensions is not None and (
            isinstance(expected_dimensions, bool)
            or not isinstance(expected_dimensions, int)
            or expected_dimensions < 1
        ):
            raise EmbeddingConfigurationError(
                "Configured embedding dimensions must be positive."
            )

        self._provider = provider
        self._model = model
        self._max_batch_size = max_batch_size
        self._expected_dimensions = expected_dimensions

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed one non-empty text and return its finite numeric vector."""
        return self.embed_texts((text,))[0]

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed non-empty texts in provider-sized batches, preserving order."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingInputError("Embedding input must be a sequence of texts.")
        if not texts:
            raise EmbeddingInputError("At least one text is required.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingInputError("Embedding texts must be non-empty strings.")

        vectors: list[EmbeddingVector] = []
        dimensions = self._expected_dimensions
        for start in range(0, len(texts), self._max_batch_size):
            batch = tuple(texts[start : start + self._max_batch_size])
            try:
                raw_vectors = tuple(
                    self._provider.embed_batch(batch, model=self._model)
                )
            except Exception:
                raise EmbeddingProviderError(
                    "The embedding provider request failed."
                ) from None

            if len(raw_vectors) != len(batch):
                raise EmbeddingOutputError(
                    "The embedding provider returned an unexpected vector count."
                )

            for raw_vector in raw_vectors:
                vector = self._validate_vector(raw_vector)
                if dimensions is None:
                    dimensions = len(vector)
                if len(vector) != dimensions:
                    raise EmbeddingOutputError(
                        "The embedding provider returned inconsistent vector dimensions."
                    )
                vectors.append(vector)

        return tuple(vectors)

    @staticmethod
    def _validate_vector(raw_vector: Sequence[Real]) -> EmbeddingVector:
        if isinstance(raw_vector, (str, bytes)) or not isinstance(
            raw_vector, Sequence
        ):
            raise EmbeddingOutputError(
                "The embedding provider returned an invalid vector."
            )
        if not raw_vector:
            raise EmbeddingOutputError(
                "The embedding provider returned an empty vector."
            )

        vector: list[float] = []
        for value in raw_vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmbeddingOutputError(
                    "The embedding provider returned a non-numeric vector value."
                )
            try:
                numeric_value = float(value)
            except (OverflowError, ValueError):
                raise EmbeddingOutputError(
                    "The embedding provider returned an invalid vector value."
                ) from None
            if not isfinite(numeric_value):
                raise EmbeddingOutputError(
                    "The embedding provider returned a non-finite vector value."
                )
            vector.append(numeric_value)
        return tuple(vector)


def create_embedding_service(
    settings: Settings,
    providers: Mapping[str, EmbeddingProvider],
) -> EmbeddingService:
    """Build a service from explicit settings and registered provider adapters."""
    provider_name = settings.embedding_provider
    model = settings.embedding_model
    if not provider_name or not provider_name.strip() or not model or not model.strip():
        raise EmbeddingConfigurationError(
            "Embedding provider and model must be configured."
        )

    provider = providers.get(provider_name)
    if provider is None:
        raise EmbeddingConfigurationError(
            "The configured embedding provider is unavailable."
        )

    return EmbeddingService(
        provider,
        model=model,
        expected_dimensions=settings.embedding_dimensions,
    )
