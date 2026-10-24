from collections.abc import Sequence
from numbers import Real
from typing import cast

import pytest

from app.core.config import Settings
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingInputError,
    EmbeddingOutputError,
    EmbeddingProviderError,
    EmbeddingService,
    create_embedding_service,
)


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        max_batch_size: int = 2,
        vectors: dict[str, Sequence[Real]] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.vectors = vectors or {}
        self.failure = failure
        self.calls: list[tuple[tuple[str, ...], str]] = []
        self.closed = False

    def embed_batch(
        self,
        texts: tuple[str, ...],
        *,
        model: str,
    ) -> Sequence[Sequence[Real]]:
        self.calls.append((texts, model))
        if self.failure is not None:
            raise self.failure
        return [self.vectors[text] for text in texts]

    def close(self) -> None:
        self.closed = True


def test_embeds_single_text() -> None:
    provider = FakeEmbeddingProvider(vectors={"alpha": (0.1, 0.2)})
    service = EmbeddingService(provider, model="test-model", expected_dimensions=2)

    assert service.embed_text("alpha") == (0.1, 0.2)
    assert provider.calls == [(("alpha",), "test-model")]


def test_embeds_multiple_texts_in_input_order() -> None:
    provider = FakeEmbeddingProvider(
        vectors={
            "alpha": (1, 0),
            "beta": (0, 1),
        }
    )
    service = EmbeddingService(provider, model="test-model")

    assert service.embed_texts(("alpha", "beta")) == ((1.0, 0.0), (0.0, 1.0))


def test_batches_at_provider_limit_without_dropping_or_reordering() -> None:
    texts = ("one", "two", "three", "four", "five")
    provider = FakeEmbeddingProvider(
        vectors={text: (index,) for index, text in enumerate(texts)}
    )
    service = EmbeddingService(provider, model="model")

    vectors = service.embed_texts(texts)

    assert [batch for batch, _ in provider.calls] == [
        ("one", "two"),
        ("three", "four"),
        ("five",),
    ]
    assert vectors == ((0.0,), (1.0,), (2.0,), (3.0,), (4.0,))


@pytest.mark.parametrize("texts", [(), ("",), ("   ",), ("valid", None)])
def test_rejects_empty_or_invalid_text(texts: tuple[object, ...]) -> None:
    service = EmbeddingService(
        FakeEmbeddingProvider(vectors={"valid": (0.1,)}),
        model="model",
    )

    with pytest.raises(EmbeddingInputError):
        service.embed_texts(cast(Sequence[str], texts))


def test_rejects_non_sequence_text_input() -> None:
    service = EmbeddingService(FakeEmbeddingProvider(), model="model")

    with pytest.raises(EmbeddingInputError):
        service.embed_texts("not a batch")


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [(1.0,)],
    ],
)
def test_rejects_output_count_mismatch(vectors: list[tuple[float, ...]]) -> None:
    class MismatchedProvider(FakeEmbeddingProvider):
        def embed_batch(
            self,
            texts: tuple[str, ...],
            *,
            model: str,
        ) -> Sequence[Sequence[Real]]:
            return vectors

    service = EmbeddingService(MismatchedProvider(), model="model")

    with pytest.raises(EmbeddingOutputError, match="vector count"):
        service.embed_texts(("one", "two"))


def test_rejects_inconsistent_dimensions_within_batch() -> None:
    provider = FakeEmbeddingProvider(
        vectors={"one": (1.0, 2.0), "two": (3.0,)}
    )
    service = EmbeddingService(provider, model="model")

    with pytest.raises(EmbeddingOutputError, match="inconsistent vector dimensions"):
        service.embed_texts(("one", "two"))


def test_rejects_dimension_mismatch_across_provider_batches() -> None:
    class ChangingDimensionProvider(FakeEmbeddingProvider):
        def embed_batch(
            self,
            texts: tuple[str, ...],
            *,
            model: str,
        ) -> Sequence[Sequence[Real]]:
            self.calls.append((texts, model))
            return (
                [(1.0, 2.0), (2.0, 3.0)]
                if texts == ("one", "two")
                else [(3.0,)]
            )

    service = EmbeddingService(ChangingDimensionProvider(), model="model")

    with pytest.raises(EmbeddingOutputError, match="inconsistent vector dimensions"):
        service.embed_texts(("one", "two", "three"))


def test_rejects_configured_dimension_mismatch() -> None:
    service = EmbeddingService(
        FakeEmbeddingProvider(vectors={"one": (1.0, 2.0)}),
        model="model",
        expected_dimensions=3,
    )

    with pytest.raises(EmbeddingOutputError, match="inconsistent vector dimensions"):
        service.embed_text("one")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_values(value: float) -> None:
    service = EmbeddingService(
        FakeEmbeddingProvider(vectors={"invalid": (value,)}),
        model="model",
    )

    with pytest.raises(EmbeddingOutputError, match="non-finite"):
        service.embed_text("invalid")


def test_rejects_non_numeric_values() -> None:
    class NonNumericProvider(FakeEmbeddingProvider):
        def embed_batch(
            self,
            texts: tuple[str, ...],
            *,
            model: str,
        ) -> Sequence[Sequence[Real]]:
            return cast(Sequence[Sequence[Real]], [("not numeric",)])

    service = EmbeddingService(NonNumericProvider(), model="model")

    with pytest.raises(EmbeddingOutputError, match="non-numeric"):
        service.embed_text("invalid")


def test_provider_failure_is_safe_and_does_not_expose_exception_details() -> None:
    provider = FakeEmbeddingProvider(
        failure=RuntimeError("secret-token=do-not-leak")
    )
    service = EmbeddingService(provider, model="model")

    with pytest.raises(EmbeddingProviderError) as error:
        service.embed_text("text")

    assert "secret-token" not in str(error.value)
    assert error.value.__cause__ is None


def test_missing_gemini_api_key_is_rejected() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(EmbeddingConfigurationError, match="API key"):
        create_embedding_service(settings)


def test_unregistered_configured_provider_is_rejected_safely() -> None:
    settings = Settings(
        embedding_provider="unavailable",
        embedding_model="test-model",
        _env_file=None,
    )

    with pytest.raises(EmbeddingConfigurationError, match="unavailable"):
        create_embedding_service(settings, {})


def test_factory_uses_configured_provider_model_and_dimension() -> None:
    provider = FakeEmbeddingProvider(vectors={"text": (0.25, 0.75)})
    settings = Settings(
        embedding_provider="fake",
        embedding_model="test-model",
        embedding_dimensions=2,
        _env_file=None,
    )
    service = create_embedding_service(settings, {"fake": provider})

    assert service.embed_text("text") == (0.25, 0.75)
    assert provider.calls == [(("text",), "test-model")]


def test_service_close_releases_provider_resources() -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider, model="model")

    service.close()

    assert provider.closed is True
