from collections.abc import Callable, Sequence
from math import hypot
from numbers import Real
from types import SimpleNamespace
from unittest.mock import Mock

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
from app.services.gemini_embedding_provider import (
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MAX_BATCH_SIZE,
    GeminiEmbeddingProvider,
    create_gemini_embedding_provider,
)


def fake_client(
    response_for: Callable[[list[str]], object],
) -> Mock:
    client = Mock()
    client.models.embed_content.side_effect = lambda **kwargs: response_for(
        kwargs["contents"]
    )
    return client


def test_gemini_provider_embeds_single_text_with_model_and_dimensions() -> None:
    client = fake_client(
        lambda contents: SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.25, 0.75])]
        )
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=2,
        client=client,
    )
    service = EmbeddingService(
        provider,
        model=GEMINI_EMBEDDING_MODEL,
        expected_dimensions=2,
    )

    assert service.embed_text("one text") == pytest.approx(
        (1 / (10**0.5), 3 / (10**0.5))
    )
    call = client.models.embed_content.call_args.kwargs
    assert call["model"] == GEMINI_EMBEDDING_MODEL
    assert call["contents"] == ["one text"]
    assert call["config"].output_dimensionality == 2


def test_gemini_provider_batches_and_preserves_input_order() -> None:
    texts = tuple(f"text-{index}" for index in range(GEMINI_MAX_BATCH_SIZE + 2))

    def response_for(contents: list[str]) -> SimpleNamespace:
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[float(text.split("-")[1]) + 1, 1.0]
                )
                for text in contents
            ]
        )

    client = fake_client(response_for)
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=2,
        client=client,
    )
    service = EmbeddingService(
        provider,
        model=GEMINI_EMBEDDING_MODEL,
        expected_dimensions=2,
    )

    vectors = service.embed_texts(texts)

    assert [call.kwargs["contents"] for call in client.models.embed_content.call_args_list] == [
        list(texts[:GEMINI_MAX_BATCH_SIZE]),
        list(texts[GEMINI_MAX_BATCH_SIZE:]),
    ]
    assert [hypot(*vector) for vector in vectors] == pytest.approx(
        (1.0,) * len(texts)
    )
    assert [vector[0] for vector in vectors] == sorted(
        vector[0] for vector in vectors
    )


def test_gemini_factory_initializes_sdk_client_without_leaking_key() -> None:
    settings = Settings(
        gemini_api_key="private-key-value",
        _env_file=None,
    )
    assert "private-key-value" not in repr(settings)

    provider = create_gemini_embedding_provider(settings)

    assert isinstance(provider, GeminiEmbeddingProvider)
    provider.close()


def test_gemini_factory_rejects_missing_api_key() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(EmbeddingConfigurationError, match="API key"):
        create_gemini_embedding_provider(settings)


def test_default_factory_uses_gemini_and_requires_api_key() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(EmbeddingConfigurationError, match="API key"):
        create_embedding_service(settings)


def test_gemini_provider_failure_is_safe() -> None:
    client = Mock()
    client.models.embed_content.side_effect = RuntimeError(
        "api_key=private-key-value"
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=2,
        client=client,
    )

    with pytest.raises(EmbeddingProviderError) as error:
        provider.embed_batch(("text",), model=GEMINI_EMBEDDING_MODEL)

    assert "private-key-value" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(embeddings=None),
        SimpleNamespace(embeddings=[SimpleNamespace(values=None)]),
        SimpleNamespace(embeddings=[object()]),
    ],
)
def test_gemini_provider_rejects_malformed_responses(response: object) -> None:
    client = Mock()
    client.models.embed_content.return_value = response
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=2,
        client=client,
    )

    with pytest.raises(EmbeddingOutputError, match="malformed response"):
        provider.embed_batch(("text",), model=GEMINI_EMBEDDING_MODEL)


def test_gemini_provider_rejects_configured_dimension_mismatch() -> None:
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1, 0.2])]
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=3,
        client=client,
    )
    service = EmbeddingService(
        provider,
        model=GEMINI_EMBEDDING_MODEL,
        expected_dimensions=3,
    )

    with pytest.raises(EmbeddingOutputError, match="inconsistent vector dimensions"):
        service.embed_text("text")


def test_1024_dimensional_embeddings_are_requested_validated_and_l2_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [3.0] * 1024
    client = fake_client(
        lambda _contents: SimpleNamespace(
            embeddings=[SimpleNamespace(values=values)]
        )
    )
    monkeypatch.setattr(
        "app.services.gemini_embedding_provider.genai.Client",
        lambda *, api_key: client,
    )
    settings = Settings(
        gemini_api_key="test-key",
        embedding_dimensions=1024,
        _env_file=None,
    )
    provider = create_gemini_embedding_provider(settings)
    service = EmbeddingService(
        provider,
        model=GEMINI_EMBEDDING_MODEL,
        expected_dimensions=1024,
    )

    vector = service.embed_text("text")

    assert len(vector) == 1024
    assert hypot(*vector) == pytest.approx(1.0)
    assert vector == pytest.approx((1 / 32,) * 1024)
    assert client.models.embed_content.call_args.kwargs[
        "config"
    ].output_dimensionality == 1024


def test_gemini_3072_dimensional_embeddings_are_not_renormalized() -> None:
    values = [2.0] * 3072
    client = fake_client(
        lambda _contents: SimpleNamespace(
            embeddings=[SimpleNamespace(values=values)]
        )
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=3072,
        client=client,
    )

    vector = provider.embed_batch(
        ("text",),
        model=GEMINI_EMBEDDING_MODEL,
    )[0]

    assert vector == values
    assert hypot(*vector) != pytest.approx(1.0)


def test_gemini_non_default_dimension_rejects_zero_magnitude_vector() -> None:
    client = fake_client(
        lambda _contents: SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.0, 0.0])]
        )
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=2,
        client=client,
    )

    with pytest.raises(EmbeddingOutputError, match="zero or invalid vector"):
        provider.embed_batch(("text",), model=GEMINI_EMBEDDING_MODEL)


def test_gemini_provider_rejects_batches_over_declared_limit() -> None:
    client = Mock()
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=2,
        client=client,
    )

    with pytest.raises(EmbeddingInputError, match="1 to 100"):
        provider.embed_batch(
            tuple("text" for _ in range(GEMINI_MAX_BATCH_SIZE + 1)),
            model=GEMINI_EMBEDDING_MODEL,
        )

    client.models.embed_content.assert_not_called()


def test_factory_rejects_other_gemini_models() -> None:
    settings = Settings(
        embedding_model="different-model",
        gemini_api_key="test-key",
        _env_file=None,
    )

    with pytest.raises(EmbeddingConfigurationError, match="requires"):
        create_gemini_embedding_provider(settings)


def test_response_vectors_retain_numeric_values_for_service_validation() -> None:
    response_vectors: Sequence[Sequence[Real]] = ([1, 2],)
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=list(response_vectors[0])),
        ]
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        output_dimensions=3072,
        client=client,
    )

    assert provider.embed_batch(("text",), model=GEMINI_EMBEDDING_MODEL) == (
        [1, 2],
    )
