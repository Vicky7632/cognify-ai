import json
from uuid import UUID
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.gemini_grounded_answer_provider import (
    GeminiGroundedAnswerProvider,
    create_gemini_grounded_answer_provider,
)
from app.services.grounded_answer_service import (
    GROUNDING_INSTRUCTIONS,
    GroundedAnswerConfigurationError,
    GroundedAnswerPrompt,
    GroundedAnswerProviderError,
    GroundedSource,
)


class FakeModels:
    def __init__(
        self,
        *,
        response: object | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.response = response or SimpleNamespace(text="Grounded answer.")
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: object,
    ) -> object:
        if self.failure is not None:
            raise self.failure
        self.calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return self.response


class FakeGeminiClient:
    def __init__(self, models: FakeModels | None = None) -> None:
        self.models = models or FakeModels()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_prompt() -> GroundedAnswerPrompt:
    return GroundedAnswerPrompt(
        system_instruction=GROUNDING_INSTRUCTIONS,
        query="What is RAG?",
        sources=(
            GroundedSource(
                document_id=UUID("00000000-0000-0000-0000-000000000001"),
                segment_id=UUID("00000000-0000-0000-0000-000000000002"),
                ordinal=0,
                page_number=3,
                original_filename="guide.pdf",
                file_type="pdf",
                text_content="RAG retrieves evidence before generating.",
            ),
        ),
    )


def test_gemini_provider_sends_system_instructions_and_json_context() -> None:
    prompt = make_prompt()
    fake_client = FakeGeminiClient()
    provider = GeminiGroundedAnswerProvider(
        api_key="test-key",
        model="explicit-test-model",
        client=fake_client,
    )

    answer = provider.generate_answer(prompt)

    assert answer == "Grounded answer."
    call = fake_client.models.calls[0]
    assert call["model"] == "explicit-test-model"
    assert call["config"].system_instruction == prompt.system_instruction
    payload = json.loads(call["contents"])
    assert payload == {
        "query": "What is RAG?",
        "untrusted_document_context": [
            {
                "document_id": "00000000-0000-0000-0000-000000000001",
                "segment_id": "00000000-0000-0000-0000-000000000002",
                "ordinal": 0,
                "page_number": 3,
                "original_filename": "guide.pdf",
                "file_type": "pdf",
                "text": "RAG retrieves evidence before generating.",
            }
        ],
    }
    assert payload["untrusted_document_context"][0]["text"] not in (
        call["config"].system_instruction
    )


def test_gemini_provider_replaces_sdk_exception_without_leaking_secret() -> None:
    provider = GeminiGroundedAnswerProvider(
        api_key="test-key",
        model="explicit-test-model",
        client=FakeGeminiClient(
            FakeModels(failure=RuntimeError("secret api credential"))
        ),
    )

    with pytest.raises(GroundedAnswerProviderError) as error:
        provider.generate_answer(make_prompt())

    assert str(error.value) == "The grounded answer provider request failed."
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [SimpleNamespace(text=None), SimpleNamespace(text="  ")],
)
def test_gemini_provider_rejects_missing_or_empty_answer(response: object) -> None:
    provider = GeminiGroundedAnswerProvider(
        api_key="test-key",
        model="explicit-test-model",
        client=FakeGeminiClient(FakeModels(response=response)),
    )

    with pytest.raises(GroundedAnswerProviderError, match="invalid response"):
        provider.generate_answer(make_prompt())


def test_provider_factory_requires_explicit_provider_model_and_key() -> None:
    with pytest.raises(GroundedAnswerConfigurationError, match="supported LLM"):
        create_gemini_grounded_answer_provider(Settings(_env_file=None))

    with pytest.raises(GroundedAnswerConfigurationError, match="model"):
        create_gemini_grounded_answer_provider(
            Settings(
                llm_provider="gemini",
                gemini_api_key="test-key",
                _env_file=None,
            )
        )

    with pytest.raises(GroundedAnswerConfigurationError, match="API key"):
        create_gemini_grounded_answer_provider(
            Settings(
                llm_provider="gemini",
                llm_model="explicit-test-model",
                gemini_api_key=None,
                _env_file=None,
            )
        )


def test_provider_factory_uses_explicit_model_without_defaulting() -> None:
    provider = create_gemini_grounded_answer_provider(
        Settings(
            llm_provider="gemini",
            llm_model="explicit-test-model",
            gemini_api_key="test-key",
            _env_file=None,
        ),
        client=FakeGeminiClient(),
    )

    assert isinstance(provider, GeminiGroundedAnswerProvider)
