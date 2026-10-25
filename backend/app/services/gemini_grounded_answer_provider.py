import json
from typing import Protocol

from google import genai
from google.genai import types

from app.core.config import Settings
from app.services.grounded_answer_service import (
    GroundedAnswerConfigurationError,
    GroundedAnswerPrompt,
    GroundedAnswerProviderError,
)


class _GeminiModelClient(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
    ) -> object:
        ...


class _GeminiClient(Protocol):
    models: _GeminiModelClient

    def close(self) -> None:
        ...


class GeminiGroundedAnswerProvider:
    """Gemini generation adapter using a model explicitly selected by settings."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: _GeminiClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise GroundedAnswerConfigurationError(
                "Gemini API key is not configured for answer generation."
            )
        if not model.strip():
            raise GroundedAnswerConfigurationError(
                "An LLM model must be configured for answer generation."
            )
        try:
            self._client = (
                client if client is not None else genai.Client(api_key=api_key)
            )
        except Exception:
            raise GroundedAnswerConfigurationError(
                "Unable to initialize the grounded answer provider."
            ) from None
        self._model = model

    def generate_answer(self, prompt: GroundedAnswerPrompt) -> str:
        content = json.dumps(
            {
                "query": prompt.query,
                "untrusted_document_context": [
                    {
                        "document_id": str(source.document_id),
                        "segment_id": str(source.segment_id),
                        "ordinal": source.ordinal,
                        "page_number": source.page_number,
                        "original_filename": source.original_filename,
                        "file_type": source.file_type,
                        "text": source.text_content,
                    }
                    for source in prompt.sources
                ],
            },
            ensure_ascii=False,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=prompt.system_instruction,
                ),
            )
            answer = getattr(response, "text", None)
        except Exception:
            raise GroundedAnswerProviderError(
                "The grounded answer provider request failed."
            ) from None
        if not isinstance(answer, str) or not answer.strip():
            raise GroundedAnswerProviderError(
                "The grounded answer provider returned an invalid response."
            )
        return answer.strip()

    def close(self) -> None:
        self._client.close()


def create_gemini_grounded_answer_provider(
    settings: Settings,
    *,
    client: _GeminiClient | None = None,
) -> GeminiGroundedAnswerProvider:
    """Build the configured Gemini answer provider without choosing a model."""
    if settings.llm_provider != "gemini":
        raise GroundedAnswerConfigurationError(
            "A supported LLM provider must be configured."
        )
    if settings.llm_model is None or not settings.llm_model.strip():
        raise GroundedAnswerConfigurationError(
            "An LLM model must be configured for answer generation."
        )
    if settings.gemini_api_key is None:
        raise GroundedAnswerConfigurationError(
            "Gemini API key is not configured for answer generation."
        )
    return GeminiGroundedAnswerProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.llm_model,
        client=client,
    )
