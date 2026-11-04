from pathlib import Path

import pytest

from app.models.document import DocumentFileType
from app.transcription import (
    Transcriber,
    TranscriptionError,
    TranscriptionErrorCode,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
)


class FakeTranscriber:
    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.request: TranscriptionRequest | None = None

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.request = request
        return self.result


def test_fake_transcriber_obeys_typed_request_and_preserves_source_order() -> None:
    request = TranscriptionRequest(
        audio_path=Path("lesson.wav"),
        file_type=DocumentFileType.WAV,
    )
    expected = TranscriptionResult(
        segments=(
            TranscriptionSegment("First spoken segment.", 0.0, 1.25),
            TranscriptionSegment("Second spoken segment.", 1.5, 2.75),
        )
    )
    transcriber: Transcriber = FakeTranscriber(expected)

    result = transcriber.transcribe(request)

    assert result is expected
    assert isinstance(transcriber, FakeTranscriber)
    assert transcriber.request is request
    assert [segment.text for segment in result.segments] == [
        "First spoken segment.",
        "Second spoken segment.",
    ]


@pytest.mark.parametrize(
    "file_type",
    [DocumentFileType.PDF, DocumentFileType.TXT, DocumentFileType.MP4, "flac"],
)
def test_request_rejects_unsupported_media(file_type: object) -> None:
    with pytest.raises(TranscriptionError) as raised:
        TranscriptionRequest(Path("lesson.media"), file_type)

    assert raised.value.code is TranscriptionErrorCode.UNSUPPORTED_MEDIA
    assert str(raised.value) == "Transcription failed: unsupported_media."


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_segment_rejects_empty_transcription_text(text: str) -> None:
    with pytest.raises(TranscriptionError) as raised:
        TranscriptionSegment(text, 0.0, 1.0)

    assert raised.value.code is TranscriptionErrorCode.INVALID_TRANSCRIPTION


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds"),
    [
        (-0.1, 1.0),
        (0.0, float("inf")),
        (float("nan"), 1.0),
        (1.0, 0.5),
        (True, 1.0),
        ("0", 1.0),
    ],
)
def test_segment_rejects_invalid_timestamps(
    start_seconds: object,
    end_seconds: object,
) -> None:
    with pytest.raises(TranscriptionError) as raised:
        TranscriptionSegment("Spoken text.", start_seconds, end_seconds)

    assert raised.value.code is TranscriptionErrorCode.MALFORMED_OUTPUT


def test_result_rejects_empty_transcription() -> None:
    with pytest.raises(TranscriptionError) as raised:
        TranscriptionResult(())

    assert raised.value.code is TranscriptionErrorCode.INVALID_TRANSCRIPTION


def test_result_rejects_malformed_or_out_of_order_segments() -> None:
    with pytest.raises(TranscriptionError) as malformed:
        TranscriptionResult((object(),))
    assert malformed.value.code is TranscriptionErrorCode.MALFORMED_OUTPUT

    with pytest.raises(TranscriptionError) as out_of_order:
        TranscriptionResult(
            (
                TranscriptionSegment("Later.", 2.0, 3.0),
                TranscriptionSegment("Earlier.", 1.0, 1.5),
            )
        )
    assert out_of_order.value.code is TranscriptionErrorCode.MALFORMED_OUTPUT


def test_error_categories_are_safe_and_stable() -> None:
    error = TranscriptionError(TranscriptionErrorCode.RUNTIME_UNAVAILABLE)
    timeout = TranscriptionError(TranscriptionErrorCode.TIMEOUT)

    assert error.code is TranscriptionErrorCode.RUNTIME_UNAVAILABLE
    assert str(error) == "Transcription failed: runtime_unavailable."
    assert timeout.code is TranscriptionErrorCode.TIMEOUT
    assert str(timeout) == "Transcription failed: timeout."
