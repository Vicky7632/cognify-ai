"""Provider-neutral transcription contracts."""

from app.transcription.contracts import (
    Transcriber,
    TranscriptionError,
    TranscriptionErrorCode,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
)

__all__ = [
    "Transcriber",
    "TranscriptionError",
    "TranscriptionErrorCode",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionSegment",
]
