"""Provider-neutral transcription contracts."""

from app.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from app.transcription.contracts import (
    Transcriber,
    TranscriptionError,
    TranscriptionErrorCode,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
)

__all__ = [
    "FasterWhisperTranscriber",
    "Transcriber",
    "TranscriptionError",
    "TranscriptionErrorCode",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionSegment",
]
