from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Protocol

from app.models.document import DocumentFileType


class TranscriptionErrorCode(StrEnum):
    UNSUPPORTED_MEDIA = "unsupported_media"
    INVALID_TRANSCRIPTION = "invalid_transcription"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    TIMEOUT = "timeout"
    MALFORMED_OUTPUT = "malformed_output"


class TranscriptionError(RuntimeError):
    """Safe transcription failure that exposes only a stable error category."""

    def __init__(self, code: TranscriptionErrorCode) -> None:
        self.code = TranscriptionErrorCode(code)
        super().__init__(f"Transcription failed: {self.code.value}.")


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio_path: Path = field(repr=False)
    file_type: DocumentFileType

    def __post_init__(self) -> None:
        if not isinstance(self.audio_path, Path):
            raise TypeError("audio_path must be a Path.")
        try:
            file_type = DocumentFileType(self.file_type)
        except (TypeError, ValueError) as error:
            raise TranscriptionError(
                TranscriptionErrorCode.UNSUPPORTED_MEDIA
            ) from error
        if file_type not in (DocumentFileType.MP3, DocumentFileType.WAV):
            raise TranscriptionError(TranscriptionErrorCode.UNSUPPORTED_MEDIA)
        object.__setattr__(self, "file_type", file_type)


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    text: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise TranscriptionError(
                TranscriptionErrorCode.INVALID_TRANSCRIPTION
            )
        if (
            isinstance(self.start_seconds, bool)
            or not isinstance(self.start_seconds, (int, float))
            or isinstance(self.end_seconds, bool)
            or not isinstance(self.end_seconds, (int, float))
            or not isfinite(self.start_seconds)
            or not isfinite(self.end_seconds)
            or self.start_seconds < 0
            or self.end_seconds < self.start_seconds
        ):
            raise TranscriptionError(TranscriptionErrorCode.MALFORMED_OUTPUT)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    segments: tuple[TranscriptionSegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.segments, tuple):
            raise TranscriptionError(TranscriptionErrorCode.MALFORMED_OUTPUT)
        if not self.segments:
            raise TranscriptionError(
                TranscriptionErrorCode.INVALID_TRANSCRIPTION
            )
        if any(
            not isinstance(segment, TranscriptionSegment)
            for segment in self.segments
        ):
            raise TranscriptionError(TranscriptionErrorCode.MALFORMED_OUTPUT)
        if any(
            current.start_seconds < previous.start_seconds
            for previous, current in zip(self.segments, self.segments[1:])
        ):
            raise TranscriptionError(TranscriptionErrorCode.MALFORMED_OUTPUT)


class Transcriber(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Return timestamped text in source order without exposing provider details."""
        ...
