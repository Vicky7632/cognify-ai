from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol
from uuid import UUID

from app.models.document import DocumentFileType, DocumentProcessingErrorCode


@dataclass(frozen=True, slots=True)
class DocumentProcessingInput:
    document_id: UUID
    original_filename: str
    file_type: DocumentFileType
    storage_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ProcessedSegment:
    text: str
    page_number: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Processed segment text must not be empty.")
        if self.page_number is not None and (
            not isinstance(self.page_number, int) or self.page_number < 1
        ):
            raise ValueError("Page number must be positive.")
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("Both media timestamps must be supplied together.")
        if self.start_seconds is not None:
            if not isinstance(self.start_seconds, (int, float)) or not isinstance(
                self.end_seconds, (int, float)
            ):
                raise ValueError("Media timestamps must be numeric.")
            if (
                not isfinite(self.start_seconds)
                or not isfinite(self.end_seconds)
                or self.start_seconds < 0
                or self.end_seconds < self.start_seconds
            ):
                raise ValueError("Media timestamps must be nonnegative and ordered.")


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    segments: tuple[ProcessedSegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.segments, tuple):
            raise TypeError("Processing result segments must be an immutable tuple.")
        if not self.segments:
            raise ValueError("Processing result must contain at least one segment.")
        if not all(isinstance(segment, ProcessedSegment) for segment in self.segments):
            raise TypeError("Processing result contains an invalid segment.")


class DocumentProcessingError(Exception):
    """Expected failure with a safe, persistable error code only."""

    def __init__(self, code: DocumentProcessingErrorCode) -> None:
        self.code = DocumentProcessingErrorCode(code)
        super().__init__("Document processing failed.")


class DocumentProcessor(Protocol):
    def process(self, document: DocumentProcessingInput) -> ProcessingResult:
        """Process one document without accessing API or database objects."""
        ...
