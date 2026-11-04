from pathlib import Path

from pypdf import PdfReader

from app.models.document import DocumentFileType, DocumentProcessingErrorCode
from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessingInput,
    ProcessingResult,
    ProcessedSegment,
)
from app.services.file_storage import storage_path_for_key
from app.transcription import (
    Transcriber,
    TranscriptionRequest,
)


class LocalDocumentProcessor:
    """Extract PDF pages and strict UTF-8 text from files in the upload root."""

    def __init__(
        self,
        upload_root: Path,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._upload_root = upload_root
        self._transcriber = transcriber

    def process(self, document: DocumentProcessingInput) -> ProcessingResult:
        try:
            file_type = DocumentFileType(document.file_type)
            if file_type not in (
                DocumentFileType.PDF,
                DocumentFileType.TXT,
                DocumentFileType.MP3,
                DocumentFileType.WAV,
            ):
                raise DocumentProcessingError(
                    DocumentProcessingErrorCode.PROCESSING_FAILED
                )
            path = storage_path_for_key(document.storage_key, self._upload_root)
            if not path.is_file():
                raise DocumentProcessingError(
                    DocumentProcessingErrorCode.PROCESSING_FAILED
                )
            if file_type is DocumentFileType.PDF:
                return self._process_pdf(path)
            if file_type is DocumentFileType.TXT:
                return self._process_text(path)
            if self._transcriber is None:
                raise DocumentProcessingError(
                    DocumentProcessingErrorCode.PROCESSING_FAILED
                )
            result = self._transcriber.transcribe(
                TranscriptionRequest(audio_path=path, file_type=file_type)
            )
            return ProcessingResult(
                segments=tuple(
                    ProcessedSegment(
                        text=segment.text,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                    )
                    for segment in result.segments
                )
            )
        except DocumentProcessingError:
            raise
        except Exception as error:
            raise DocumentProcessingError(
                DocumentProcessingErrorCode.PROCESSING_FAILED
            ) from error

    @staticmethod
    def _process_text(path: Path) -> ProcessingResult:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
        if not text.strip():
            raise DocumentProcessingError(
                DocumentProcessingErrorCode.PROCESSING_FAILED
            )
        return ProcessingResult(segments=(ProcessedSegment(text=text),))

    @staticmethod
    def _process_pdf(path: Path) -> ProcessingResult:
        segments: list[ProcessedSegment] = []
        with path.open("rb") as source:
            reader = PdfReader(source, strict=True)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    segments.append(
                        ProcessedSegment(
                            text=text.strip(),
                            page_number=page_number,
                        )
                    )
        if not segments:
            raise DocumentProcessingError(
                DocumentProcessingErrorCode.PROCESSING_FAILED
            )
        return ProcessingResult(segments=tuple(segments))
