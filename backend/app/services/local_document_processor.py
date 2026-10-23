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


class LocalDocumentProcessor:
    """Extract PDF pages and strict UTF-8 text from files in the upload root."""

    def __init__(self, upload_root: Path) -> None:
        self._upload_root = upload_root

    def process(self, document: DocumentProcessingInput) -> ProcessingResult:
        try:
            file_type = DocumentFileType(document.file_type)
            if file_type not in (DocumentFileType.PDF, DocumentFileType.TXT):
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
            return self._process_text(path)
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
