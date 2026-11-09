from pathlib import Path
from tempfile import TemporaryDirectory

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
from app.transcription.ffmpeg_audio_extractor import FFmpegAudioExtractor


class LocalDocumentProcessor:
    """Extract or transcribe supported documents from the upload root."""

    def __init__(
        self,
        upload_root: Path,
        transcriber: Transcriber | None = None,
        *,
        audio_extractor: FFmpegAudioExtractor | None = None,
    ) -> None:
        self._upload_root = upload_root
        self._transcriber = transcriber
        self._audio_extractor = audio_extractor

    def process(self, document: DocumentProcessingInput) -> ProcessingResult:
        try:
            file_type = DocumentFileType(document.file_type)
            if file_type not in (
                DocumentFileType.PDF,
                DocumentFileType.TXT,
                DocumentFileType.MP3,
                DocumentFileType.WAV,
                DocumentFileType.MP4,
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
            if file_type is DocumentFileType.MP4:
                return self._process_mp4(path)
            return self._transcribe(path, file_type)
        except DocumentProcessingError:
            raise
        except Exception as error:
            raise DocumentProcessingError(
                DocumentProcessingErrorCode.PROCESSING_FAILED
            ) from error

    def _process_mp4(self, path: Path) -> ProcessingResult:
        if self._audio_extractor is None:
            raise DocumentProcessingError(
                DocumentProcessingErrorCode.PROCESSING_FAILED
            )
        with TemporaryDirectory(prefix="cognify-mp4-") as temporary_directory:
            audio_path = Path(temporary_directory) / "extracted.wav"
            self._audio_extractor.extract(path, audio_path)
            return self._transcribe(audio_path, DocumentFileType.WAV)

    def _transcribe(
        self,
        audio_path: Path,
        file_type: DocumentFileType,
    ) -> ProcessingResult:
        if self._transcriber is None:
            raise DocumentProcessingError(
                DocumentProcessingErrorCode.PROCESSING_FAILED
            )
        result = self._transcriber.transcribe(
            TranscriptionRequest(audio_path=audio_path, file_type=file_type)
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
