import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.document import (
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.services import document_service
from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessingInput,
    DocumentProcessor,
    ProcessingResult,
)
from app.services.document_service import InvalidDocumentTransitionError

logger = logging.getLogger(__name__)


class DocumentNotFoundError(LookupError):
    pass


class DocumentProcessingFailedError(RuntimeError):
    def __init__(self, code: DocumentProcessingErrorCode) -> None:
        self.code = code
        super().__init__("Document processing failed.")


class DocumentProcessingConflictError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Document processing state changed during processing.")


class DocumentProcessingPersistenceError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Unable to persist document processing state.")


SessionFactory = Callable[[], Session]


def process_document(
    session_factory: SessionFactory,
    document_id: UUID,
    processor: DocumentProcessor,
) -> ProcessingResult:
    """Process synchronously; database transactions do not span processor work."""
    try:
        with session_factory() as session:
            with session.begin():
                document = document_service.get_document(session, document_id)
                if document is None:
                    raise DocumentNotFoundError("Document not found.")
                processing_input = DocumentProcessingInput(
                    document_id=document.id,
                    original_filename=document.original_filename,
                    file_type=DocumentFileType(document.file_type),
                    storage_key=document.storage_path,
                )
                document = document_service.update_document_status(
                    session,
                    document_id,
                    DocumentStatus.PROCESSING,
                )
                if document is None:
                    raise DocumentNotFoundError("Document not found.")
    except InvalidDocumentTransitionError as error:
        raise DocumentProcessingConflictError() from error
    except SQLAlchemyError:
        logger.error("Unable to begin document processing.")
        raise DocumentProcessingPersistenceError() from None

    try:
        result = processor.process(processing_input)
        if not isinstance(result, ProcessingResult):
            raise TypeError("Processor returned an invalid result.")
    except DocumentProcessingError as error:
        _record_processing_failure(
            session_factory,
            document_id,
            error.code,
        )
        raise DocumentProcessingFailedError(error.code) from None
    except Exception:
        logger.error("Document processor failed unexpectedly.")
        _record_processing_failure(
            session_factory,
            document_id,
            DocumentProcessingErrorCode.PROCESSING_FAILED,
        )
        raise DocumentProcessingFailedError(
            DocumentProcessingErrorCode.PROCESSING_FAILED
        ) from None

    try:
        with session_factory() as session:
            with session.begin():
                document = document_service.update_document_status(
                    session,
                    document_id,
                    DocumentStatus.COMPLETED,
                )
                if document is None:
                    raise DocumentNotFoundError("Document not found.")
    except InvalidDocumentTransitionError as error:
        raise DocumentProcessingConflictError() from error
    except SQLAlchemyError:
        logger.error("Unable to complete document processing.")
        raise DocumentProcessingPersistenceError() from None

    return result


def _record_processing_failure(
    session_factory: SessionFactory,
    document_id: UUID,
    error_code: DocumentProcessingErrorCode,
) -> None:
    if error_code is not DocumentProcessingErrorCode.PROCESSING_FAILED:
        raise ValueError("Unsupported document processing error code.")

    try:
        with session_factory() as session:
            with session.begin():
                document = document_service.update_document_status(
                    session,
                    document_id,
                    DocumentStatus.FAILED,
                )
                if document is None:
                    raise DocumentNotFoundError("Document not found.")
    except InvalidDocumentTransitionError as error:
        raise DocumentProcessingConflictError() from error
    except SQLAlchemyError:
        logger.error("Unable to record document processing failure.")
        raise DocumentProcessingPersistenceError() from None
