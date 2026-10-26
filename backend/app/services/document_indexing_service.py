import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import DocumentStatus
from app.services import document_service
from app.services.document_processing_service import (
    DocumentProcessor,
    process_document,
)
from app.services.document_vector_storage_service import (
    DocumentVectorStorageService,
)

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]
VectorStorageFactory = Callable[[], DocumentVectorStorageService]


class DocumentIndexingFailedError(RuntimeError):
    """Raised when processing, embedding, or vector storage does not complete."""

    def __init__(self) -> None:
        super().__init__("Document processing and indexing failed.")


class DocumentWorkerLockLostError(RuntimeError):
    """Raised when a worker loses ownership of its PostgreSQL advisory lock."""


class DocumentIndexingService:
    """Run extraction, persistence, embedding, and indexing as separate stages."""

    def __init__(
        self,
        session_factory: SessionFactory | sessionmaker[Session],
        processor: DocumentProcessor,
        vector_storage_factory: VectorStorageFactory,
        *,
        ensure_worker_lock: Callable[[], None] = lambda: None,
    ) -> None:
        self._session_factory = session_factory
        self._processor = processor
        self._vector_storage_factory = vector_storage_factory
        self._ensure_worker_lock = ensure_worker_lock

    def process_and_index(self, document_id: UUID) -> None:
        try:
            self._ensure_worker_lock()
            process_document(
                self._session_factory,
                document_id,
                self._processor,
                ensure_worker_lock=self._ensure_worker_lock,
            )
            self._ensure_worker_lock()
        except DocumentWorkerLockLostError:
            raise
        except Exception:
            self._ensure_worker_lock()
            self._mark_failed(document_id)
            raise DocumentIndexingFailedError() from None

        storage: DocumentVectorStorageService | None = None
        try:
            storage = self._vector_storage_factory()
            with self._session_factory() as session:
                segments = storage.load_document_segments(session, document_id)

            self._ensure_worker_lock()
            storage.embed_and_store_segments(
                segments,
                before_upsert=self._ensure_worker_lock,
            )
            self._ensure_worker_lock()
            with self._session_factory() as session:
                with session.begin():
                    document = document_service.update_document_status(
                        session,
                        document_id,
                        DocumentStatus.COMPLETED,
                    )
                    if document is None:
                        raise LookupError("Document not found.")
        except DocumentWorkerLockLostError:
            raise
        except Exception:
            logger.error("Document embedding and vector storage failed.")
            self._ensure_worker_lock()
            self._mark_failed(document_id)
            raise DocumentIndexingFailedError() from None
        finally:
            if storage is not None:
                try:
                    storage.close()
                except Exception:
                    logger.error("Unable to close document indexing providers.")

    def _mark_failed(self, document_id: UUID) -> None:
        try:
            with self._session_factory() as session:
                with session.begin():
                    document = document_service.get_document(session, document_id)
                    if (
                        document is not None
                        and document.processing_status == DocumentStatus.PROCESSING.value
                    ):
                        document_service.update_document_status(
                            session,
                            document_id,
                            DocumentStatus.FAILED,
                        )
        except SQLAlchemyError:
            logger.error("Unable to record document indexing failure.")
