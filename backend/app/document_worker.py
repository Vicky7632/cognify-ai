import logging
from collections.abc import Callable
from threading import Event, Thread
from uuid import UUID

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import get_engine, get_session_factory
from app.repositories import document_processing_job_repository
from app.services.document_embedding_service import DocumentEmbeddingService
from app.services.document_indexing_service import (
    DocumentIndexingService,
    DocumentWorkerLockLostError,
)
from app.services.document_processing_worker import process_next_document_job
from app.services.document_vector_storage_service import (
    DocumentVectorStorageService,
)
from app.services.embedding_service import create_embedding_service
from app.services.local_document_processor import LocalDocumentProcessor
from app.vector_store.pinecone_adapter import create_pinecone_vector_store

logger = logging.getLogger(__name__)
WORKER_LOCK_KEY = 0x434F474E494659
POLL_INTERVAL_SECONDS = 1
LOCK_CHECK_INTERVAL_SECONDS = 1
LOCK_OWNERSHIP_QUERY = text(
    "SELECT EXISTS ("
    "SELECT 1 FROM pg_locks "
    "WHERE pid = :backend_pid AND locktype = 'advisory' AND granted"
    ")"
)


def run_worker(
    engine: Engine,
    session_factory: sessionmaker[Session],
    *,
    stop_requested: Callable[[], bool] = lambda: False,
) -> None:
    """Poll durable jobs while holding a PostgreSQL singleton worker lock."""
    with engine.connect() as lock_connection:
        acquisition_attempted = False
        acquisition_result_known = False
        acquired = False
        backend_pid: int | None = None
        monitor_stop: Event | None = None
        monitor: Thread | None = None
        monitor_started = False
        try:
            try:
                with lock_connection.begin():
                    acquisition_attempted = True
                    acquisition_result = lock_connection.scalar(
                        text("SELECT pg_try_advisory_lock(:lock_key)"),
                        {"lock_key": WORKER_LOCK_KEY},
                    )
                    acquisition_result_known = True
                    acquired = bool(acquisition_result)
                    if acquired:
                        backend_pid = lock_connection.scalar(
                            text("SELECT pg_backend_pid()")
                        )
            except Exception:
                if acquired:
                    acquired = False
                    _invalidate_lock_connection(lock_connection)
                raise
            if not acquired:
                raise RuntimeError("Another document processing worker is active.")

            if not isinstance(backend_pid, int):
                acquired = False
                _invalidate_lock_connection(lock_connection)
                raise RuntimeError("Unable to verify document worker lock ownership.")

            monitor_stop = Event()
            lock_lost = Event()
            monitor = Thread(
                target=_monitor_lock_ownership,
                args=(engine, backend_pid, monitor_stop, lock_lost),
                name="document-worker-lock-monitor",
                daemon=True,
            )
            monitor.start()
            monitor_started = True

            def ensure_lock_owned() -> None:
                if lock_lost.is_set() or lock_connection.invalidated:
                    lock_lost.set()
                    raise DocumentWorkerLockLostError(
                        "Document worker advisory lock was lost."
                    )
                try:
                    lock_is_held = _advisory_lock_is_held(engine, backend_pid)
                except Exception:
                    lock_lost.set()
                    raise DocumentWorkerLockLostError(
                        "Document worker advisory lock ownership could not be verified."
                    ) from None
                if not lock_is_held:
                    lock_lost.set()
                    raise DocumentWorkerLockLostError(
                        "Document worker advisory lock was lost."
                    )

            if not _advisory_lock_is_held(engine, backend_pid):
                lock_lost.set()
            ensure_lock_owned()

            with session_factory() as session:
                with session.begin():
                    recovered = (
                        document_processing_job_repository
                        .recover_interrupted_document_processing_jobs(session)
                    )
                    queued_legacy_uploads = (
                        document_processing_job_repository
                        .enqueue_unqueued_uploaded_documents(session)
                    )
            if recovered:
                logger.info("Recovered %s interrupted document jobs.", recovered)
            if queued_legacy_uploads:
                logger.info(
                    "Queued %s uploaded documents missing processing jobs.",
                    queued_legacy_uploads,
                )

            settings = get_settings()
            while not stop_requested():
                ensure_lock_owned()
                processed = process_next_document_job(
                    session_factory,
                    lambda document_id: _create_indexing_service(
                        session_factory,
                        document_id,
                        settings,
                        ensure_worker_lock=ensure_lock_owned,
                    ),
                    ensure_worker_lock=ensure_lock_owned,
                )
                ensure_lock_owned()
                if not processed:
                    monitor_stop.wait(POLL_INTERVAL_SECONDS)
        finally:
            if monitor_stop is not None:
                monitor_stop.set()
            if monitor is not None and monitor_started:
                monitor.join(timeout=LOCK_CHECK_INTERVAL_SECONDS + 1)
            if acquired:
                _release_worker_lock(lock_connection)
            elif acquisition_attempted and not acquisition_result_known:
                _invalidate_lock_connection(lock_connection)


def _release_worker_lock(lock_connection: Connection) -> None:
    if lock_connection.invalidated:
        _invalidate_lock_connection(lock_connection)
        return
    try:
        released = lock_connection.scalar(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": WORKER_LOCK_KEY},
        )
    except Exception:
        logger.error("Unable to release document worker advisory lock.")
        _invalidate_lock_connection(lock_connection)
        return
    if not released:
        logger.warning("Document worker connection did not own its advisory lock.")


def _invalidate_lock_connection(lock_connection: Connection) -> None:
    try:
        lock_connection.invalidate()
    except Exception:
        logger.error("Unable to invalidate document worker lock connection.")
        try:
            lock_connection.detach()
        except Exception:
            logger.error("Unable to detach document worker lock connection.")
            _close_lock_driver_connection(lock_connection)
            return
        try:
            lock_connection.close()
        except Exception:
            logger.error("Unable to close detached document worker lock connection.")
            _close_lock_driver_connection(lock_connection)


def _close_lock_driver_connection(lock_connection: Connection) -> None:
    try:
        driver_connection = lock_connection.connection.driver_connection
        if driver_connection is None:
            logger.error("Unable to physically close document worker lock connection.")
            return
        driver_connection.close()
    except Exception:
        logger.error("Unable to physically close document worker lock connection.")


def _advisory_lock_is_held(engine: Engine, backend_pid: int) -> bool:
    with engine.connect() as connection:
        return bool(
            connection.scalar(
                LOCK_OWNERSHIP_QUERY,
                {"backend_pid": backend_pid},
            )
        )


def _monitor_lock_ownership(
    engine: Engine,
    backend_pid: int,
    monitor_stop: Event,
    lock_lost: Event,
) -> None:
    while not monitor_stop.is_set():
        try:
            if not _advisory_lock_is_held(engine, backend_pid):
                lock_lost.set()
                return
        except Exception:
            logger.error("Unable to verify document worker lock ownership.")
            lock_lost.set()
            return
        if monitor_stop.wait(LOCK_CHECK_INTERVAL_SECONDS):
            return


def _create_indexing_service(
    session_factory: sessionmaker[Session],
    document_id: UUID,
    settings: Settings,
    *,
    ensure_worker_lock: Callable[[], None] = lambda: None,
) -> DocumentIndexingService:
    def create_storage() -> DocumentVectorStorageService:
        embedding_service = create_embedding_service(settings)
        try:
            vector_store = create_pinecone_vector_store(settings)
        except Exception:
            embedding_service.close()
            raise
        return DocumentVectorStorageService(
            DocumentEmbeddingService(embedding_service),
            vector_store,
        )

    return DocumentIndexingService(
        session_factory,
        LocalDocumentProcessor(settings.upload_root),
        create_storage,
        ensure_worker_lock=ensure_worker_lock,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_worker(get_engine(), get_session_factory())


if __name__ == "__main__":
    main()
