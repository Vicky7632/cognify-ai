import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from app.repositories import document_processing_job_repository
from app.services.document_indexing_service import (
    DocumentIndexingService,
    DocumentWorkerLockLostError,
)

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]
IndexingServiceFactory = Callable[[UUID], DocumentIndexingService]


def process_next_document_job(
    session_factory: SessionFactory | sessionmaker[Session],
    indexing_service_factory: IndexingServiceFactory,
    *,
    ensure_worker_lock: Callable[[], None] = lambda: None,
) -> bool:
    """Claim and process one queued document job; return whether work was claimed."""
    ensure_worker_lock()
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.claim_next_document_processing_job(
                session
            )
    if job is None:
        return False

    try:
        ensure_worker_lock()
        indexing_service_factory(job.document_id).process_and_index(job.document_id)
        ensure_worker_lock()
    except DocumentWorkerLockLostError:
        raise
    except Exception:
        logger.error("Document processing job failed.")
        ensure_worker_lock()
        with session_factory() as session:
            with session.begin():
                document_processing_job_repository.fail_document_processing_job(
                    session,
                    job.job_id,
                )
    else:
        ensure_worker_lock()
        with session_factory() as session:
            with session.begin():
                document_processing_job_repository.complete_document_processing_job(
                    session,
                    job.job_id,
                )
    return True
