from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus
from app.models.document_processing_job import (
    DocumentProcessingJob,
    DocumentProcessingJobStatus,
)
from app.services import document_service

MAX_DOCUMENT_PROCESSING_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class ClaimedDocumentProcessingJob:
    job_id: UUID
    document_id: UUID


def enqueue_document_processing(
    session: Session,
    document_id: UUID,
) -> DocumentProcessingJob:
    job = DocumentProcessingJob(document_id=document_id)
    session.add(job)
    session.flush()
    return job


def enqueue_unqueued_uploaded_documents(session: Session) -> int:
    existing_job = exists().where(
        DocumentProcessingJob.document_id == Document.id
    )
    document_ids = session.scalars(
        select(Document.id).where(
            Document.processing_status == DocumentStatus.UPLOADED.value,
            ~existing_job,
        )
    ).all()
    for document_id in document_ids:
        enqueue_document_processing(session, document_id)
    return len(document_ids)


def claim_next_document_processing_job(
    session: Session,
) -> ClaimedDocumentProcessingJob | None:
    statement = (
        select(DocumentProcessingJob)
        .where(
            DocumentProcessingJob.status == DocumentProcessingJobStatus.QUEUED,
            DocumentProcessingJob.available_at <= datetime.now(UTC),
        )
        .order_by(
            DocumentProcessingJob.created_at.asc(),
            DocumentProcessingJob.id.asc(),
        )
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    job = session.scalar(statement)
    if job is None:
        return None
    job.status = DocumentProcessingJobStatus.PROCESSING.value
    job.attempts += 1
    session.flush()
    return ClaimedDocumentProcessingJob(job_id=job.id, document_id=job.document_id)


def complete_document_processing_job(
    session: Session,
    job_id: UUID,
) -> None:
    job = session.get(DocumentProcessingJob, job_id)
    if job is None or job.status != DocumentProcessingJobStatus.PROCESSING:
        raise ValueError("The processing job is not active.")
    job.status = DocumentProcessingJobStatus.COMPLETED.value
    session.flush()


def fail_document_processing_job(
    session: Session,
    job_id: UUID,
) -> None:
    job = session.get(DocumentProcessingJob, job_id)
    if job is None or job.status != DocumentProcessingJobStatus.PROCESSING:
        raise ValueError("The processing job is not active.")
    if job.attempts >= MAX_DOCUMENT_PROCESSING_ATTEMPTS:
        job.status = DocumentProcessingJobStatus.FAILED.value
    else:
        job.status = DocumentProcessingJobStatus.QUEUED.value
        job.available_at = datetime.now(UTC) + timedelta(
            seconds=2 ** (job.attempts - 1)
        )
    session.flush()


def recover_interrupted_document_processing_jobs(session: Session) -> int:
    statement = (
        select(DocumentProcessingJob)
        .where(DocumentProcessingJob.status == DocumentProcessingJobStatus.PROCESSING)
        .with_for_update()
    )
    jobs = list(session.scalars(statement).all())
    for job in jobs:
        document = session.get(Document, job.document_id)
        job.available_at = datetime.now(UTC)
        if document is None:
            job.status = DocumentProcessingJobStatus.FAILED.value
        elif document.processing_status == DocumentStatus.COMPLETED.value:
            job.status = DocumentProcessingJobStatus.COMPLETED.value
        else:
            if job.attempts >= MAX_DOCUMENT_PROCESSING_ATTEMPTS:
                if document.processing_status in (
                    DocumentStatus.PENDING.value,
                    DocumentStatus.UPLOADED.value,
                ):
                    document_service.update_document_status(
                        session,
                        document.id,
                        DocumentStatus.PROCESSING,
                    )
                if document.processing_status != DocumentStatus.FAILED.value:
                    document_service.update_document_status(
                        session,
                        document.id,
                        DocumentStatus.FAILED,
                    )
                job.status = DocumentProcessingJobStatus.FAILED.value
            else:
                if document.processing_status == DocumentStatus.PROCESSING.value:
                    document_service.update_document_status(
                        session,
                        document.id,
                        DocumentStatus.FAILED,
                    )
                job.status = DocumentProcessingJobStatus.QUEUED.value
    session.flush()
    return len(jobs)
