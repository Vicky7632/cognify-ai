from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Callable, Generator, Sequence
from datetime import UTC, datetime, timedelta
from io import BytesIO
from numbers import Real
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import document_worker
from app.core.config import Settings
from app.db.base import Base
from app.models.document import (
    Document,
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.models.document_processing_job import (
    DocumentProcessingJob,
    DocumentProcessingJobStatus,
)
from app.models.document_segment import DocumentSegment
from app.repositories import document_processing_job_repository
from app.repositories.document_repository import create_document
from app.services.document_embedding_service import DocumentEmbeddingService
from app.services.document_indexing_service import (
    DocumentIndexingFailedError,
    DocumentIndexingService,
    DocumentWorkerLockLostError,
)
from app.services.document_processing_worker import process_next_document_job
from app.services.local_document_processor import LocalDocumentProcessor
from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessingInput,
    ProcessedSegment,
    ProcessingResult,
)
from app.services.document_vector_storage_service import (
    DocumentVectorStorageService,
)
from app.services.embedding_service import EmbeddingProviderError
from app.vector_store.protocol import (
    VectorRecord,
    VectorSearchResult,
    VectorStoreError,
)
from app.transcription import (
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
)


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'document-indexing-tests.db'}",
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_uploaded_document(session_factory: sessionmaker[Session]) -> UUID:
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename="lesson.txt",
                file_type=DocumentFileType.TXT.value,
                storage_path="generated-key.txt",
                processing_status=DocumentStatus.UPLOADED,
            )
            return document.id


def create_uploaded_audio_document(
    session_factory: sessionmaker[Session],
    upload_root: Path,
    file_type: DocumentFileType,
    filename: str,
) -> tuple[UUID, UUID, str]:
    return create_uploaded_file_document(
        session_factory,
        upload_root,
        file_type,
        filename,
        b"fake audio bytes",
    )


def create_uploaded_file_document(
    session_factory: sessionmaker[Session],
    upload_root: Path,
    file_type: DocumentFileType,
    filename: str,
    content: bytes,
) -> tuple[UUID, UUID, str]:
    storage_key = f"{uuid4().hex}.{file_type.value}"
    (upload_root / storage_key).write_bytes(content)
    with session_factory() as session:
        with session.begin():
            document = create_document(
                session,
                original_filename=f"{filename}.{file_type.value}",
                file_type=file_type.value,
                storage_path=storage_key,
                processing_status=DocumentStatus.UPLOADED,
            )
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document.id,
            )
            return document.id, job.id, storage_key


def make_pdf_with_text(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(
        f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    )
    page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def make_audio_indexing_service(
    session_factory: sessionmaker[Session],
    upload_root: Path,
    transcriber,
    audio_extractor=None,
) -> tuple[DocumentIndexingService, FakeEmbeddingService, FakeVectorStore]:
    embedding = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        session_factory,
        LocalDocumentProcessor(
            upload_root,
            transcriber,
            audio_extractor=audio_extractor,
        ),
        lambda: DocumentVectorStorageService(
            DocumentEmbeddingService(embedding),
            vector_store,
        ),
    )
    return service, embedding, vector_store


class FakeWorkerLockConnection:
    invalidated = False

    def __init__(self) -> None:
        self.unlock_attempts = 0

    def __enter__(self) -> FakeWorkerLockConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def begin(self):
        return nullcontext()

    def scalar(self, statement: object, _parameters=None) -> bool | int:
        sql = str(statement)
        if "pg_try_advisory_lock" in sql:
            return True
        if "pg_backend_pid" in sql:
            return 42
        if "pg_advisory_unlock" in sql:
            self.unlock_attempts += 1
            return True
        raise AssertionError(f"Unexpected SQL statement: {sql}")


class FakeWorkerEngine:
    def __init__(self, connection: FakeWorkerLockConnection) -> None:
        self.connection = connection

    def connect(self) -> FakeWorkerLockConnection:
        return self.connection


class FakeProcessor:
    def __init__(
        self,
        result: ProcessingResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.result = result or ProcessingResult(
            segments=(
                ProcessedSegment("First persisted section."),
                ProcessedSegment("Second persisted section."),
            )
        )
        self.failure = failure

    def process(self, document: DocumentProcessingInput) -> ProcessingResult:
        if self.failure is not None:
            raise self.failure
        return self.result


class FakeEmbeddingService:
    def __init__(
        self,
        failure: Exception | None = None,
        before_embed: Callable[[], None] | None = None,
    ) -> None:
        self.failure = failure
        self.before_embed = before_embed
        self.calls: list[tuple[str, ...]] = []
        self.closed = False

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        if self.before_embed is not None:
            self.before_embed()
        self.calls.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        return tuple((float(index), 1.0) for index in range(len(texts)))

    def close(self) -> None:
        self.closed = True


class FakeVectorStore:
    def __init__(
        self,
        failure: Exception | None = None,
        before_upsert: Callable[[], None] | None = None,
    ) -> None:
        self.failure = failure
        self.before_upsert = before_upsert
        self.calls: list[tuple[VectorRecord, ...]] = []
        self.closed = False

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if self.before_upsert is not None:
            self.before_upsert()
        self.calls.append(tuple(records))
        if self.failure is not None:
            raise self.failure

    def search(
        self,
        query_vector: Sequence[Real],
        *,
        top_k: int,
        document_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


def make_indexing_service(
    session_factory: sessionmaker[Session],
    *,
    processor: FakeProcessor | None = None,
    embedding_failure: Exception | None = None,
    vector_failure: Exception | None = None,
) -> tuple[DocumentIndexingService, FakeEmbeddingService, FakeVectorStore]:
    embedding = FakeEmbeddingService(embedding_failure)
    vector_store = FakeVectorStore(vector_failure)
    service = DocumentIndexingService(
        session_factory,
        processor or FakeProcessor(),
        lambda: DocumentVectorStorageService(
            DocumentEmbeddingService(embedding),
            vector_store,
        ),
    )
    return service, embedding, vector_store


def get_document(
    session_factory: sessionmaker[Session],
    document_id: UUID,
) -> Document:
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document is not None
        return document


def get_segments(
    session_factory: sessionmaker[Session],
    document_id: UUID,
) -> list[DocumentSegment]:
    with session_factory() as session:
        return list(
            session.scalars(
                select(DocumentSegment)
                .where(DocumentSegment.document_id == document_id)
                .order_by(DocumentSegment.ordinal)
            )
        )


def assert_no_active_transaction(active_transactions: int) -> None:
    assert active_transactions == 0


def test_processing_embedding_and_upsert_complete_document_only_after_success(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    service, embedding, vector_store = make_indexing_service(session_factory)

    service.process_and_index(document_id)

    document = get_document(session_factory, document_id)
    segments = get_segments(session_factory, document_id)
    assert document.processing_status == DocumentStatus.COMPLETED.value
    assert document.processing_error_code is None
    assert [segment.text_content for segment in segments] == [
        "First persisted section.",
        "Second persisted section.",
    ]
    assert embedding.calls == [
        ("First persisted section.", "Second persisted section.")
    ]
    assert [
        (record.metadata.document_id, record.metadata.segment_id, record.metadata.ordinal)
        for record in vector_store.calls[0]
    ] == [
        (document_id, segments[0].id, 0),
        (document_id, segments[1].id, 1),
    ]
    assert embedding.closed is True
    assert vector_store.closed is True


def test_external_embedding_and_upsert_run_without_open_database_transaction(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    engine = session_factory.kw["bind"]
    assert engine is not None
    active_transactions = 0

    def transaction_started(_connection) -> None:
        nonlocal active_transactions
        active_transactions += 1

    def transaction_ended(_connection) -> None:
        nonlocal active_transactions
        active_transactions -= 1

    event.listen(engine, "begin", transaction_started)
    event.listen(engine, "commit", transaction_ended)
    event.listen(engine, "rollback", transaction_ended)
    embedding = FakeEmbeddingService(
        before_embed=lambda: assert_no_active_transaction(active_transactions)
    )
    vector_store = FakeVectorStore(
        before_upsert=lambda: assert_no_active_transaction(active_transactions)
    )
    service = DocumentIndexingService(
        session_factory,
        FakeProcessor(),
        lambda: DocumentVectorStorageService(
            DocumentEmbeddingService(embedding),
            vector_store,
        ),
    )

    try:
        service.process_and_index(document_id)
    finally:
        event.remove(engine, "begin", transaction_started)
        event.remove(engine, "commit", transaction_ended)
        event.remove(engine, "rollback", transaction_ended)

    assert active_transactions == 0


def test_extraction_failure_marks_document_failed_without_embedding(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    processor = FakeProcessor(
        failure=DocumentProcessingError(DocumentProcessingErrorCode.PROCESSING_FAILED)
    )
    service, embedding, vector_store = make_indexing_service(
        session_factory,
        processor=processor,
    )

    with pytest.raises(
        DocumentIndexingFailedError,
        match="Document processing and indexing failed",
    ):
        service.process_and_index(document_id)

    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
    assert get_segments(session_factory, document_id) == []
    assert embedding.calls == []
    assert vector_store.calls == []


@pytest.mark.parametrize(
    ("embedding_failure", "vector_failure"),
    [
        (EmbeddingProviderError("provider secret detail"), None),
        (None, VectorStoreError("vector provider secret detail")),
    ],
)
def test_embedding_or_upsert_failure_never_marks_document_completed(
    session_factory: sessionmaker[Session],
    embedding_failure: Exception | None,
    vector_failure: Exception | None,
) -> None:
    document_id = create_uploaded_document(session_factory)
    service, embedding, vector_store = make_indexing_service(
        session_factory,
        embedding_failure=embedding_failure,
        vector_failure=vector_failure,
    )

    with pytest.raises(DocumentIndexingFailedError) as error:
        service.process_and_index(document_id)

    assert str(error.value) == "Document processing and indexing failed."
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
    assert len(get_segments(session_factory, document_id)) == 2
    assert "secret detail" not in str(error.value)
    if embedding_failure is not None:
        assert vector_store.calls == []
    else:
        assert len(vector_store.calls) == 1


def test_reprocessing_reuses_segment_ids_and_vector_ids(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    first_service, _, first_store = make_indexing_service(
        session_factory,
        vector_failure=VectorStoreError("temporary failure"),
    )
    with pytest.raises(DocumentIndexingFailedError):
        first_service.process_and_index(document_id)
    original_segment_ids = [segment.id for segment in get_segments(
        session_factory,
        document_id,
    )]

    retry_service, _, retry_store = make_indexing_service(session_factory)
    retry_service.process_and_index(document_id)
    retried_segment_ids = [segment.id for segment in get_segments(
        session_factory,
        document_id,
    )]

    assert retried_segment_ids == original_segment_ids
    assert [
        record.id for record in first_store.calls[0]
    ] == [record.id for record in retry_store.calls[0]]
    assert get_document(session_factory, document_id).processing_status == (
        DocumentStatus.COMPLETED.value
    )


def test_durable_job_worker_marks_job_complete_after_indexing(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id
    service, _, _ = make_indexing_service(session_factory)

    processed = process_next_document_job(
        session_factory,
        lambda _document_id: service,
    )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.COMPLETED.value
        assert job.attempts == 1
    assert processed is True
    assert get_document(session_factory, document_id).processing_status == (
        DocumentStatus.COMPLETED.value
    )


def test_background_worker_transcribes_mp3_and_wav_and_persists_segments(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    mp3_document_id, mp3_job_id, _mp3_storage_key = create_uploaded_audio_document(
        session_factory,
        upload_root,
        DocumentFileType.MP3,
        "audio-mp3",
    )
    wav_document_id, wav_job_id, _wav_storage_key = create_uploaded_audio_document(
        session_factory,
        upload_root,
        DocumentFileType.WAV,
        "audio-wav",
    )
    settings = Settings(
        upload_root=upload_root,
        transcription_model_path=tmp_path / "model",
        _env_file=None,
    )
    monkeypatch.setattr(document_worker, "get_settings", lambda: settings)
    monkeypatch.setattr(document_worker, "_advisory_lock_is_held", lambda *_args: True)
    monkeypatch.setattr(document_worker, "POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        document_worker,
        "create_embedding_service",
        lambda _settings: FakeEmbeddingService(),
    )
    monkeypatch.setattr(
        document_worker,
        "create_pinecone_vector_store",
        lambda _settings: FakeVectorStore(),
    )

    expected_by_type = {
        DocumentFileType.MP3: ("MP3 transcript.", 0.25, 1.75),
        DocumentFileType.WAV: ("WAV transcript.", 2.0, 3.5),
    }
    created_transcribers = []

    class FakeTranscriber:
        def __init__(
            self,
            _model_path,
            *,
            compute_type: str,
            cpu_threads: int,
            max_audio_seconds: float,
        ) -> None:
            assert compute_type == settings.transcription_compute_type
            assert cpu_threads == settings.transcription_cpu_threads
            assert max_audio_seconds == settings.transcription_max_audio_seconds
            self.requests: list[TranscriptionRequest] = []
            created_transcribers.append(self)

        def transcribe(
            self,
            request: TranscriptionRequest,
        ) -> TranscriptionResult:
            self.requests.append(request)
            text, start_seconds, end_seconds = expected_by_type[request.file_type]
            return TranscriptionResult(
                (
                    TranscriptionSegment(
                        text,
                        start_seconds,
                        end_seconds,
                    ),
                )
            )

    monkeypatch.setattr(
        document_worker,
        "FasterWhisperTranscriber",
        FakeTranscriber,
    )
    lock_connection = FakeWorkerLockConnection()

    worker_checks = 0

    def stop_after_audio_jobs() -> bool:
        nonlocal worker_checks
        worker_checks += 1
        return worker_checks > 5 or bool(
            created_transcribers
            and len(created_transcribers[0].requests) >= 2
        )

    document_worker.run_worker(
        FakeWorkerEngine(lock_connection),
        session_factory,
        stop_requested=stop_after_audio_jobs,
    )

    assert len(created_transcribers) == 1
    assert {request.file_type for request in created_transcribers[0].requests} == {
        DocumentFileType.MP3,
        DocumentFileType.WAV,
    }
    for document_id, job_id, file_type in (
        (mp3_document_id, mp3_job_id, DocumentFileType.MP3),
        (wav_document_id, wav_job_id, DocumentFileType.WAV),
    ):
        expected_text, start_seconds, end_seconds = expected_by_type[file_type]
        persisted_segments = get_segments(session_factory, document_id)
        assert [segment.text_content for segment in persisted_segments] == [
            expected_text
        ]
        assert [
            (segment.start_seconds, segment.end_seconds)
            for segment in persisted_segments
        ] == [(start_seconds, end_seconds)]
        assert get_document(session_factory, document_id).processing_status == (
            DocumentStatus.COMPLETED.value
        )
        with session_factory() as session:
            job = session.get(DocumentProcessingJob, job_id)
            assert job is not None
            assert job.status == DocumentProcessingJobStatus.COMPLETED.value
            assert job.attempts == 1
    assert lock_connection.unlock_attempts == 1


def test_worker_processes_supported_formats_sequentially_including_mp4(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    files = (
        (
            DocumentFileType.PDF,
            "lesson-pdf",
            make_pdf_with_text("PDF worker text."),
        ),
        (DocumentFileType.MP4, "lesson-mp4", b"video placeholder"),
        (DocumentFileType.TXT, "lesson-txt", b"TXT worker text."),
        (DocumentFileType.MP3, "lesson-mp3", b"fake MP3 audio"),
        (DocumentFileType.WAV, "lesson-wav", b"fake WAV audio"),
    )
    uploaded: list[tuple[UUID, UUID, DocumentFileType, str]] = []
    for index, (file_type, filename, content) in enumerate(files):
        uploaded_file = create_uploaded_file_document(
            session_factory,
            upload_root,
            file_type,
            filename,
            content,
        )
        uploaded.append((*uploaded_file[:2], file_type, uploaded_file[2]))
        with session_factory() as session:
            with session.begin():
                job = session.get(DocumentProcessingJob, uploaded_file[1])
                assert job is not None
                job.created_at = datetime.now(UTC) + timedelta(seconds=index)

    settings = Settings(
        upload_root=upload_root,
        transcription_model_path=tmp_path / "model",
        _env_file=None,
    )
    monkeypatch.setattr(document_worker, "get_settings", lambda: settings)
    monkeypatch.setattr(
        document_worker,
        "_advisory_lock_is_held",
        lambda *_args: True,
    )

    processing_order: list[DocumentFileType] = []
    original_processor = LocalDocumentProcessor

    class TrackingLocalDocumentProcessor:
        def __init__(
            self,
            root: Path,
            transcriber=None,
            *,
            audio_extractor=None,
        ) -> None:
            self._processor = original_processor(
                root,
                transcriber,
                audio_extractor=audio_extractor,
            )

        def process(
            self,
            processing_input: DocumentProcessingInput,
        ) -> ProcessingResult:
            processing_order.append(processing_input.file_type)
            return self._processor.process(processing_input)

    monkeypatch.setattr(
        document_worker,
        "LocalDocumentProcessor",
        TrackingLocalDocumentProcessor,
    )

    expected_audio = {
        DocumentFileType.MP3: ("MP3 worker transcript.", 0.25, 1.25),
        DocumentFileType.WAV: ("WAV worker transcript.", 1.5, 2.5),
        DocumentFileType.MP4: ("MP4 worker transcript.", 2.0, 3.0),
    }

    class FakeTranscriber:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.requests: list[TranscriptionRequest] = []

        def transcribe(
            self,
            request: TranscriptionRequest,
        ) -> TranscriptionResult:
            self.requests.append(request)
            audio_bytes = request.audio_path.read_bytes()
            if audio_bytes == b"MP4 extracted audio":
                text, start_seconds, end_seconds = expected_audio[
                    DocumentFileType.MP4
                ]
            elif audio_bytes == b"fake MP3 audio":
                text, start_seconds, end_seconds = expected_audio[
                    DocumentFileType.MP3
                ]
            else:
                text, start_seconds, end_seconds = expected_audio[
                    DocumentFileType.WAV
                ]
            return TranscriptionResult(
                (TranscriptionSegment(text, start_seconds, end_seconds),)
            )

    transcribers: list[FakeTranscriber] = []

    def create_transcriber(*args: object, **kwargs: object) -> FakeTranscriber:
        transcriber = FakeTranscriber(*args, **kwargs)
        transcribers.append(transcriber)
        return transcriber

    monkeypatch.setattr(
        document_worker,
        "FasterWhisperTranscriber",
        create_transcriber,
    )

    extracted_paths: list[Path] = []

    class FakeAudioExtractor:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, Path]] = []

        def extract(self, input_video: Path, output_audio: Path) -> Path:
            self.calls.append((input_video, output_audio))
            extracted_paths.append(output_audio)
            output_audio.write_bytes(b"MP4 extracted audio")
            return output_audio

    audio_extractors: list[FakeAudioExtractor] = []

    def create_audio_extractor() -> FakeAudioExtractor:
        extractor = FakeAudioExtractor()
        audio_extractors.append(extractor)
        return extractor

    monkeypatch.setattr(
        document_worker,
        "FFmpegAudioExtractor",
        create_audio_extractor,
    )

    embeddings: list[FakeEmbeddingService] = []
    vector_stores: list[FakeVectorStore] = []

    def create_embedding(_settings: Settings) -> FakeEmbeddingService:
        embedding = FakeEmbeddingService()
        embeddings.append(embedding)
        return embedding

    def create_vector_store(_settings: Settings) -> FakeVectorStore:
        vector_store = FakeVectorStore()
        vector_stores.append(vector_store)
        return vector_store

    monkeypatch.setattr(
        document_worker,
        "create_embedding_service",
        create_embedding,
    )
    monkeypatch.setattr(
        document_worker,
        "create_pinecone_vector_store",
        create_vector_store,
    )

    supported = uploaded
    lock_connection = FakeWorkerLockConnection()

    def all_jobs_reached_expected_state() -> bool:
        with session_factory() as session:
            supported_jobs = [
                session.get(DocumentProcessingJob, item[1])
                for item in supported
            ]
        return (
            all(
                job is not None
                and job.status == DocumentProcessingJobStatus.COMPLETED.value
                for job in supported_jobs
            )
        )

    document_worker.run_worker(
        FakeWorkerEngine(lock_connection),
        session_factory,
        stop_requested=all_jobs_reached_expected_state,
    )

    assert processing_order == [item[2] for item in uploaded]
    assert len(transcribers) == 1
    assert [request.file_type for request in transcribers[0].requests] == [
        DocumentFileType.WAV,
        DocumentFileType.MP3,
        DocumentFileType.WAV,
    ]
    assert len(audio_extractors) == 1
    assert len(audio_extractors[0].calls) == 1
    assert audio_extractors[0].calls[0][0] == upload_root / uploaded[1][3]
    assert audio_extractors[0].calls[0][1].suffix == ".wav"
    assert not extracted_paths[0].parent.exists()
    assert (upload_root / uploaded[1][3]).exists()

    for document_id, job_id, file_type, _storage_key in supported:
        document = get_document(session_factory, document_id)
        assert document.processing_status == DocumentStatus.COMPLETED.value
        assert document.processing_error_code is None
        with session_factory() as session:
            job = session.get(DocumentProcessingJob, job_id)
            assert job is not None
            assert job.status == DocumentProcessingJobStatus.COMPLETED.value
            assert job.attempts == 1

        segments = get_segments(session_factory, document_id)
        assert len(segments) == 1
        if file_type is DocumentFileType.PDF:
            assert segments[0].text_content == "PDF worker text."
            assert segments[0].page_number == 1
            assert segments[0].start_seconds is None
        elif file_type is DocumentFileType.TXT:
            assert segments[0].text_content == "TXT worker text."
            assert segments[0].page_number is None
            assert segments[0].start_seconds is None
        else:
            expected_text, start_seconds, end_seconds = expected_audio[file_type]
            assert segments[0].text_content == expected_text
            assert (
                segments[0].start_seconds,
                segments[0].end_seconds,
            ) == (start_seconds, end_seconds)

    assert len(embeddings) == len(supported)
    assert len(vector_stores) == len(supported)
    assert all(embedding.closed for embedding in embeddings)
    assert all(vector_store.closed for vector_store in vector_stores)
    assert lock_connection.unlock_attempts == 1


def test_audio_failure_retries_without_blocking_next_job_and_exhausts(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    (
        failed_document_id,
        failed_job_id,
        failed_storage_key,
    ) = create_uploaded_audio_document(
        session_factory,
        upload_root,
        DocumentFileType.MP3,
        "fails",
    )
    (
        successful_document_id,
        successful_job_id,
        successful_storage_key,
    ) = create_uploaded_audio_document(
        session_factory,
        upload_root,
        DocumentFileType.WAV,
        "succeeds",
    )
    with session_factory() as session:
        with session.begin():
            failed_job = session.get(DocumentProcessingJob, failed_job_id)
            successful_job = session.get(DocumentProcessingJob, successful_job_id)
            assert failed_job is not None
            assert successful_job is not None
            successful_job.created_at = datetime.now(UTC)
            failed_job.created_at = successful_job.created_at - timedelta(seconds=2)

    class FakeTranscriber:
        def __init__(self) -> None:
            self.requests: list[TranscriptionRequest] = []

        def transcribe(
            self,
            request: TranscriptionRequest,
        ) -> TranscriptionResult:
            self.requests.append(request)
            if request.audio_path.name == failed_storage_key:
                raise RuntimeError("controlled transcription failure")
            return TranscriptionResult(
                (TranscriptionSegment("Recovered transcript.", 0.0, 1.0),)
            )

    transcriber = FakeTranscriber()

    def create_service(_document_id: UUID) -> DocumentIndexingService:
        return make_audio_indexing_service(
            session_factory,
            upload_root,
            transcriber,
        )[0]

    assert process_next_document_job(session_factory, create_service)
    with session_factory() as session:
        failed_job = session.get(DocumentProcessingJob, failed_job_id)
        assert failed_job is not None
        assert failed_job.status == DocumentProcessingJobStatus.QUEUED.value
        assert failed_job.attempts == 1
    assert get_document(session_factory, failed_document_id).processing_status == (
        DocumentStatus.FAILED.value
    )

    assert process_next_document_job(session_factory, create_service)
    with session_factory() as session:
        successful_job = session.get(DocumentProcessingJob, successful_job_id)
        assert successful_job is not None
        assert successful_job.status == DocumentProcessingJobStatus.COMPLETED.value
        assert successful_job.attempts == 1
    assert [
        segment.text_content
        for segment in get_segments(session_factory, successful_document_id)
    ] == ["Recovered transcript."]

    for _ in range(
        document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS - 1
    ):
        with session_factory() as session:
            with session.begin():
                failed_job = session.get(DocumentProcessingJob, failed_job_id)
                assert failed_job is not None
                failed_job.available_at = datetime.now(UTC)
        assert process_next_document_job(session_factory, create_service)

    with session_factory() as session:
        failed_job = session.get(DocumentProcessingJob, failed_job_id)
        assert failed_job is not None
        assert failed_job.status == DocumentProcessingJobStatus.FAILED.value
        assert failed_job.attempts == (
            document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
        )
    failed_document = get_document(session_factory, failed_document_id)
    assert failed_document.processing_status == DocumentStatus.FAILED.value
    assert failed_document.processing_error_code == "processing_failed"
    assert get_segments(session_factory, failed_document_id) == []
    assert [request.audio_path.name for request in transcriber.requests] == [
        failed_storage_key,
        successful_storage_key,
        failed_storage_key,
        failed_storage_key,
    ]


def test_failed_mp4_job_retries_without_blocking_next_queued_job(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    failed_document_id, failed_job_id, failed_storage_key = (
        create_uploaded_audio_document(
            session_factory,
            upload_root,
            DocumentFileType.MP4,
            "fails",
        )
    )
    successful_document_id, successful_job_id, _successful_storage_key = (
        create_uploaded_audio_document(
            session_factory,
            upload_root,
            DocumentFileType.WAV,
            "succeeds",
        )
    )
    (upload_root / failed_storage_key).write_bytes(b"fake MP4 video")

    with session_factory() as session:
        with session.begin():
            failed_job = session.get(DocumentProcessingJob, failed_job_id)
            successful_job = session.get(DocumentProcessingJob, successful_job_id)
            assert failed_job is not None
            assert successful_job is not None
            successful_job.created_at = datetime.now(UTC)
            failed_job.created_at = successful_job.created_at - timedelta(seconds=2)

    extracted_paths: list[Path] = []

    class FakeAudioExtractor:
        def extract(self, _input_video: Path, output_audio: Path) -> Path:
            extracted_paths.append(output_audio)
            output_audio.write_bytes(b"failed MP4 audio")
            return output_audio

    class FakeTranscriber:
        def __init__(self) -> None:
            self.requests: list[TranscriptionRequest] = []

        def transcribe(
            self,
            request: TranscriptionRequest,
        ) -> TranscriptionResult:
            self.requests.append(request)
            if request.audio_path.read_bytes() == b"failed MP4 audio":
                raise RuntimeError("controlled MP4 transcription failure")
            return TranscriptionResult(
                (TranscriptionSegment("Following WAV transcript.", 0.5, 1.5),)
            )

    audio_extractor = FakeAudioExtractor()
    transcriber = FakeTranscriber()

    def create_service(_document_id: UUID) -> DocumentIndexingService:
        return make_audio_indexing_service(
            session_factory,
            upload_root,
            transcriber,
            audio_extractor=audio_extractor,
        )[0]

    assert process_next_document_job(session_factory, create_service)
    with session_factory() as session:
        failed_job = session.get(DocumentProcessingJob, failed_job_id)
        assert failed_job is not None
        assert failed_job.status == DocumentProcessingJobStatus.QUEUED.value
        assert failed_job.attempts == 1
    assert get_document(session_factory, failed_document_id).processing_status == (
        DocumentStatus.FAILED.value
    )
    assert extracted_paths and not extracted_paths[-1].parent.exists()
    assert (upload_root / failed_storage_key).read_bytes() == b"fake MP4 video"

    assert process_next_document_job(session_factory, create_service)
    with session_factory() as session:
        successful_job = session.get(DocumentProcessingJob, successful_job_id)
        assert successful_job is not None
        assert successful_job.status == DocumentProcessingJobStatus.COMPLETED.value
        assert successful_job.attempts == 1
    assert [
        segment.text_content
        for segment in get_segments(session_factory, successful_document_id)
    ] == ["Following WAV transcript."]

    for _ in range(
        document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS - 1
    ):
        with session_factory() as session:
            with session.begin():
                failed_job = session.get(DocumentProcessingJob, failed_job_id)
                assert failed_job is not None
                failed_job.available_at = datetime.now(UTC)
        assert process_next_document_job(session_factory, create_service)
        assert not extracted_paths[-1].parent.exists()

    with session_factory() as session:
        failed_job = session.get(DocumentProcessingJob, failed_job_id)
        assert failed_job is not None
        assert failed_job.status == DocumentProcessingJobStatus.FAILED.value
        assert failed_job.attempts == (
            document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
        )
    assert get_document(session_factory, failed_document_id).processing_status == (
        DocumentStatus.FAILED.value
    )
    assert get_segments(session_factory, failed_document_id) == []
    assert len(extracted_paths) == (
        document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
    )
    assert all(not path.parent.exists() for path in extracted_paths)
    assert [request.file_type for request in transcriber.requests] == [
        DocumentFileType.WAV,
        DocumentFileType.WAV,
        DocumentFileType.WAV,
        DocumentFileType.WAV,
    ]


def test_transient_audio_failure_succeeds_on_eligible_retry_once(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    document_id, job_id, storage_key = create_uploaded_audio_document(
        session_factory,
        upload_root,
        DocumentFileType.WAV,
        "retry succeeds",
    )

    class FailOnceTranscriber:
        def __init__(self) -> None:
            self.requests: list[TranscriptionRequest] = []

        def transcribe(
            self,
            request: TranscriptionRequest,
        ) -> TranscriptionResult:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise RuntimeError("controlled transient transcription failure")
            return TranscriptionResult(
                (TranscriptionSegment("Retry transcript.", 0.5, 1.5),)
            )

    transcriber = FailOnceTranscriber()
    indexing_services: list[
        tuple[DocumentIndexingService, FakeEmbeddingService, FakeVectorStore]
    ] = []

    def create_service(
        _document_id: UUID,
    ) -> DocumentIndexingService:
        service_parts = make_audio_indexing_service(
            session_factory,
            upload_root,
            transcriber,
        )
        indexing_services.append(service_parts)
        return service_parts[0]

    assert process_next_document_job(session_factory, create_service)
    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.QUEUED.value
        assert job.attempts == 1
    failed_document = get_document(session_factory, document_id)
    assert failed_document.processing_status == DocumentStatus.FAILED.value
    assert failed_document.processing_error_code == "processing_failed"
    assert get_segments(session_factory, document_id) == []

    with session_factory() as session:
        with session.begin():
            job = session.get(DocumentProcessingJob, job_id)
            assert job is not None
            job.available_at = datetime.now(UTC)

    assert process_next_document_job(session_factory, create_service)
    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.COMPLETED.value
        assert job.attempts == 2
    completed_document = get_document(session_factory, document_id)
    assert completed_document.processing_status == DocumentStatus.COMPLETED.value
    assert completed_document.processing_error_code is None

    segments = get_segments(session_factory, document_id)
    assert len(segments) == 1
    segment_id = segments[0].id
    assert segments[0].text_content == "Retry transcript."
    assert (segments[0].start_seconds, segments[0].end_seconds) == (0.5, 1.5)
    assert [request.audio_path.name for request in transcriber.requests] == [
        storage_key,
        storage_key,
    ]

    assert len(indexing_services) == 2
    assert indexing_services[0][1].calls == []
    assert indexing_services[0][2].calls == []
    successful_embedding = indexing_services[1][1]
    successful_vector_store = indexing_services[1][2]
    assert successful_embedding.calls == [("Retry transcript.",)]
    assert len(successful_vector_store.calls) == 1
    assert successful_vector_store.calls[0][0].metadata.segment_id == segment_id
    assert successful_embedding.closed is True
    assert successful_vector_store.closed is True

    assert not process_next_document_job(session_factory, create_service)
    assert len(transcriber.requests) == 2
    assert len(indexing_services) == 2


def test_lock_loss_during_failed_embedding_skips_failure_and_retry_writes(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id

    lock_lost = False

    def ensure_worker_lock() -> None:
        if lock_lost:
            raise DocumentWorkerLockLostError("Worker lock was lost.")

    def lose_lock_during_embedding() -> None:
        nonlocal lock_lost
        lock_lost = True

    embedding = FakeEmbeddingService(
        failure=EmbeddingProviderError("provider operation failed"),
        before_embed=lose_lock_during_embedding,
    )
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        session_factory,
        FakeProcessor(),
        lambda: DocumentVectorStorageService(
            DocumentEmbeddingService(embedding),
            vector_store,
        ),
        ensure_worker_lock=ensure_worker_lock,
    )

    with pytest.raises(DocumentWorkerLockLostError, match="Worker lock was lost"):
        process_next_document_job(
            session_factory,
            lambda _document_id: service,
            ensure_worker_lock=ensure_worker_lock,
        )

    document = get_document(session_factory, document_id)
    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None

    assert document.processing_status == DocumentStatus.PROCESSING.value
    assert document.processing_error_code is None
    assert job.status == DocumentProcessingJobStatus.PROCESSING.value
    assert job.attempts == 1
    assert embedding.calls == [("First persisted section.", "Second persisted section.")]
    assert vector_store.calls == []
    assert embedding.closed is True
    assert vector_store.closed is True


def test_worker_startup_queues_legacy_uploaded_documents_only_once(
    session_factory: sessionmaker[Session],
) -> None:
    queued_document_id = create_uploaded_document(session_factory)
    unqueued_document_id = create_uploaded_document(session_factory)

    with session_factory() as session:
        with session.begin():
            document_processing_job_repository.enqueue_document_processing(
                session,
                queued_document_id,
            )
    with session_factory() as session:
        with session.begin():
            queued_count = (
                document_processing_job_repository
                .enqueue_unqueued_uploaded_documents(session)
            )
    with session_factory() as session:
        jobs = list(session.scalars(select(DocumentProcessingJob)))

    assert queued_count == 1
    assert {job.document_id for job in jobs} == {
        queued_document_id,
        unqueued_document_id,
    }


def test_worker_failure_queues_retry_and_keeps_document_failed(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id
    service, _, _ = make_indexing_service(
        session_factory,
        vector_failure=VectorStoreError("provider details"),
    )

    assert process_next_document_job(
        session_factory,
        lambda _document_id: service,
    )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.QUEUED.value
        assert job.attempts == 1
    document = get_document(session_factory, document_id)
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"


def test_worker_stops_retrying_after_maximum_attempts(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job_id = job.id
    service, _, _ = make_indexing_service(
        session_factory,
        vector_failure=VectorStoreError("provider details"),
    )

    for _ in range(
        document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
    ):
        assert process_next_document_job(
            session_factory,
            lambda _document_id: service,
        )
        with session_factory() as session:
            with session.begin():
                job = session.get(DocumentProcessingJob, job_id)
                assert job is not None
                job.available_at = datetime.now(UTC)

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.FAILED.value
        assert job.attempts == (
            document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
        )


def test_interrupted_job_recovery_requeues_work_and_fails_stale_document(
    session_factory: sessionmaker[Session],
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            document_service_result = session.get(Document, document_id)
            assert document_service_result is not None
            document_service_result.processing_status = DocumentStatus.PROCESSING.value
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job.status = DocumentProcessingJobStatus.PROCESSING.value
            job.attempts = 1
            job_id = job.id

    with session_factory() as session:
        with session.begin():
            recovered = (
                document_processing_job_repository
                .recover_interrupted_document_processing_jobs(session)
            )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.QUEUED.value
    document = get_document(session_factory, document_id)
    assert recovered == 1
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"


@pytest.mark.parametrize(
    "document_status",
    [DocumentStatus.UPLOADED, DocumentStatus.PROCESSING],
)
def test_exhausted_interrupted_job_fails_job_and_document_safely(
    session_factory: sessionmaker[Session],
    document_status: DocumentStatus,
) -> None:
    document_id = create_uploaded_document(session_factory)
    with session_factory() as session:
        with session.begin():
            document = session.get(Document, document_id)
            assert document is not None
            document.processing_status = document_status.value
            job = document_processing_job_repository.enqueue_document_processing(
                session,
                document_id,
            )
            job.status = DocumentProcessingJobStatus.PROCESSING.value
            job.attempts = (
                document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
            )
            job_id = job.id

    with session_factory() as session:
        with session.begin():
            recovered = (
                document_processing_job_repository
                .recover_interrupted_document_processing_jobs(session)
            )

    with session_factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        assert job is not None
        assert job.status == DocumentProcessingJobStatus.FAILED.value
        assert job.attempts == (
            document_processing_job_repository.MAX_DOCUMENT_PROCESSING_ATTEMPTS
        )
    document = get_document(session_factory, document_id)
    assert recovered == 1
    assert document.processing_status == DocumentStatus.FAILED.value
    assert document.processing_error_code == "processing_failed"
