from collections.abc import Generator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.application import create_app
from app.core.config import Settings
from app.db.base import Base
from app.db.session import get_db
from app.models.document import Document
from app.models.document_segment import DocumentSegment


@pytest.fixture
def client(tmp_path) -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        with session_factory() as session:
            event.listen(
                session,
                "after_commit",
                lambda _: transaction_events.append("commit"),
            )
            event.listen(
                session,
                "after_rollback",
                lambda _: transaction_events.append("rollback"),
            )
            yield session

    transaction_events: list[str] = []
    failing_statements: set[str] = set()

    def fail_selected_statement(
        connection,
        cursor,
        statement: str,
        parameters,
        context,
        executemany: bool,
    ) -> None:
        sql_operation = statement.lstrip().split(maxsplit=1)[0].upper()
        if sql_operation in failing_statements:
            raise OperationalError(
                statement,
                parameters,
                Exception("synthetic database detail"),
            )

    event.listen(engine, "before_cursor_execute", fail_selected_statement)
    settings = Settings(
        upload_root=tmp_path / "uploads",
        max_upload_size_bytes=128,
        _env_file=None,
    )
    app = create_app(settings)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as test_client:
        app.state.transaction_events = transaction_events
        app.state.failing_statements = failing_statements
        app.state.session_factory = session_factory
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_create_document_returns_safe_metadata(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        json={"original_filename": "lecture.pdf", "file_type": "pdf"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "lecture.pdf"
    assert body["file_type"] == "pdf"
    assert body["processing_status"] == "pending"
    assert body["processing_error_code"] is None
    assert "storage_path" not in body
    assert UUID(body["id"])
    assert client.app.state.transaction_events == ["commit"]


def test_create_does_not_accept_client_storage_paths(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        json={
            "original_filename": "lecture.pdf",
            "file_type": "pdf",
            "storage_path": "C:\\private\\arbitrary.pdf",
        },
    )

    assert response.status_code == 422


def test_create_preserves_legitimate_unicode_filename(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        json={"original_filename": "講義ノート.pdf", "file_type": "pdf"},
    )

    assert response.status_code == 201
    assert response.json()["original_filename"] == "講義ノート.pdf"


def test_upload_pdf_stores_content_and_safe_metadata(client: TestClient) -> None:
    content = b"%PDF-1.7\nrealistic small PDF fixture"

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("lecture.pdf", content, "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "lecture.pdf"
    assert body["file_type"] == "pdf"
    assert body["processing_status"] == "uploaded"
    assert body["processing_error_code"] is None
    assert "storage_path" not in body
    files = list(client.app.state.settings.upload_root.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(".pdf")
    assert "lecture" not in files[0].name
    assert files[0].read_bytes() == content
    with client.app.state.session_factory() as session:
        document = session.get(Document, UUID(body["id"]))
    assert document is not None
    assert document.storage_path == files[0].name


def test_upload_txt_preserves_unicode_filename_and_content(
    client: TestClient,
) -> None:
    content = "Cognify 読み物\n".encode("utf-8")

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("講義ノート.txt", content, "application/octet-stream")},
    )

    assert response.status_code == 201
    assert response.json()["original_filename"] == "講義ノート.txt"
    stored_file = next(client.app.state.settings.upload_root.iterdir())
    assert stored_file.name.endswith(".txt")
    assert stored_file.read_bytes() == content


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("clip.mp3", "audio/mpeg", b"ID3\x04\x00\x00audio"),
        ("clip.wav", "audio/wav", b"RIFF\x04\x00\x00\x00WAVEdata"),
        ("clip.mp4", "video/mp4", b"\x00\x00\x00\x18ftypisomdata"),
    ],
)
def test_upload_accepts_supported_media_signatures(
    client: TestClient,
    filename: str,
    content_type: str,
    content: bytes,
) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, content, content_type)},
    )

    assert response.status_code == 201
    assert response.json()["file_type"] == filename.rsplit(".", 1)[1]


def test_upload_preserves_unicode_pdf_filename(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("講義ノート.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["original_filename"] == "講義ノート.pdf"


@pytest.mark.parametrize(
    "filename",
    ["folder/lecture.pdf", "folder\\lecture.pdf", "C:lecture.pdf", ""],
)
def test_upload_rejects_invalid_or_missing_filename(
    client: TestClient,
    filename: str,
) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, b"%PDF-1.7\ncontent", "application/pdf")},
    )

    assert response.status_code == 422
    assert list(client.app.state.settings.upload_root.glob("*")) == []


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("lecture.docx", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_upload_rejects_missing_file(client: TestClient) -> None:
    response = client.post("/api/v1/documents/upload")

    assert response.status_code == 422


def test_upload_rejects_more_than_one_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files=[
            ("file", ("one.pdf", b"%PDF-1.7\none", "application/pdf")),
            ("file", ("two.pdf", b"%PDF-1.7\ntwo", "application/pdf")),
        ],
    )

    assert response.status_code == 422
    assert list(client.app.state.settings.upload_root.glob("*")) == []


def test_upload_rejects_unexpected_multipart_fields(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("one.pdf", b"%PDF-1.7\none", "application/pdf")},
        data={"destination": "outside"},
    )

    assert response.status_code == 422
    assert list(client.app.state.settings.upload_root.glob("*")) == []


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 422
    assert list(client.app.state.settings.upload_root.glob("*")) == []


def test_upload_enforces_configured_maximum_size(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.pdf", b"%PDF-" + b"x" * 129, "application/pdf")},
    )

    assert response.status_code == 413
    assert list(client.app.state.settings.upload_root.glob("*")) == []


def test_upload_rejects_mismatched_file_signature(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("lecture.pdf", b"not a PDF", "application/pdf")},
    )

    assert response.status_code == 422
    assert list(client.app.state.settings.upload_root.glob("*")) == []


def test_upload_storage_failure_returns_safe_error(client: TestClient, tmp_path) -> None:
    upload_root = tmp_path / "not-a-directory"
    upload_root.write_text("blocking file", encoding="utf-8")
    client.app.state.settings.upload_root = upload_root

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("lecture.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )

    assert response.status_code == 500
    assert "blocking file" not in response.text
    assert "not-a-directory" not in response.text


def test_database_failure_removes_newly_stored_file(client: TestClient) -> None:
    client.app.state.failing_statements.add("INSERT")

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("lecture.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )

    assert response.status_code == 500
    assert "database" not in response.text.lower()
    assert client.app.state.transaction_events == ["rollback"]
    assert list(client.app.state.settings.upload_root.glob("*")) == []
    client.app.state.failing_statements.clear()


@pytest.mark.parametrize(
    "filename",
    [
        "folder/lecture.pdf",
        "folder\\lecture.pdf",
        "   ",
    ],
)
def test_create_rejects_path_like_or_whitespace_filename(
    client: TestClient,
    filename: str,
) -> None:
    response = client.post(
        "/api/v1/documents",
        json={"original_filename": filename, "file_type": "pdf"},
    )

    assert response.status_code == 422


def test_status_update_rejects_unexpected_storage_path(client: TestClient) -> None:
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "lecture.pdf", "file_type": "pdf"},
    )
    document_id = created.json()["id"]

    response = client.patch(
        f"/api/v1/documents/{document_id}/status",
        json={"status": "processing", "storage_path": "C:\\private\\file.pdf"},
    )

    assert response.status_code == 422
    assert client.get(f"/api/v1/documents/{document_id}").json()[
        "processing_status"
    ] == "pending"


def test_list_and_get_documents(client: TestClient) -> None:
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "lecture.txt", "file_type": "txt"},
    )
    document_id = created.json()["id"]

    listed = client.get("/api/v1/documents")
    retrieved = client.get(f"/api/v1/documents/{document_id}")

    assert listed.status_code == 200
    assert [document["id"] for document in listed.json()] == [document_id]
    assert all("storage_path" not in document for document in listed.json())
    assert retrieved.status_code == 200
    assert retrieved.json() == listed.json()[0]
    assert "storage_path" not in retrieved.json()


def test_get_document_segments_returns_segments_in_ordinal_order(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "lesson.pdf", "file_type": "pdf"},
    )
    document_id = UUID(created.json()["id"])
    with client.app.state.session_factory() as session:
        with session.begin():
            session.add_all(
                [
                    DocumentSegment(
                        document_id=document_id,
                        ordinal=2,
                        text_content="Third page",
                        page_number=3,
                    ),
                    DocumentSegment(
                        document_id=document_id,
                        ordinal=0,
                        text_content="First page",
                        page_number=1,
                    ),
                    DocumentSegment(
                        document_id=document_id,
                        ordinal=1,
                        text_content="Second page",
                        page_number=2,
                    ),
                ]
            )

    response = client.get(f"/api/v1/documents/{document_id}/segments")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert [segment["ordinal"] for segment in body["segments"]] == [0, 1, 2]
    assert [segment["text_content"] for segment in body["segments"]] == [
        "First page",
        "Second page",
        "Third page",
    ]
    assert [segment["page_number"] for segment in body["segments"]] == [1, 2, 3]
    assert all(
        segment["document_id"] == str(document_id)
        for segment in body["segments"]
    )
    assert all(UUID(segment["id"]) for segment in body["segments"])
    assert all(segment["start_seconds"] is None for segment in body["segments"])
    assert all(segment["end_seconds"] is None for segment in body["segments"])


def test_get_document_segments_paginates_but_reports_total_count(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "lesson.txt", "file_type": "txt"},
    )
    document_id = UUID(created.json()["id"])
    with client.app.state.session_factory() as session:
        with session.begin():
            session.add_all(
                [
                    DocumentSegment(
                        document_id=document_id,
                        ordinal=ordinal,
                        text_content=f"Segment {ordinal}",
                    )
                    for ordinal in range(4)
                ]
            )

    response = client.get(
        f"/api/v1/documents/{document_id}/segments",
        params={"limit": 2, "offset": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert [segment["ordinal"] for segment in body["segments"]] == [1, 2]


def test_get_segments_for_existing_document_without_segments_returns_empty_page(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "empty.txt", "file_type": "txt"},
    )

    response = client.get(f"/api/v1/documents/{created.json()['id']}/segments")

    assert response.status_code == 200
    assert response.json() == {
        "segments": [],
        "total": 0,
        "limit": 50,
        "offset": 0,
    }


def test_get_segments_for_missing_document_returns_404(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/documents/00000000-0000-0000-0000-000000000001/segments"
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("limit", 0), ("limit", 101), ("offset", -1)],
)
def test_get_document_segments_rejects_invalid_pagination(
    client: TestClient,
    parameter: str,
    value: int,
) -> None:
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "lesson.txt", "file_type": "txt"},
    )

    response = client.get(
        f"/api/v1/documents/{created.json()['id']}/segments",
        params={parameter: value},
    )

    assert response.status_code == 422


def test_update_status_and_delete_document(client: TestClient) -> None:
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "lecture.mp3", "file_type": "mp3"},
    )
    document_id = created.json()["id"]

    updated = client.patch(
        f"/api/v1/documents/{document_id}/status",
        json={"status": "processing"},
    )

    assert updated.status_code == 200
    assert updated.json()["processing_status"] == "processing"
    assert "storage_path" not in updated.json()
    assert client.app.state.transaction_events == ["commit", "commit"]

    deleted = client.delete(f"/api/v1/documents/{document_id}")

    assert deleted.status_code == 204
    assert client.app.state.transaction_events == ["commit", "commit", "commit"]
    assert client.get(f"/api/v1/documents/{document_id}").status_code == 404


def test_upload_lifecycle_transitions_persist_safe_failure_code(
    client: TestClient,
) -> None:
    uploaded = client.post(
        "/api/v1/documents/upload",
        files={"file": ("lesson.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    document_id = uploaded.json()["id"]
    assert uploaded.json()["processing_status"] == "uploaded"

    started = client.patch(
        f"/api/v1/documents/{document_id}/status",
        json={"status": "processing"},
    )
    assert started.status_code == 200
    assert started.json()["processing_status"] == "processing"
    assert started.json()["processing_error_code"] is None

    failed = client.patch(
        f"/api/v1/documents/{document_id}/status",
        json={"status": "failed"},
    )
    assert failed.status_code == 200
    assert failed.json()["processing_status"] == "failed"
    assert failed.json()["processing_error_code"] == "processing_failed"
    assert "storage_path" not in failed.json()
    assert "exception" not in failed.text.lower()

    persisted_failure = client.get(f"/api/v1/documents/{document_id}")
    assert persisted_failure.json()["processing_error_code"] == "processing_failed"

    retried = client.patch(
        f"/api/v1/documents/{document_id}/status",
        json={"status": "processing"},
    )
    assert retried.status_code == 200
    assert retried.json()["processing_error_code"] is None

    completed = client.patch(
        f"/api/v1/documents/{document_id}/status",
        json={"status": "completed"},
    )
    assert completed.status_code == 200
    assert completed.json()["processing_status"] == "completed"
    assert completed.json()["processing_error_code"] is None


@pytest.mark.parametrize(
    ("initial_status", "requested_status"),
    [
        ("uploaded", "completed"),
        ("uploaded", "failed"),
        ("processing", "processing"),
        ("completed", "processing"),
    ],
)
def test_invalid_status_transitions_return_conflict(
    client: TestClient,
    initial_status: str,
    requested_status: str,
) -> None:
    created = client.post(
        "/api/v1/documents/upload",
        files={"file": ("lesson.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    document_id = created.json()["id"]
    if initial_status in ("processing", "completed"):
        started = client.patch(
            f"/api/v1/documents/{document_id}/status",
            json={"status": "processing"},
        )
        assert started.status_code == 200
    if initial_status == "completed":
        finished = client.patch(
            f"/api/v1/documents/{document_id}/status",
            json={"status": "completed"},
        )
        assert finished.status_code == 200

    before = client.get(f"/api/v1/documents/{document_id}").json()
    response = client.patch(
        f"/api/v1/documents/{document_id}/status",
        json={"status": requested_status},
    )

    assert response.status_code == 409
    after = client.get(f"/api/v1/documents/{document_id}").json()
    assert after["processing_status"] == before["processing_status"]
    assert after["processing_error_code"] == before["processing_error_code"]


def test_invalid_file_type_and_status_return_422(client: TestClient) -> None:
    invalid_file = client.post(
        "/api/v1/documents",
        json={"original_filename": "lecture.docx", "file_type": "docx"},
    )
    created = client.post(
        "/api/v1/documents",
        json={"original_filename": "lecture.pdf", "file_type": "pdf"},
    )
    invalid_status = client.patch(
        f"/api/v1/documents/{created.json()['id']}/status",
        json={"status": "archived"},
    )

    assert invalid_file.status_code == 422
    assert invalid_status.status_code == 422
    assert client.get("/api/v1/documents").json()[0]["processing_status"] == "pending"


def test_malformed_uuid_returns_422(client: TestClient) -> None:
    response = client.get("/api/v1/documents/not-a-uuid")

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/v1/documents/00000000-0000-0000-0000-000000000001", None),
        (
            "patch",
            "/api/v1/documents/00000000-0000-0000-0000-000000000001/status",
            {"status": "completed"},
        ),
        ("delete", "/api/v1/documents/00000000-0000-0000-0000-000000000001", None),
    ],
)
def test_missing_documents_return_404(
    client: TestClient,
    method: str,
    path: str,
    body: dict[str, str] | None,
) -> None:
    response = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)

    assert response.status_code == 404


def test_invalid_request_leaves_no_document_write(client: TestClient) -> None:
    invalid = client.post(
        "/api/v1/documents",
        json={"original_filename": "lecture.docx", "file_type": "docx"},
    )

    assert invalid.status_code == 422
    assert client.app.state.transaction_events == []
    created = client.get("/api/v1/documents")
    assert created.status_code == 200
    assert created.json() == []


def test_missing_mutation_rolls_back_transaction(client: TestClient) -> None:
    response = client.patch(
        "/api/v1/documents/00000000-0000-0000-0000-000000000001/status",
        json={"status": "completed"},
    )

    assert response.status_code == 404
    assert client.app.state.transaction_events == ["rollback"]


@pytest.mark.parametrize(
    ("method", "path", "body", "operation"),
    [
        (
            "post",
            "/api/v1/documents",
            {"original_filename": "failed.pdf", "file_type": "pdf"},
            "INSERT",
        ),
        (
            "patch",
            "/api/v1/documents/00000000-0000-0000-0000-000000000001/status",
            {"status": "completed"},
            "UPDATE",
        ),
        (
            "delete",
            "/api/v1/documents/00000000-0000-0000-0000-000000000001",
            None,
            "DELETE",
        ),
    ],
)
def test_database_operation_failure_rolls_back_and_hides_details(
    client: TestClient,
    method: str,
    path: str,
    body: dict[str, str] | None,
    operation: str,
) -> None:
    if operation != "INSERT":
        created = client.post(
            "/api/v1/documents",
            json={"original_filename": "existing.pdf", "file_type": "pdf"},
        )
        document_id = created.json()["id"]
        path = path.replace(
            "00000000-0000-0000-0000-000000000001",
            document_id,
        )
        if operation == "UPDATE":
            started = client.patch(
                f"/api/v1/documents/{document_id}/status",
                json={"status": "processing"},
            )
            assert started.status_code == 200
        client.app.state.transaction_events.clear()

    client.app.state.failing_statements.add(operation)
    response = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    client.app.state.failing_statements.clear()

    assert response.status_code == 500
    assert "synthetic database detail" not in response.text
    assert client.app.state.transaction_events == ["rollback"]

    if operation == "INSERT":
        assert client.get("/api/v1/documents").json() == []
    elif operation == "UPDATE":
        assert (
            client.get(f"/api/v1/documents/{document_id}").json()[
                "processing_status"
            ]
            == "processing"
        )
        assert (
            client.get(f"/api/v1/documents/{document_id}").json()[
                "processing_error_code"
            ]
            is None
        )
    else:
        assert client.get(f"/api/v1/documents/{document_id}").status_code == 200
