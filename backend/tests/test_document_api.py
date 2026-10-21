from collections.abc import Generator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.application import create_app
from app.core.config import Settings
from app.db.base import Base
from app.db.session import get_db
from app.models.document import Document


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
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
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        app.state.transaction_events = transaction_events
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
