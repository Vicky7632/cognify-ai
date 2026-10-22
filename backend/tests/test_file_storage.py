import asyncio
import os
from pathlib import Path

import pytest

from app.models.document import DocumentFileType
from app.services import file_storage
from app.services.file_storage import (
    FileStorageError,
    InvalidUploadError,
    store_upload,
    storage_path_for_key,
)


class InterruptedUpload:
    async def read(self, size: int) -> bytes:
        if not hasattr(self, "sent_chunk"):
            self.sent_chunk = True
            return b"%PDF-1.7\n"
        raise OSError("synthetic read failure")


class FixedUpload:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.sent_chunk = False

    async def read(self, size: int) -> bytes:
        assert size == file_storage.CHUNK_SIZE
        if self.sent_chunk:
            return b""
        self.sent_chunk = True
        return self.content


def test_interrupted_upload_removes_partial_file(tmp_path: Path) -> None:
    with pytest.raises(FileStorageError, match="Unable to read uploaded file"):
        asyncio.run(
            store_upload(
                InterruptedUpload(),
                file_type=DocumentFileType.PDF,
                upload_root=tmp_path / "uploads",
                max_size_bytes=100,
            )
        )

    assert list((tmp_path / "uploads").iterdir()) == []


def test_invalid_utf8_text_is_rejected_and_removed(tmp_path: Path) -> None:
    with pytest.raises(InvalidUploadError, match="valid UTF-8"):
        asyncio.run(
            store_upload(
                FixedUpload(b"\xff\xfe"),
                file_type=DocumentFileType.TXT,
                upload_root=tmp_path / "uploads",
                max_size_bytes=100,
            )
        )

    assert list((tmp_path / "uploads").iterdir()) == []


def test_write_failure_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fdopen = os.fdopen

    class FailingWriter:
        def __init__(self, descriptor: int, mode: str) -> None:
            self.file = original_fdopen(descriptor, mode)

        def __enter__(self) -> "FailingWriter":
            return self

        def __exit__(self, *args) -> None:
            self.file.close()

        def write(self, content: bytes) -> None:
            self.file.write(content[:2])
            raise OSError("synthetic disk failure")

    monkeypatch.setattr(file_storage.os, "fdopen", FailingWriter)

    with pytest.raises(FileStorageError, match="Unable to write uploaded file"):
        asyncio.run(
            store_upload(
                FixedUpload(b"%PDF-1.7\ncontent"),
                file_type=DocumentFileType.PDF,
                upload_root=tmp_path / "uploads",
                max_size_bytes=100,
            )
        )

    assert list((tmp_path / "uploads").iterdir()) == []


def test_file_open_failure_closes_descriptor_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fdopen(_descriptor: int, _mode: str) -> None:
        raise OSError("synthetic file-open failure")

    monkeypatch.setattr(file_storage.os, "fdopen", fail_fdopen)

    with pytest.raises(FileStorageError, match="Unable to write uploaded file"):
        asyncio.run(
            store_upload(
                FixedUpload(b"%PDF-1.7\ncontent"),
                file_type=DocumentFileType.PDF,
                upload_root=tmp_path / "uploads",
                max_size_bytes=100,
            )
        )

    assert list((tmp_path / "uploads").iterdir()) == []


def test_storage_key_cannot_escape_upload_root(tmp_path: Path) -> None:
    with pytest.raises(FileStorageError, match="Invalid internal storage key"):
        storage_path_for_key("../outside.pdf", tmp_path)
