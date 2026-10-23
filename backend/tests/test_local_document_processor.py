from collections.abc import Callable, Generator
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pypdf import PdfWriter
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document import (
    DocumentFileType,
    DocumentProcessingErrorCode,
    DocumentStatus,
)
from app.services import document_service
from app.services.document_processing_service import (
    DocumentProcessingFailedError,
    process_document,
)
from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessingInput,
)
from app.services.local_document_processor import LocalDocumentProcessor


@pytest.fixture
def upload_root(tmp_path: Path) -> Path:
    root = tmp_path / "uploads"
    root.mkdir()
    return root


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'extraction-tests.db'}",
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def make_input(
    storage_key: str,
    file_type: DocumentFileType,
) -> DocumentProcessingInput:
    return DocumentProcessingInput(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        original_filename="untrusted-original-name",
        file_type=file_type,
        storage_key=storage_key,
    )


def make_pdf(page_texts: tuple[str | None, ...]) -> bytes:
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids "
            f"[{' '.join(f'{4 + 2 * index} 0 R' for index in range(len(page_texts)))}] "
            f"/Count {len(page_texts)} >>"
        ).encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, page_text in enumerate(page_texts):
        page_id = 4 + 2 * index
        content_id = page_id + 1
        content = (
            b""
            if page_text is None
            else (
                b"BT /F1 12 Tf 72 720 Td ("
                + page_text.encode("ascii")
                + b") Tj ET"
            )
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id in range(1, len(objects) + 1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def write_upload(upload_root: Path, file_type: str, content: bytes) -> str:
    storage_key = f"{uuid4().hex}.{file_type}"
    (upload_root / storage_key).write_bytes(content)
    return storage_key


def expect_safe_failure(
    processor: LocalDocumentProcessor,
    processing_input: DocumentProcessingInput,
) -> None:
    with pytest.raises(DocumentProcessingError) as raised:
        processor.process(processing_input)

    assert raised.value.code is DocumentProcessingErrorCode.PROCESSING_FAILED
    assert str(raised.value) == "Document processing failed."


def test_txt_extraction_preserves_unicode_and_decodes_utf8_bom(
    upload_root: Path,
) -> None:
    expected_text = "Cognify 文書\nwith UTF-8."
    storage_key = write_upload(
        upload_root,
        "txt",
        b"\xef\xbb\xbf" + expected_text.encode("utf-8"),
    )

    result = LocalDocumentProcessor(upload_root).process(
        make_input(storage_key, DocumentFileType.TXT)
    )

    assert len(result.segments) == 1
    assert result.segments[0].text == expected_text
    assert result.segments[0].page_number is None


def test_pdf_extraction_preserves_1_based_page_numbers_and_skips_blank_pages(
    upload_root: Path,
) -> None:
    storage_key = write_upload(
        upload_root,
        "pdf",
        make_pdf(("First page text", None, "Third page text")),
    )

    result = LocalDocumentProcessor(upload_root).process(
        make_input(storage_key, DocumentFileType.PDF)
    )

    assert [(segment.page_number, segment.text) for segment in result.segments] == [
        (1, "First page text"),
        (3, "Third page text"),
    ]


@pytest.mark.parametrize(
    ("file_type", "content"),
    [
        ("txt", b" \r\n\t"),
        ("txt", b"\xff\xfe invalid UTF-8"),
        ("pdf", make_pdf((None, None))),
        ("pdf", b"%PDF-1.4 malformed"),
    ],
)
def test_empty_invalid_or_nonextractable_content_fails_safely(
    upload_root: Path,
    file_type: str,
    content: bytes,
) -> None:
    storage_key = write_upload(upload_root, file_type, content)
    document_type = DocumentFileType(file_type)

    expect_safe_failure(
        LocalDocumentProcessor(upload_root),
        make_input(storage_key, document_type),
    )


def test_missing_file_fails_without_exposing_storage_root(upload_root: Path) -> None:
    storage_key = f"{uuid4().hex}.txt"

    expect_safe_failure(
        LocalDocumentProcessor(upload_root),
        make_input(storage_key, DocumentFileType.TXT),
    )


def test_unsupported_file_type_fails_safely(upload_root: Path) -> None:
    expect_safe_failure(
        LocalDocumentProcessor(upload_root),
        make_input(f"{uuid4().hex}.mp3", DocumentFileType.MP3),
    )


def test_storage_key_cannot_escape_upload_root(
    upload_root: Path,
    tmp_path: Path,
) -> None:
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("must not be read", encoding="utf-8")

    expect_safe_failure(
        LocalDocumentProcessor(upload_root),
        make_input("../outside.txt", DocumentFileType.TXT),
    )


def test_orchestrator_sanitizes_extraction_failure(
    upload_root: Path,
    session_factory: sessionmaker[Session],
) -> None:
    malformed_pdf = b"%PDF-1.4 internal path and private bytes"
    storage_key = write_upload(upload_root, "pdf", malformed_pdf)
    with session_factory() as session:
        with session.begin():
            document = document_service.create_document(
                session,
                original_filename="lesson.pdf",
                file_type=DocumentFileType.PDF,
                storage_path=storage_key,
                processing_status=DocumentStatus.UPLOADED,
            )
            document_id = document.id

    with pytest.raises(DocumentProcessingFailedError) as raised:
        process_document(
            session_factory,
            document_id,
            LocalDocumentProcessor(upload_root),
        )

    assert str(raised.value) == "Document processing failed."
    with session_factory() as session:
        document = session.get(type(document), document_id)
        assert document is not None
        assert document.processing_status == DocumentStatus.FAILED.value
        assert document.processing_error_code == "processing_failed"


def test_generated_empty_pdf_has_no_extractable_text(upload_root: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    output = BytesIO()
    writer.write(output)
    storage_key = write_upload(upload_root, "pdf", output.getvalue())

    expect_safe_failure(
        LocalDocumentProcessor(upload_root),
        make_input(storage_key, DocumentFileType.PDF),
    )
