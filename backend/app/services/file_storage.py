import codecs
import logging
import os
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.models.document import DocumentFileType

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024
STORAGE_KEY_PATTERN = re.compile(r"^[0-9a-f]{32}\.(pdf|txt|mp3|wav|mp4)$")


class InvalidUploadError(ValueError):
    pass


class UploadTooLargeError(InvalidUploadError):
    pass


class FileStorageError(Exception):
    pass


class UploadReadError(Exception):
    pass


def storage_path_for_key(storage_key: str, upload_root: Path) -> Path:
    if not STORAGE_KEY_PATTERN.fullmatch(storage_key):
        raise FileStorageError("Invalid internal storage key.")

    root = upload_root.resolve()
    path = (root / storage_key).resolve()
    if path.parent != root:
        raise FileStorageError("Storage key is outside the upload root.")
    return path


def _validate_content(
    file_type: DocumentFileType,
    prefix: bytes,
    decoder: codecs.IncrementalDecoder | None,
) -> None:
    valid = {
        DocumentFileType.PDF: prefix.startswith(b"%PDF-"),
        DocumentFileType.TXT: decoder is not None,
        DocumentFileType.MP3: prefix.startswith(b"ID3")
        or (
            len(prefix) >= 2
            and prefix[0] == 0xFF
            and prefix[1] & 0xE0 == 0xE0
        ),
        DocumentFileType.WAV: len(prefix) >= 12
        and prefix[:4] == b"RIFF"
        and prefix[8:12] == b"WAVE",
        DocumentFileType.MP4: len(prefix) >= 8 and prefix[4:8] == b"ftyp",
    }[file_type]
    if not valid:
        raise InvalidUploadError("File content does not match its extension.")


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Unable to remove an incomplete uploaded file.")


async def store_upload(
    upload: UploadFile,
    *,
    file_type: DocumentFileType,
    upload_root: Path,
    max_size_bytes: int,
) -> str:
    """Stream an upload to a unique file and return its internal storage key."""
    path: Path | None = None
    descriptor: int | None = None
    created = False

    def cleanup_partial_file() -> None:
        nonlocal descriptor
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                logger.warning("Unable to close an incomplete uploaded file.")
            descriptor = None
        if created and path is not None:
            _remove_partial_file(path)

    try:
        upload_root.mkdir(parents=True, exist_ok=True)
        storage_key = f"{uuid4().hex}.{file_type.value}"
        path = storage_path_for_key(storage_key, upload_root)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
        decoder = (
            codecs.getincrementaldecoder("utf-8")("strict")
            if file_type is DocumentFileType.TXT
            else None
        )
        total_bytes = 0
        prefix = bytearray()

        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            while True:
                try:
                    chunk = await upload.read(CHUNK_SIZE)
                except Exception as error:
                    raise UploadReadError from error
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    raise UploadTooLargeError("File exceeds the configured size limit.")
                if file_type is DocumentFileType.TXT and b"\0" in chunk:
                    raise InvalidUploadError("File contains invalid binary content.")
                if len(prefix) < 12:
                    prefix.extend(chunk[: 12 - len(prefix)])
                if decoder is not None:
                    try:
                        decoder.decode(chunk, final=False)
                    except UnicodeDecodeError as error:
                        raise InvalidUploadError(
                            "Text file must contain valid UTF-8."
                        ) from error
                destination.write(chunk)

            if total_bytes == 0:
                raise InvalidUploadError("Uploaded file is empty.")
            if decoder is not None:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError as error:
                    raise InvalidUploadError(
                        "Text file must contain valid UTF-8."
                    ) from error
            _validate_content(file_type, bytes(prefix), decoder)
            destination.flush()

        return storage_key
    except (InvalidUploadError,):
        cleanup_partial_file()
        raise
    except UploadReadError as error:
        cleanup_partial_file()
        raise FileStorageError("Unable to read uploaded file.") from error
    except FileStorageError:
        cleanup_partial_file()
        raise
    except OSError as error:
        cleanup_partial_file()
        raise FileStorageError("Unable to write uploaded file.") from error
    except Exception:
        cleanup_partial_file()
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def delete_stored_file(storage_key: str, upload_root: Path) -> None:
    path = storage_path_for_key(storage_key, upload_root)
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise FileStorageError("Unable to remove uploaded file.") from error
