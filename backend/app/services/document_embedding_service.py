from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories import document_segment_repository
from app.services import document_service
from app.services.embedding_service import (
    EmbeddingBatch,
    EmbeddingInputError,
    EmbeddingOutputError,
)


class TextEmbeddingService(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Return one validated vector per input text, preserving order."""
        ...


class DocumentEmbeddingNotFoundError(LookupError):
    """Raised when the requested document does not exist."""


@dataclass(frozen=True, slots=True)
class EmbeddedDocumentSegment:
    document_id: UUID
    segment_id: UUID
    ordinal: int
    vector: tuple[float, ...]


class DocumentEmbeddingService:
    """Generate in-memory embeddings for all persisted segments of a document."""

    def __init__(self, embedding_service: TextEmbeddingService) -> None:
        self._embedding_service = embedding_service

    def embed_document(
        self,
        session: Session,
        document_id: UUID,
    ) -> tuple[EmbeddedDocumentSegment, ...]:
        """Embed ordered segments without persisting vectors or changing status."""
        if document_service.get_document(session, document_id) is None:
            raise DocumentEmbeddingNotFoundError("Document not found.")

        segments = document_segment_repository.list_document_segments(
            session,
            document_id,
        )
        if not segments:
            raise EmbeddingInputError(
                "The document has no persisted segments to embed."
            )

        for segment in segments:
            if (
                not isinstance(segment.id, UUID)
                or not isinstance(segment.document_id, UUID)
                or segment.document_id != document_id
                or isinstance(segment.ordinal, bool)
                or not isinstance(segment.ordinal, int)
                or segment.ordinal < 0
                or not isinstance(segment.text_content, str)
                or not segment.text_content.strip()
                or not _has_valid_provenance(segment)
            ):
                raise EmbeddingInputError(
                    "The document contains malformed segment data."
                )

        ordered_segments = sorted(segments, key=lambda segment: segment.ordinal)
        if (
            len({segment.ordinal for segment in ordered_segments})
            != len(ordered_segments)
            or len({segment.id for segment in ordered_segments})
            != len(ordered_segments)
        ):
            raise EmbeddingInputError(
                "The document contains malformed segment data."
            )

        texts = [segment.text_content for segment in ordered_segments]
        vectors = self._embedding_service.embed_texts(texts)
        self._validate_embedding_batch(vectors, len(ordered_segments))

        return tuple(
            EmbeddedDocumentSegment(
                document_id=document_id,
                segment_id=segment.id,
                ordinal=segment.ordinal,
                vector=vector,
            )
            for segment, vector in zip(ordered_segments, vectors, strict=True)
        )

    @staticmethod
    def _validate_embedding_batch(
        vectors: EmbeddingBatch,
        expected_count: int,
    ) -> None:
        if not isinstance(vectors, (tuple, list)) or len(vectors) != expected_count:
            raise EmbeddingOutputError(
                "The embedding service returned an unexpected vector count."
            )

        dimensions: int | None = None
        for vector in vectors:
            if not isinstance(vector, (tuple, list)) or not vector:
                raise EmbeddingOutputError(
                    "The embedding service returned an invalid vector."
                )
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise EmbeddingOutputError(
                    "The embedding service returned inconsistent vector dimensions."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                for value in vector
            ):
                raise EmbeddingOutputError(
                    "The embedding service returned an invalid vector value."
                )


def _has_valid_provenance(segment: object) -> bool:
    page_number = getattr(segment, "page_number", None)
    if page_number is not None and (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number < 1
    ):
        return False

    start_seconds = getattr(segment, "start_seconds", None)
    end_seconds = getattr(segment, "end_seconds", None)
    if (start_seconds is None) != (end_seconds is None):
        return False
    if start_seconds is None:
        return True
    if (
        isinstance(start_seconds, bool)
        or not isinstance(start_seconds, Real)
        or isinstance(end_seconds, bool)
        or not isinstance(end_seconds, Real)
        or not isfinite(start_seconds)
        or not isfinite(end_seconds)
        or start_seconds < 0
        or end_seconds < start_seconds
    ):
        return False
    return True
