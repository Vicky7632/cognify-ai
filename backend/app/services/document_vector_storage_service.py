from collections.abc import Sequence
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from app.services.document_embedding_service import (
    DocumentEmbeddingService,
    DocumentSegmentEmbeddingInput,
    EmbeddedDocumentSegment,
)
from app.vector_store.pinecone_adapter import vector_record_for_segment
from app.vector_store.protocol import VectorStore


class DocumentVectorStorageService:
    """Generate document embeddings and upsert them into an injected vector store."""

    def __init__(
        self,
        embedding_pipeline: DocumentEmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_pipeline = embedding_pipeline
        self._vector_store = vector_store

    def close(self) -> None:
        """Release embedding and vector-store provider resources."""
        try:
            self._embedding_pipeline.close()
        finally:
            self._vector_store.close()

    def embed_and_store(
        self,
        session: Session,
        document_id: UUID,
    ) -> tuple[EmbeddedDocumentSegment, ...]:
        """Return results only after all generated vectors are upserted."""
        segments = self.load_document_segments(session, document_id)
        return self.embed_and_store_segments(segments)

    def load_document_segments(
        self,
        session: Session,
        document_id: UUID,
    ) -> tuple[DocumentSegmentEmbeddingInput, ...]:
        """Load validated segment values for embedding after the session closes."""
        return self._embedding_pipeline.load_document_segments(session, document_id)

    def embed_and_store_segments(
        self,
        segments: Sequence[DocumentSegmentEmbeddingInput],
        *,
        before_upsert: Callable[[], None] | None = None,
    ) -> tuple[EmbeddedDocumentSegment, ...]:
        """Embed loaded segments and upsert without needing a database session."""
        embedded_segments = self._embedding_pipeline.embed_segments(segments)
        if before_upsert is not None:
            before_upsert()
        self._vector_store.upsert(
            tuple(
                vector_record_for_segment(segment)
                for segment in embedded_segments
            )
        )
        return embedded_segments
