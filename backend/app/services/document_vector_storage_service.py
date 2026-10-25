from uuid import UUID

from sqlalchemy.orm import Session

from app.services.document_embedding_service import (
    DocumentEmbeddingService,
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

    def embed_and_store(
        self,
        session: Session,
        document_id: UUID,
    ) -> tuple[EmbeddedDocumentSegment, ...]:
        """Return results only after all generated vectors are upserted."""
        embedded_segments = self._embedding_pipeline.embed_document(
            session,
            document_id,
        )
        self._vector_store.upsert(
            tuple(
                vector_record_for_segment(segment)
                for segment in embedded_segments
            )
        )
        return embedded_segments
