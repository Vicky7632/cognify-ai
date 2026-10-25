from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pinecone import Pinecone

from app.core.config import Settings
from app.services.document_embedding_service import EmbeddedDocumentSegment
from app.vector_store.protocol import (
    VectorMetadata,
    VectorMetadataValue,
    VectorRecord,
    VectorSearchResult,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreInputError,
    validate_query_vector,
    validate_vector_records,
)


class _PineconeIndex(Protocol):
    def upsert(
        self,
        *,
        vectors: Sequence[dict[str, object]],
        namespace: str = "",
        batch_size: int | None = None,
        show_progress: bool = True,
        max_concurrency: int = 8,
        timeout: float | None = None,
        total_timeout: float | None = None,
    ) -> object:
        ...

    def query(
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        filter: Mapping[str, object] | None = None,
        include_metadata: bool = False,
    ) -> object:
        ...


class _PineconeClient(Protocol):
    def describe_index(self, name: str) -> object:
        ...

    def Index(self, name: str = "", host: str = "", **kwargs: object) -> _PineconeIndex:
        ...

    def close(self) -> None:
        ...


class PineconeVectorStore:
    """Pinecone implementation of the provider-neutral vector store contract."""

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        expected_dimensions: int,
        batch_size: int = 100,
        client: _PineconeClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise VectorStoreConfigurationError(
                "Pinecone API key is not configured."
            )
        if not index_name.strip():
            raise VectorStoreConfigurationError(
                "Pinecone index name is not configured."
            )
        if (
            isinstance(expected_dimensions, bool)
            or not isinstance(expected_dimensions, int)
            or expected_dimensions < 1
        ):
            raise VectorStoreConfigurationError(
                "Configured vector dimensions must be positive."
            )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise VectorStoreConfigurationError(
                "Pinecone upsert batch size must be positive."
            )

        try:
            self._client = client if client is not None else Pinecone(api_key=api_key)
            index_description = self._client.describe_index(name=index_name)
            index_dimensions = getattr(index_description, "dimension", None)
            if not isinstance(index_dimensions, int) or isinstance(
                index_dimensions, bool
            ):
                raise VectorStoreConfigurationError(
                    "Pinecone index dimension could not be verified."
                )
            if index_dimensions != expected_dimensions:
                raise VectorStoreConfigurationError(
                    "Pinecone index dimension does not match the configured embedding "
                    f"dimension ({index_dimensions} != {expected_dimensions})."
                )
            self._index = self._client.Index(name=index_name)
        except VectorStoreConfigurationError:
            self._close_after_initialization_failure()
            raise
        except Exception:
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index could not be found or verified."
            ) from None

        self._expected_dimensions = expected_dimensions
        self._batch_size = batch_size

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Validate and upsert records in configured batches."""
        validate_vector_records(records, expected_dimensions=self._expected_dimensions)
        for start in range(0, len(records), self._batch_size):
            batch = records[start : start + self._batch_size]
            vectors = [
                {
                    "id": record.id,
                    "values": list(record.values),
                    "metadata": record.metadata.as_dict(),
                }
                for record in batch
            ]
            try:
                self._index.upsert(vectors=vectors, show_progress=False)
            except Exception:
                raise VectorStoreError("Pinecone vector upsert failed.") from None

    def search(
        self,
        query_vector: Sequence[Real],
        *,
        top_k: int,
        document_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search nearest vectors, optionally restricting matches to one document."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise VectorStoreInputError("top_k must be a positive integer.")
        if document_id is not None and not isinstance(document_id, UUID):
            raise VectorStoreInputError("Document filter must be a UUID.")
        vector = validate_query_vector(
            query_vector,
            expected_dimensions=self._expected_dimensions,
        )
        metadata_filter = (
            {"document_id": {"$eq": str(document_id)}}
            if document_id is not None
            else None
        )
        try:
            response = self._index.query(
                vector=vector,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True,
            )
        except Exception:
            raise VectorStoreError("Pinecone similarity search failed.") from None
        return _parse_search_response(response)

    def close(self) -> None:
        """Close the SDK client and its underlying connections."""
        self._client.close()

    def _close_after_initialization_failure(self) -> None:
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception:
            raise VectorStoreConfigurationError(
                "Pinecone client cleanup failed during initialization."
            ) from None


def stable_segment_vector_id(document_id: UUID, segment_id: UUID) -> str:
    """Return a deterministic vector ID for a document segment."""
    return str(uuid5(NAMESPACE_URL, f"cognify-ai:{document_id}:{segment_id}"))


def vector_record_for_segment(segment: EmbeddedDocumentSegment) -> VectorRecord:
    return VectorRecord(
        id=stable_segment_vector_id(segment.document_id, segment.segment_id),
        values=segment.vector,
        metadata=VectorMetadata(
            document_id=segment.document_id,
            segment_id=segment.segment_id,
            ordinal=segment.ordinal,
        ),
    )


def _parse_search_response(response: object) -> tuple[VectorSearchResult, ...]:
    try:
        matches = getattr(response, "matches", None)
        if not isinstance(matches, Sequence) or isinstance(matches, (str, bytes)):
            raise ValueError

        results: list[VectorSearchResult] = []
        for match in matches:
            vector_id = getattr(match, "id", None)
            score = getattr(match, "score", None)
            metadata = getattr(match, "metadata", None)
            if (
                not isinstance(vector_id, str)
                or not vector_id.strip()
                or isinstance(score, bool)
                or not isinstance(score, Real)
                or not isfinite(score)
                or not isinstance(metadata, Mapping)
            ):
                raise ValueError

            document_id = UUID(_required_metadata_string(metadata, "document_id"))
            segment_id = UUID(_required_metadata_string(metadata, "segment_id"))
            ordinal = metadata.get("ordinal")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or vector_id != stable_segment_vector_id(document_id, segment_id)
            ):
                raise ValueError
            normalized_metadata = _normalize_metadata(metadata)
            results.append(
                VectorSearchResult(
                    vector_id=vector_id,
                    document_id=document_id,
                    segment_id=segment_id,
                    ordinal=ordinal,
                    score=float(score),
                    metadata=MappingProxyType(normalized_metadata),
                )
            )
        return tuple(results)
    except Exception:
        raise VectorStoreError(
            "Pinecone returned a malformed similarity response."
        ) from None


def _required_metadata_string(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value


def _normalize_metadata(
    metadata: Mapping[str, object],
) -> dict[str, VectorMetadataValue]:
    normalized: dict[str, VectorMetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError
        if isinstance(value, (str, bool, int)):
            normalized[key] = value
        elif isinstance(value, float):
            if not isfinite(value):
                raise ValueError
            normalized[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, bool, int))
            or isinstance(item, float) and isfinite(item)
            for item in value
        ):
            normalized[key] = tuple(value)
        else:
            raise ValueError
    return normalized


def create_pinecone_vector_store(
    settings: Settings,
    *,
    client: _PineconeClient | None = None,
) -> PineconeVectorStore:
    """Construct and validate Pinecone configuration from application settings."""
    if settings.pinecone_api_key is None:
        raise VectorStoreConfigurationError("Pinecone API key is not configured.")
    if not settings.pinecone_index or not settings.pinecone_index.strip():
        raise VectorStoreConfigurationError("Pinecone index name is not configured.")

    return PineconeVectorStore(
        api_key=settings.pinecone_api_key.get_secret_value(),
        index_name=settings.pinecone_index,
        expected_dimensions=settings.embedding_dimensions,
        batch_size=settings.pinecone_upsert_batch_size,
        client=client,
    )
