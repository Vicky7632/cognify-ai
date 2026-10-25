from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol
from uuid import UUID


class VectorStoreConfigurationError(RuntimeError):
    """Raised when vector storage cannot be configured safely."""


class VectorStoreInputError(ValueError):
    """Raised when a vector record is invalid."""


class VectorStoreError(RuntimeError):
    """Raised when the vector storage provider cannot complete an operation."""


@dataclass(frozen=True, slots=True)
class VectorMetadata:
    document_id: UUID
    segment_id: UUID
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, UUID) or not isinstance(
            self.segment_id, UUID
        ):
            raise VectorStoreInputError("Vector metadata IDs must be UUID values.")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise VectorStoreInputError("Vector metadata ordinal must be nonnegative.")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "document_id": str(self.document_id),
            "segment_id": str(self.segment_id),
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True, slots=True)
class VectorRecord:
    id: str
    values: tuple[float, ...]
    metadata: VectorMetadata


class VectorStore(Protocol):
    """Provider-neutral interface for upserting typed vector records."""

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Upsert records or raise an application-level provider error."""
        ...

    def close(self) -> None:
        """Release provider resources."""
        ...


def validate_vector_records(
    records: Sequence[VectorRecord],
    *,
    expected_dimensions: int,
) -> None:
    if not records:
        raise VectorStoreInputError("At least one vector record is required.")

    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, VectorRecord):
            raise VectorStoreInputError("Vector record is malformed.")
        if not isinstance(record.id, str) or not record.id.strip():
            raise VectorStoreInputError("Vector ID must not be empty.")
        if record.id in seen_ids:
            raise VectorStoreInputError("Vector IDs must be unique within an upsert.")
        seen_ids.add(record.id)
        if not isinstance(record.metadata, VectorMetadata):
            raise VectorStoreInputError("Vector metadata is malformed.")
        if not isinstance(record.values, (tuple, list)) or not record.values:
            raise VectorStoreInputError("Vector values must not be empty.")
        if len(record.values) != expected_dimensions:
            raise VectorStoreInputError(
                "Vector dimensions do not match the configured index dimension."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in record.values
        ):
            raise VectorStoreInputError(
                "Vector values must be finite numeric values."
            )
