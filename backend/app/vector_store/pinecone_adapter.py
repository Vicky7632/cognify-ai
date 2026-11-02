from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Protocol
from urllib.parse import urlsplit
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
        namespace: str = "",
        filter: Mapping[str, object] | None = None,
        include_metadata: bool = False,
    ) -> object:
        ...


class _PineconeIndexes(Protocol):
    def list(self) -> Iterable[object]:
        ...


class _PineconeClient(Protocol):
    @property
    def indexes(self) -> _PineconeIndexes:
        ...

    def describe_index(self, name: str) -> object:
        ...

    def Index(self, name: str = "", host: str = "", **kwargs: object) -> _PineconeIndex:
        ...

    def close(self) -> None:
        ...


class _PineconeDiagnosticError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        category: str,
    ) -> None:
        self.status_code = status_code
        self.category = category
        RuntimeError.__init__(self, message)


class _PineconeVectorStoreError(_PineconeDiagnosticError, VectorStoreError):
    pass


class _PineconeConfigurationError(
    _PineconeDiagnosticError,
    VectorStoreConfigurationError,
):
    pass


class PineconeVectorStore:
    """Pinecone implementation of the provider-neutral vector store contract."""

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        expected_dimensions: int,
        host: str | None = None,
        namespace: str = "cognify",
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
        if host is not None:
            _validate_pinecone_host(host)
        if not namespace.strip():
            raise VectorStoreConfigurationError(
                "Pinecone namespace must not be empty."
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
        except Exception as error:
            raise _configuration_error(
                error,
                _index_verification_error_message(error, "initializing the client"),
            ) from None

        if host is None:
            try:
                index_description = self._client.describe_index(name=index_name)
            except Exception as error:
                self._close_after_initialization_failure()
                raise _configuration_error(
                    error,
                    _index_verification_error_message(error, "describing the index"),
                ) from None
        else:
            index_description = self._find_index_by_name(index_name, host)

        self._validate_index_description(
            index_description,
            expected_dimensions=expected_dimensions,
            host_mode=host is not None,
        )

        try:
            if host is None:
                self._index = self._client.Index(name=index_name)
            else:
                self._index = self._client.Index(host=host)
        except Exception as error:
            self._close_after_initialization_failure()
            raise _configuration_error(
                error,
                _index_verification_error_message(error, "connecting to the index"),
            ) from None

        self._expected_dimensions = expected_dimensions
        self._namespace = namespace
        self._batch_size = batch_size

    def _find_index_by_name(self, index_name: str, host: str) -> object:
        try:
            indexes = tuple(self._client.indexes.list())
        except Exception as error:
            self._close_after_initialization_failure()
            raise _configuration_error(
                error,
                _index_verification_error_message(error, "listing indexes"),
            ) from None

        try:
            matching_indexes = tuple(
                index
                for index in indexes
                if getattr(index, "name", None) == index_name
            )
        except Exception:
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index listing response could not be parsed."
            ) from None

        if not matching_indexes:
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index was not found in the configured project listing."
            )
        if len(matching_indexes) != 1:
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index listing response contains an ambiguous match."
            )

        index_description = matching_indexes[0]
        try:
            listed_host = getattr(index_description, "host", None)
        except Exception:
            listed_host = None
        if not isinstance(listed_host, str) or not listed_host.strip():
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index host metadata is missing or unsupported."
            )
        try:
            hosts_match = _canonical_pinecone_host(listed_host) == (
                _canonical_pinecone_host(host)
            )
        except VectorStoreConfigurationError:
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index host metadata is missing or unsupported."
            ) from None
        if not hosts_match:
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Configured Pinecone host does not match the listed index host."
            )
        return index_description

    def _validate_index_description(
        self,
        index_description: object,
        *,
        expected_dimensions: int,
        host_mode: bool,
    ) -> None:
        try:
            index_dimensions = getattr(index_description, "dimension", None)
            index_metric = getattr(index_description, "metric", None)
        except Exception:
            if not host_mode:
                self._close_after_initialization_failure()
                raise VectorStoreConfigurationError(
                    "Pinecone index description response could not be parsed."
                ) from None
            index_dimensions = None
            index_metric = None

        if not isinstance(index_dimensions, int) or isinstance(
            index_dimensions, bool
        ):
            self._close_after_initialization_failure()
            if host_mode:
                message = (
                    "Pinecone index metadata is missing or unsupported: "
                    "dimension cannot be verified."
                )
            else:
                message = (
                    "Pinecone index description response is malformed: "
                    "dimension is missing or invalid."
                )
            raise VectorStoreConfigurationError(message)
        if not isinstance(index_metric, str) or not index_metric.strip():
            self._close_after_initialization_failure()
            if host_mode:
                message = (
                    "Pinecone index metadata is missing or unsupported: "
                    "metric cannot be verified."
                )
            else:
                message = (
                    "Pinecone index description response is malformed: "
                    "metric is missing or invalid."
                )
            raise VectorStoreConfigurationError(message)
        if index_dimensions != expected_dimensions:
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index dimension does not match the configured embedding "
                f"dimension ({index_dimensions} != {expected_dimensions})."
            )
        if index_metric != "cosine":
            self._close_after_initialization_failure()
            raise VectorStoreConfigurationError(
                "Pinecone index metric must be cosine."
            )

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
                self._index.upsert(
                    vectors=vectors,
                    namespace=self._namespace,
                    show_progress=False,
                )
            except Exception as error:
                raise _vector_store_error(
                    error,
                    "Pinecone vector upsert failed.",
                ) from None

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
                namespace=self._namespace,
                filter=metadata_filter,
                include_metadata=True,
            )
        except Exception as error:
            raise _vector_store_error(
                error,
                "Pinecone similarity search failed.",
            ) from None
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
            content_hash=segment.content_hash,
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


def _validate_pinecone_host(host: str) -> None:
    if not isinstance(host, str) or not host or host != host.strip():
        raise VectorStoreConfigurationError("Pinecone host is invalid.")
    try:
        parsed = urlsplit(host if "://" in host else f"//{host}")
        if (
            parsed.scheme not in ("", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        parsed.port
    except ValueError:
        raise VectorStoreConfigurationError("Pinecone host is invalid.") from None


def _canonical_pinecone_host(host: str) -> str:
    _validate_pinecone_host(host)
    parsed = urlsplit(host if "://" in host else f"//{host}")
    hostname = parsed.hostname
    if hostname is None:
        raise VectorStoreConfigurationError("Pinecone host is invalid.")
    return f"{hostname.rstrip('.').lower()}:{parsed.port or 443}"


_PROVIDER_ERROR_CODE_CATEGORIES = {
    "UNSUPPORTED_API_VERSION": "unsupported_api_version",
    "INVALID_ARGUMENT": "invalid_request",
    "BAD_REQUEST": "invalid_request",
    "UNAUTHENTICATED": "authentication",
    "PERMISSION_DENIED": "authentication",
    "NOT_FOUND": "not_found",
    "RESOURCE_EXHAUSTED": "rate_limited",
    "DEADLINE_EXCEEDED": "timeout",
    "UNAVAILABLE": "provider_unavailable",
    "INTERNAL": "server_error",
}


def _safe_provider_error_category(error: Exception) -> str | None:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            error_code = getattr(current, "error_code", None)
        except Exception:
            error_code = None
        if isinstance(error_code, str):
            category = _PROVIDER_ERROR_CODE_CATEGORIES.get(error_code)
            if category is not None:
                return category
        current = current.__cause__ or current.__context__
    return None


def _safe_provider_diagnostics(error: Exception) -> tuple[int | None, str]:
    try:
        raw_status_code = getattr(error, "status_code", None)
    except Exception:
        raw_status_code = None
    status_code = (
        raw_status_code
        if isinstance(raw_status_code, int)
        and not isinstance(raw_status_code, bool)
        and 100 <= raw_status_code <= 599
        else None
    )
    type_names = {
        error_type.__name__.lower()
        for error_type in _exception_chain_types(error)
    }
    provider_category = _safe_provider_error_category(error)

    if provider_category is not None:
        category = provider_category
    elif status_code in (401, 403):
        category = "authentication"
    elif status_code == 404 or any("notfound" in name for name in type_names):
        category = "not_found"
    elif status_code in (408, 504) or any("timeout" in name for name in type_names):
        category = "timeout"
    elif status_code == 429 or any("ratelimit" in name for name in type_names):
        category = "rate_limited"
    elif status_code in (400, 422):
        category = "invalid_request"
    elif status_code is not None and status_code >= 500:
        category = "server_error"
    elif any("connection" in name or "network" in name for name in type_names):
        category = "network"
    elif any(
        "parsing" in name
        or "jsondecode" in name
        or name
        in {
            "pineconeapiattributeerror",
            "pineconeapityperror",
            "pineconeapivalueerror",
        }
        for name in type_names
    ):
        category = "response_parsing"
    else:
        category = "provider_error"
    return status_code, category


def _configuration_error(
    error: Exception,
    message: str,
) -> _PineconeConfigurationError:
    status_code, category = _safe_provider_diagnostics(error)
    return _PineconeConfigurationError(
        message,
        status_code=status_code,
        category=category,
    )


def _vector_store_error(
    error: Exception,
    message: str,
) -> _PineconeVectorStoreError:
    status_code, category = _safe_provider_diagnostics(error)
    return _PineconeVectorStoreError(
        message,
        status_code=status_code,
        category=category,
    )


def _index_verification_error_message(error: Exception, operation: str) -> str:
    status_code = getattr(error, "status_code", None)
    error_types = _exception_chain_types(error)
    type_names = {error_type.__name__.lower() for error_type in error_types}
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        if status_code in (401, 403) or any(
            "unauthorized" in name or "forbidden" in name or "apikey" in name
            for name in type_names
        ):
            return f"Pinecone authentication or authorization failed while {operation}."
        if status_code == 404 or any("notfound" in name for name in type_names):
            return f"Pinecone index was not found (HTTP 404) while {operation}."

    if any(
        "timeout" in name
        or "connection" in name
        or "network" in name
        or name in {"maxretryerror", "newconnectionerror"}
        for name in type_names
    ):
        return f"Pinecone network or timeout failure while {operation}."
    if any(
        "parsing" in name
        or "jsondecode" in name
        or name in {
            "pineconeapiattributeerror",
            "pineconeapityperror",
            "pineconeapivalueerror",
        }
        for name in type_names
    ):
        return f"Pinecone SDK response parsing failed while {operation}."
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return (
            f"Pinecone provider returned HTTP {status_code} while {operation}."
        )
    return f"Pinecone provider request failed while {operation}."


def _exception_chain_types(error: Exception) -> tuple[type[BaseException], ...]:
    error_types: list[type[BaseException]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_types.append(type(current))
        current = current.__cause__ or current.__context__
    return tuple(error_types)


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
        host=settings.pinecone_host,
        namespace=settings.pinecone_namespace,
        batch_size=settings.pinecone_upsert_batch_size,
        client=client,
    )
