from collections.abc import Sequence
from hashlib import sha256
from types import SimpleNamespace
from traceback import format_exception
from uuid import UUID

import pytest
from pinecone import ApiError

from app.core.config import Settings
from app.vector_store.pinecone_adapter import (
    PineconeVectorStore,
    create_pinecone_vector_store,
    stable_segment_vector_id,
)
from app.vector_store.protocol import (
    VectorMetadata,
    VectorRecord,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreInputError,
)

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
SEGMENT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{index:012d}")
    for index in range(2, 8)
)


def make_record(
    index: int,
    *,
    values: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> VectorRecord:
    return VectorRecord(
        id=stable_segment_vector_id(DOCUMENT_ID, SEGMENT_IDS[index]),
        values=values,
        metadata=VectorMetadata(
            document_id=DOCUMENT_ID,
            segment_id=SEGMENT_IDS[index],
            ordinal=index,
            content_hash=sha256(f"segment {index}".encode("utf-8")).hexdigest(),
        ),
    )


class FakeIndex:
    def __init__(
        self,
        failure: Exception | None = None,
        query_response: object | None = None,
    ) -> None:
        self.failure = failure
        self.batches: list[list[dict[str, object]]] = []
        self.upsert_namespaces: list[str] = []
        self.query_response = query_response or SimpleNamespace(matches=[])
        self.queries: list[dict[str, object]] = []

    def upsert(
        self,
        *,
        vectors: Sequence[dict[str, object]],
        namespace: str = "",
        show_progress: bool = True,
    ) -> object:
        if self.failure is not None:
            raise self.failure
        self.batches.append(list(vectors))
        self.upsert_namespaces.append(namespace)
        return SimpleNamespace(upserted_count=len(vectors))

    def query(
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        namespace: str = "",
        filter: dict[str, object] | None = None,
        include_metadata: bool = False,
    ) -> object:
        if self.failure is not None:
            raise self.failure
        self.queries.append(
            {
                "vector": list(vector),
                "top_k": top_k,
                "namespace": namespace,
                "filter": filter,
                "include_metadata": include_metadata,
            }
        )
        return self.query_response


class FakeClient:
    def __init__(
        self,
        *,
        dimension: int = 3,
        metric: str = "cosine",
        host: str = "https://test-index-host.svc.pinecone.io",
        describe_failure: Exception | None = None,
        describe_response: object | None = None,
        listed_indexes: list[object] | None = None,
        list_failure: Exception | None = None,
        index_failure: Exception | None = None,
        index: FakeIndex | None = None,
    ) -> None:
        self.dimension = dimension
        self.metric = metric
        self.host = host
        self.describe_failure = describe_failure
        self.describe_response = describe_response
        self.listed_indexes = listed_indexes
        self.list_failure = list_failure
        self.index_failure = index_failure
        self.fake_index = index or FakeIndex()
        self.described_names: list[str] = []
        self.opened_names: list[str] = []
        self.opened_hosts: list[str] = []
        self.list_calls = 0
        self.closed = False

    @property
    def indexes(self):
        return self

    def list(self) -> list[object]:
        self.list_calls += 1
        if self.list_failure is not None:
            raise self.list_failure
        if self.listed_indexes is not None:
            return self.listed_indexes
        return [
            SimpleNamespace(
                name="test-index",
                host=self.host,
                dimension=self.dimension,
                metric=self.metric,
            )
        ]

    def describe_index(self, *, name: str) -> object:
        self.described_names.append(name)
        if self.describe_failure is not None:
            raise self.describe_failure
        if self.describe_response is not None:
            return self.describe_response
        return SimpleNamespace(dimension=self.dimension, metric=self.metric)

    def Index(self, *, name: str = "", host: str = "") -> FakeIndex:
        if name:
            self.opened_names.append(name)
        if host:
            self.opened_hosts.append(host)
        if self.index_failure is not None:
            raise self.index_failure
        return self.fake_index

    def close(self) -> None:
        self.closed = True


def make_store(
    client: FakeClient | None = None,
    *,
    batch_size: int = 100,
) -> tuple[PineconeVectorStore, FakeClient]:
    fake_client = client or FakeClient()
    return (
        PineconeVectorStore(
            api_key="test-key",
            index_name="test-index",
            expected_dimensions=3,
            batch_size=batch_size,
            client=fake_client,
        ),
        fake_client,
    )


def test_vector_ids_are_stable_for_document_and_segment_identity() -> None:
    first = stable_segment_vector_id(DOCUMENT_ID, SEGMENT_IDS[0])

    assert first == stable_segment_vector_id(DOCUMENT_ID, SEGMENT_IDS[0])
    assert first != stable_segment_vector_id(DOCUMENT_ID, SEGMENT_IDS[1])


def test_single_upsert_preserves_vector_and_metadata() -> None:
    store, client = make_store()

    store.upsert([make_record(0)])

    assert client.described_names == ["test-index"]
    assert client.opened_names == ["test-index"]
    assert client.fake_index.batches == [
        [
            {
                "id": make_record(0).id,
                "values": [0.1, 0.2, 0.3],
                "metadata": {
                    "document_id": str(DOCUMENT_ID),
                    "segment_id": str(SEGMENT_IDS[0]),
                    "ordinal": 0,
                    "content_hash": sha256(b"segment 0").hexdigest(),
                },
            }
        ]
    ]
    assert client.fake_index.upsert_namespaces == ["cognify"]


def test_upsert_batches_large_inputs_in_order() -> None:
    store, client = make_store(batch_size=2)
    records = [make_record(index) for index in range(5)]

    store.upsert(records)

    assert [
        [vector["id"] for vector in batch]
        for batch in client.fake_index.batches
    ] == [
        [record.id for record in records[0:2]],
        [record.id for record in records[2:4]],
        [record.id for record in records[4:5]],
    ]


@pytest.mark.parametrize(
    "record",
    [
        VectorRecord("", (0.1, 0.2, 0.3), VectorMetadata(DOCUMENT_ID, SEGMENT_IDS[0], 0)),
        VectorRecord("   ", (0.1, 0.2, 0.3), VectorMetadata(DOCUMENT_ID, SEGMENT_IDS[0], 0)),
        VectorRecord("empty", (), VectorMetadata(DOCUMENT_ID, SEGMENT_IDS[0], 0)),
        VectorRecord("short", (0.1, 0.2), VectorMetadata(DOCUMENT_ID, SEGMENT_IDS[0], 0)),
        VectorRecord("nan", (0.1, float("nan"), 0.3), VectorMetadata(DOCUMENT_ID, SEGMENT_IDS[0], 0)),
        VectorRecord("infinity", (0.1, float("inf"), 0.3), VectorMetadata(DOCUMENT_ID, SEGMENT_IDS[0], 0)),
        VectorRecord("boolean", (0.1, True, 0.3), VectorMetadata(DOCUMENT_ID, SEGMENT_IDS[0], 0)),
        VectorRecord("bad metadata", (0.1, 0.2, 0.3), None),
    ],
)
def test_invalid_vectors_are_rejected_before_upsert(record: VectorRecord) -> None:
    store, client = make_store()

    with pytest.raises(VectorStoreInputError):
        store.upsert([record])

    assert client.fake_index.batches == []


def test_inconsistent_dimensions_within_request_are_rejected() -> None:
    store, client = make_store()

    with pytest.raises(VectorStoreInputError, match="dimensions"):
        store.upsert(
            [
                make_record(0),
                make_record(1, values=(0.1, 0.2)),
            ]
        )

    assert client.fake_index.batches == []


def test_empty_upsert_is_rejected() -> None:
    store, client = make_store()

    with pytest.raises(VectorStoreInputError, match="At least one"):
        store.upsert([])

    assert client.fake_index.batches == []


@pytest.mark.parametrize(
    ("api_key", "index_name", "message"),
    [
        ("", "test-index", "API key"),
        ("test-key", " ", "index name"),
    ],
)
def test_missing_required_constructor_configuration_is_rejected(
    api_key: str,
    index_name: str,
    message: str,
) -> None:
    with pytest.raises(VectorStoreConfigurationError, match=message):
        PineconeVectorStore(
            api_key=api_key,
            index_name=index_name,
            expected_dimensions=3,
        )


def test_factory_requires_api_key_and_index_configuration() -> None:
    with pytest.raises(VectorStoreConfigurationError, match="API key"):
        create_pinecone_vector_store(
            Settings(
                pinecone_api_key=None,
                pinecone_index=None,
                _env_file=None,
            )
        )

    settings = Settings(
        pinecone_api_key="test-key",
        pinecone_index=None,
        _env_file=None,
    )
    with pytest.raises(VectorStoreConfigurationError, match="index name"):
        create_pinecone_vector_store(settings)


def test_factory_uses_configured_settings_and_validates_index() -> None:
    settings = Settings(
        pinecone_api_key="test-key",
        pinecone_index="configured-index",
        pinecone_namespace="configured-namespace",
        embedding_dimensions=3,
        pinecone_upsert_batch_size=7,
        _env_file=None,
    )
    client = FakeClient()

    store = create_pinecone_vector_store(settings, client=client)

    assert isinstance(store, PineconeVectorStore)
    assert client.described_names == ["configured-index"]
    assert client.opened_names == ["configured-index"]
    assert client.list_calls == 0
    assert client.opened_hosts == []
    store.upsert([make_record(0)])
    assert client.fake_index.upsert_namespaces == ["configured-namespace"]


def test_factory_uses_requested_live_smoke_configuration_with_fake_client() -> None:
    settings = Settings(
        pinecone_api_key="test-key",
        pinecone_index="cognify-ai",
        pinecone_namespace="cognify",
        embedding_dimensions=1024,
        _env_file=None,
    )
    index = FakeIndex()
    client = FakeClient(dimension=1024, metric="cosine", index=index)

    store = create_pinecone_vector_store(settings, client=client)
    store.upsert([make_record(0, values=(0.1,) * 1024)])
    results = store.search((0.2,) * 1024, top_k=1)

    assert client.described_names == ["cognify-ai"]
    assert client.opened_names == ["cognify-ai"]
    assert client.list_calls == 0
    assert client.opened_hosts == []
    assert index.upsert_namespaces == ["cognify"]
    assert index.queries[0]["namespace"] == "cognify"
    assert len(index.batches[0][0]["values"]) == 1024
    assert results == ()


def make_host_mode_store(
    client: FakeClient,
    *,
    host: str = "test-index-host.svc.pinecone.io",
) -> PineconeVectorStore:
    return PineconeVectorStore(
        api_key="test-key",
        index_name="test-index",
        expected_dimensions=3,
        host=host,
        client=client,
    )


def test_host_mode_lists_and_validates_index_then_opens_configured_host() -> None:
    client = FakeClient()

    store = make_host_mode_store(client)

    assert isinstance(store, PineconeVectorStore)
    assert client.list_calls == 1
    assert client.described_names == []
    assert client.opened_names == []
    assert client.opened_hosts == ["test-index-host.svc.pinecone.io"]


def test_factory_passes_configured_host_to_host_mode() -> None:
    settings = Settings(
        pinecone_api_key="test-key",
        pinecone_index="configured-index",
        pinecone_host="https://test-index-host.svc.pinecone.io",
        embedding_dimensions=3,
        _env_file=None,
    )
    client = FakeClient(
        listed_indexes=[
            SimpleNamespace(
                name="configured-index",
                host="https://test-index-host.svc.pinecone.io",
                dimension=3,
                metric="cosine",
            )
        ]
    )

    store = create_pinecone_vector_store(settings, client=client)

    assert isinstance(store, PineconeVectorStore)
    assert client.list_calls == 1
    assert client.described_names == []
    assert client.opened_names == []
    assert client.opened_hosts == ["https://test-index-host.svc.pinecone.io"]


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (
            FakeClient(host="different-host.svc.pinecone.io"),
            "does not match the listed index host",
        ),
        (
            FakeClient(listed_indexes=[]),
            "was not found in the configured project listing",
        ),
        (
            FakeClient(dimension=4),
            "dimension does not match the configured embedding dimension",
        ),
        (
            FakeClient(metric="dotproduct"),
            "metric must be cosine",
        ),
        (
            FakeClient(
                listed_indexes=[
                    SimpleNamespace(
                        name="test-index",
                        host="https://test-index-host.svc.pinecone.io",
                    )
                ]
            ),
            "metadata is missing or unsupported",
        ),
        (
            FakeClient(
                listed_indexes=[
                    SimpleNamespace(
                        name="test-index",
                        host="https://test-index-host.svc.pinecone.io",
                        dimension=3,
                    )
                ]
            ),
            "metadata is missing or unsupported",
        ),
    ],
)
def test_host_mode_rejects_invalid_index_before_opening_host(
    client: FakeClient,
    message: str,
) -> None:
    with pytest.raises(VectorStoreConfigurationError, match=message):
        make_host_mode_store(client)

    assert client.list_calls == 1
    assert client.described_names == []
    assert client.opened_names == []
    assert client.opened_hosts == []
    assert client.closed is True


@pytest.mark.parametrize(
    "host",
    [
        "",
        " host.svc.pinecone.io",
        "http://host.svc.pinecone.io",
        "https://user@host.svc.pinecone.io",
        "https://host.svc.pinecone.io/path",
        "https://host.svc.pinecone.io?token=hidden",
    ],
)
def test_invalid_configured_host_is_rejected_without_sdk_calls(host: str) -> None:
    client = FakeClient()

    with pytest.raises(VectorStoreConfigurationError, match="host is invalid"):
        make_host_mode_store(client, host=host)

    assert client.list_calls == 0
    assert client.described_names == []
    assert client.opened_hosts == []


def test_index_metric_must_be_cosine() -> None:
    client = FakeClient(metric="dotproduct")

    with pytest.raises(VectorStoreConfigurationError, match="metric must be cosine"):
        make_store(client)

    assert client.closed is True
    assert client.opened_names == []


def test_index_dimension_must_match_configured_embedding_dimension() -> None:
    client = FakeClient(dimension=4)

    with pytest.raises(
        VectorStoreConfigurationError,
        match="dimension does not match the configured embedding dimension",
    ):
        make_store(client)

    assert client.closed is True
    assert client.opened_names == []


class FakeHttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("provider response contains secret test-key")
        self.status_code = status_code


class ResponseParsingError(Exception):
    pass


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FakeHttpError(401), "authentication or authorization failed"),
        (FakeHttpError(403), "authentication or authorization failed"),
        (FakeHttpError(404), "index was not found"),
        (TimeoutError("secret test-key"), "network or timeout failure"),
        (ConnectionError("secret test-key"), "network or timeout failure"),
        (ResponseParsingError("secret test-key"), "SDK response parsing failed"),
    ],
)
def test_describe_index_failures_have_safe_diagnostics(
    failure: Exception,
    message: str,
) -> None:
    client = FakeClient(
        describe_failure=failure
    )

    with pytest.raises(VectorStoreConfigurationError) as error:
        make_store(client)

    assert message in str(error.value)
    assert "describing the index" in str(error.value)
    assert "test-key" not in str(error.value)
    assert client.closed is True


def test_index_connection_failure_is_distinguished_from_describe_failure() -> None:
    client = FakeClient(index_failure=TimeoutError("secret test-key"))

    with pytest.raises(VectorStoreConfigurationError) as error:
        make_store(client)

    assert str(error.value) == (
        "Pinecone network or timeout failure while connecting to the index."
    )
    assert "test-key" not in str(error.value)
    assert client.described_names == ["test-index"]
    assert client.opened_names == ["test-index"]
    assert client.closed is True


@pytest.mark.parametrize(
    "description",
    [
        SimpleNamespace(metric="cosine"),
        SimpleNamespace(dimension=3),
        SimpleNamespace(dimension="3", metric="cosine"),
    ],
)
def test_malformed_describe_response_is_reported_safely(
    description: object,
) -> None:
    client = FakeClient(describe_response=description)

    with pytest.raises(
        VectorStoreConfigurationError,
        match="index description response is malformed",
    ):
        make_store(client)

    assert client.closed is True
    assert client.opened_names == []


def test_generic_describe_failure_is_safe_and_identifies_operation() -> None:
    client = FakeClient(
        describe_failure=RuntimeError("request failed using secret test-key")
    )

    with pytest.raises(VectorStoreConfigurationError) as error:
        make_store(client)

    assert str(error.value) == (
        "Pinecone provider request failed while describing the index."
    )
    assert "test-key" not in str(error.value)
    assert client.closed is True


def test_sdk_upsert_failure_is_reported_without_provider_details() -> None:
    client = FakeClient(
        index=FakeIndex(
            failure=RuntimeError("request failed using secret test-key")
        )
    )
    store, _ = make_store(client)

    with pytest.raises(VectorStoreError) as error:
        store.upsert([make_record(0)])

    assert str(error.value) == "Pinecone vector upsert failed."
    assert "test-key" not in str(error.value)


@pytest.mark.parametrize("operation", ("upsert", "search"))
def test_sdk_api_error_preserves_cause_and_safe_diagnostics(
    operation: str,
) -> None:
    failure = ApiError(
        "provider message containing secret-value",
        400,
        body={"error": {"message": "sensitive response body"}},
        headers={"Authorization": "Bearer sensitive-header"},
        error_code="sensitive-error-code",
        request_id="sensitive-request-id",
    )
    client = FakeClient(index=FakeIndex(failure=failure))
    store, _ = make_store(client)

    with pytest.raises(VectorStoreError) as error:
        if operation == "upsert":
            store.upsert([make_record(0)])
        else:
            store.search((0.1, 0.2, 0.3), top_k=1)

    assert str(error.value) == (
        "Pinecone vector upsert failed."
        if operation == "upsert"
        else "Pinecone similarity search failed."
    )
    assert error.value.__cause__ is None
    assert error.value.__context__ is failure
    assert error.value.__suppress_context__ is True
    assert error.value.status_code == 400
    assert error.value.category == "invalid_request"
    for sensitive_value in (
        "provider message",
        "secret-value",
        "sensitive response body",
        "sensitive-header",
        "sensitive-error-code",
        "sensitive-request-id",
    ):
        assert sensitive_value not in str(error.value)


@pytest.mark.parametrize(
    ("provider_code", "expected_category"),
    [
        ("UNSUPPORTED_API_VERSION", "unsupported_api_version"),
        ("INVALID_ARGUMENT", "invalid_request"),
        ("UNAUTHENTICATED", "authentication"),
        ("NOT_FOUND", "not_found"),
        ("RESOURCE_EXHAUSTED", "rate_limited"),
        ("DEADLINE_EXCEEDED", "timeout"),
        ("UNAVAILABLE", "provider_unavailable"),
        ("INTERNAL", "server_error"),
    ],
)
def test_allowlisted_sdk_error_code_maps_to_safe_category(
    provider_code: str,
    expected_category: str,
) -> None:
    failure = ApiError(
        "sensitive provider message",
        400,
        body={"error": {"code": provider_code}},
        error_code=provider_code,
    )
    client = FakeClient(index=FakeIndex(failure=failure))
    store, _ = make_store(client)

    with pytest.raises(VectorStoreError) as error:
        store.upsert([make_record(0)])

    assert str(error.value) == "Pinecone vector upsert failed."
    assert error.value.__cause__ is None
    assert error.value.__context__ is failure
    assert error.value.__suppress_context__ is True
    assert error.value.status_code == 400
    assert error.value.category == expected_category
    assert provider_code not in str(error.value)
    assert "sensitive provider message" not in str(error.value)


@pytest.mark.parametrize(
    "provider_code",
    [
        "UNRECOGNIZED_PROVIDER_CODE",
        None,
        400,
        {"code": "UNSUPPORTED_API_VERSION"},
        ["UNSUPPORTED_API_VERSION"],
    ],
)
def test_unknown_missing_or_malformed_sdk_error_code_uses_safe_fallback(
    provider_code: object,
) -> None:
    failure = ApiError(
        "sensitive provider message",
        400,
        body={"error": {"message": "sensitive response body"}},
        error_code=provider_code,
    )
    client = FakeClient(index=FakeIndex(failure=failure))
    store, _ = make_store(client)

    with pytest.raises(VectorStoreError) as error:
        store.upsert([make_record(0)])

    assert str(error.value) == "Pinecone vector upsert failed."
    assert error.value.__cause__ is None
    assert error.value.__context__ is failure
    assert error.value.__suppress_context__ is True
    assert error.value.status_code == 400
    assert error.value.category == "invalid_request"
    assert "sensitive provider message" not in str(error.value)
    assert "sensitive response body" not in str(error.value)


def test_formatted_traceback_hides_provider_details_but_keeps_internal_context() -> None:
    failure = ApiError(
        "provider message with secret-key, private-host, and user document text",
        400,
        body={
            "error": {
                "message": "raw response body with secret-key and private-user-data"
            }
        },
        headers={"Authorization": "secret-header-value"},
        error_code="sensitive-provider-code",
        request_id="sensitive-request-id",
    )
    client = FakeClient(index=FakeIndex(failure=failure))
    store, _ = make_store(client)

    with pytest.raises(VectorStoreError) as error:
        store.upsert([make_record(0)])

    formatted_traceback = "".join(format_exception(error.value))

    assert error.value.__cause__ is None
    assert error.value.__context__ is failure
    assert error.value.__suppress_context__ is True
    assert str(error.value) == "Pinecone vector upsert failed."
    for sensitive_value in (
        "secret-key",
        "private-host",
        "user document text",
        "raw response body",
        "private-user-data",
        "secret-header-value",
        "sensitive-provider-code",
        "sensitive-request-id",
    ):
        assert sensitive_value not in formatted_traceback


def test_sdk_initialization_error_preserves_context_and_safe_diagnostics() -> None:
    failure = FakeHttpError(401)
    client = FakeClient(describe_failure=failure)

    with pytest.raises(VectorStoreConfigurationError) as error:
        make_store(client)

    assert error.value.__cause__ is None
    assert error.value.__context__ is failure
    assert error.value.__suppress_context__ is True
    assert error.value.status_code == 401
    assert error.value.category == "authentication"
    assert "secret test-key" not in str(error.value)
    assert "secret test-key" not in "".join(format_exception(error.value))


def make_match(
    index: int,
    *,
    score: float,
    extra_metadata: dict[str, object] | None = None,
) -> SimpleNamespace:
    record = make_record(index)
    metadata: dict[str, object] = record.metadata.as_dict()
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return SimpleNamespace(id=record.id, score=score, metadata=metadata)


def test_search_parses_ordered_results_and_applies_document_filter() -> None:
    response = SimpleNamespace(
        matches=[
            make_match(
                1,
                score=0.91,
                extra_metadata={
                    "source": "lesson.pdf",
                    "tags": ["lecture", "pdf"],
                },
            ),
            make_match(0, score=0.76),
        ]
    )
    client = FakeClient(index=FakeIndex(query_response=response))
    store, _ = make_store(client)

    results = store.search(
        (0.4, 0.5, 0.6),
        top_k=2,
        document_id=DOCUMENT_ID,
    )

    assert client.fake_index.queries == [
        {
            "vector": [0.4, 0.5, 0.6],
            "top_k": 2,
            "namespace": "cognify",
            "filter": {"document_id": {"$eq": str(DOCUMENT_ID)}},
            "include_metadata": True,
        }
    ]
    assert [
        (
            result.vector_id,
            result.document_id,
            result.segment_id,
            result.ordinal,
            result.score,
        )
        for result in results
    ] == [
        (make_record(1).id, DOCUMENT_ID, SEGMENT_IDS[1], 1, 0.91),
        (make_record(0).id, DOCUMENT_ID, SEGMENT_IDS[0], 0, 0.76),
    ]
    assert results[0].metadata["source"] == "lesson.pdf"
    assert results[0].metadata["tags"] == ("lecture", "pdf")


def test_search_without_document_filter_requests_global_matches() -> None:
    store, client = make_store()

    assert store.search((0.1, 0.2, 0.3), top_k=5) == ()
    assert client.fake_index.queries[0]["filter"] is None
    assert client.fake_index.queries[0]["top_k"] == 5
    assert client.fake_index.queries[0]["namespace"] == "cognify"


@pytest.mark.parametrize(
    ("query_vector", "top_k", "message"),
    [
        ((), 1, "empty"),
        ((0.1, 0.2), 1, "dimensions"),
        ((0.1, float("nan"), 0.3), 1, "finite"),
        ((0.1, float("inf"), 0.3), 1, "finite"),
        ((0.1, 10**1000, 0.3), 1, "finite"),
        ((0.1, True, 0.3), 1, "finite"),
        ((0.1, 0.2, 0.3), 0, "top_k"),
        ((0.1, 0.2, 0.3), True, "top_k"),
    ],
)
def test_invalid_search_input_is_rejected_before_sdk_call(
    query_vector: tuple[object, ...],
    top_k: int,
    message: str,
) -> None:
    store, client = make_store()

    with pytest.raises(VectorStoreInputError, match=message):
        store.search(query_vector, top_k=top_k)

    assert client.fake_index.queries == []


def test_invalid_document_filter_is_rejected_before_sdk_call() -> None:
    store, client = make_store()

    with pytest.raises(VectorStoreInputError, match="Document filter"):
        store.search((0.1, 0.2, 0.3), top_k=1, document_id="not-a-uuid")

    assert client.fake_index.queries == []


@pytest.mark.parametrize(
    "match",
    [
        SimpleNamespace(id="", score=0.5, metadata={}),
        SimpleNamespace(id="vector", score=float("nan"), metadata={}),
        SimpleNamespace(id="vector", score=0.5, metadata=None),
        SimpleNamespace(
            id="vector",
            score=0.5,
            metadata={
                "document_id": "invalid",
                "segment_id": str(SEGMENT_IDS[0]),
                "ordinal": 0,
            },
        ),
        SimpleNamespace(
            id="wrong-id",
            score=0.5,
            metadata={
                "document_id": str(DOCUMENT_ID),
                "segment_id": str(SEGMENT_IDS[0]),
                "ordinal": 0,
            },
        ),
        SimpleNamespace(
            id=make_record(0).id,
            score=0.5,
            metadata={
                "document_id": str(DOCUMENT_ID),
                "segment_id": str(SEGMENT_IDS[0]),
                "ordinal": True,
            },
        ),
    ],
)
def test_malformed_search_match_raises_safe_error(match: object) -> None:
    client = FakeClient(
        index=FakeIndex(query_response=SimpleNamespace(matches=[match]))
    )
    store, _ = make_store(client)

    with pytest.raises(VectorStoreError, match="malformed similarity response"):
        store.search((0.1, 0.2, 0.3), top_k=1)


def test_search_provider_failure_is_safe() -> None:
    client = FakeClient(
        index=FakeIndex(
            failure=RuntimeError("request failed using secret test-key")
        )
    )
    store, _ = make_store(client)

    with pytest.raises(VectorStoreError) as error:
        store.search((0.1, 0.2, 0.3), top_k=1)

    assert str(error.value) == "Pinecone similarity search failed."
    assert "test-key" not in str(error.value)
