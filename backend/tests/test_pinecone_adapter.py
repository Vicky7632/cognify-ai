from collections.abc import Sequence
from types import SimpleNamespace
from uuid import UUID

import pytest

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
        ),
    )


class FakeIndex:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.batches: list[list[dict[str, object]]] = []

    def upsert(
        self,
        *,
        vectors: Sequence[dict[str, object]],
        show_progress: bool = True,
    ) -> object:
        if self.failure is not None:
            raise self.failure
        self.batches.append(list(vectors))
        return SimpleNamespace(upserted_count=len(vectors))


class FakeClient:
    def __init__(
        self,
        *,
        dimension: int = 3,
        describe_failure: Exception | None = None,
        index: FakeIndex | None = None,
    ) -> None:
        self.dimension = dimension
        self.describe_failure = describe_failure
        self.fake_index = index or FakeIndex()
        self.described_names: list[str] = []
        self.opened_names: list[str] = []
        self.closed = False

    def describe_index(self, *, name: str) -> object:
        self.described_names.append(name)
        if self.describe_failure is not None:
            raise self.describe_failure
        return SimpleNamespace(dimension=self.dimension)

    def Index(self, *, name: str) -> FakeIndex:
        self.opened_names.append(name)
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
                },
            }
        ]
    ]


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
        embedding_dimensions=3,
        pinecone_upsert_batch_size=7,
        _env_file=None,
    )
    client = FakeClient()

    store = create_pinecone_vector_store(settings, client=client)

    assert isinstance(store, PineconeVectorStore)
    assert client.described_names == ["configured-index"]
    assert client.opened_names == ["configured-index"]


def test_index_dimension_must_match_configured_embedding_dimension() -> None:
    client = FakeClient(dimension=4)

    with pytest.raises(VectorStoreConfigurationError, match="dimension"):
        make_store(client)

    assert client.closed is True
    assert client.opened_names == []


def test_missing_index_is_reported_without_provider_details() -> None:
    client = FakeClient(
        describe_failure=RuntimeError("request failed using secret test-key")
    )

    with pytest.raises(VectorStoreConfigurationError) as error:
        make_store(client)

    assert str(error.value) == "Pinecone index could not be found or verified."
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
