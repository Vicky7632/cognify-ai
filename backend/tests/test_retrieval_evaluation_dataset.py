import json
from pathlib import Path

import pytest

from app.services.retrieval_evaluation_dataset import (
    DEFAULT_DATASET_PATH,
    RetrievalEvaluationDatasetError,
    load_retrieval_evaluation_dataset,
)


def _write_dataset(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_versioned_synthetic_dataset() -> None:
    dataset = load_retrieval_evaluation_dataset()

    assert DEFAULT_DATASET_PATH.is_file()
    assert dataset.dataset_version == "1.0.0"
    assert dataset.schema_version == "1.0"
    assert len(dataset.chunks) == 4
    assert len(dataset.examples) == 4
    assert any(not example.sufficient_evidence for example in dataset.examples)
    assert any(len(example.relevant_chunk_ids) > 1 for example in dataset.examples)


def test_rejects_missing_dataset_fields(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, {"dataset_version": "1.0.0"})

    with pytest.raises(RetrievalEvaluationDatasetError, match="missing required fields"):
        load_retrieval_evaluation_dataset(path)


def test_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    payload["examples"][1]["query_id"] = payload["examples"][0]["query_id"]

    with pytest.raises(RetrievalEvaluationDatasetError, match="query_id values must be unique"):
        load_retrieval_evaluation_dataset(_write_dataset(tmp_path, payload))


@pytest.mark.parametrize("version", ("1", "1.0", "latest", "1.0.0-beta"))
def test_rejects_invalid_dataset_versions(tmp_path: Path, version: str) -> None:
    payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    payload["dataset_version"] = version

    with pytest.raises(RetrievalEvaluationDatasetError, match="MAJOR.MINOR.PATCH"):
        load_retrieval_evaluation_dataset(_write_dataset(tmp_path, payload))


def test_rejects_malformed_example(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    payload["examples"][0]["sufficient_evidence"] = "yes"

    with pytest.raises(RetrievalEvaluationDatasetError, match="must be a boolean"):
        load_retrieval_evaluation_dataset(_write_dataset(tmp_path, payload))


def test_rejects_answerable_example_without_relevant_chunks(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    payload["examples"][0]["relevant_chunk_ids"] = []

    with pytest.raises(
        RetrievalEvaluationDatasetError,
        match="must include relevant chunks when evidence is sufficient",
    ):
        load_retrieval_evaluation_dataset(_write_dataset(tmp_path, payload))


def test_accepts_answerable_example_with_relevant_chunks(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))

    dataset = load_retrieval_evaluation_dataset(_write_dataset(tmp_path, payload))

    assert dataset.examples[0].sufficient_evidence is True
    assert dataset.examples[0].relevant_chunk_ids == ("chunk-break-timing",)


def test_accepts_unanswerable_example_without_relevant_chunks(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))

    dataset = load_retrieval_evaluation_dataset(_write_dataset(tmp_path, payload))
    unanswerable = next(
        example for example in dataset.examples if not example.sufficient_evidence
    )

    assert unanswerable.relevant_chunk_ids == ()
