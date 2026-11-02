import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

SCHEMA_VERSION = "1.0"
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "retrieval-evaluation"
    / "v1.json"
)
_DATASET_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

JsonObject: TypeAlias = dict[str, object]


class RetrievalEvaluationDatasetError(ValueError):
    """Raised when an offline retrieval evaluation dataset is invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationChunk:
    chunk_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationExample:
    query_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]
    sufficient_evidence: bool
    reference_answer: str | None
    key_facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationDataset:
    dataset_version: str
    schema_version: str
    chunks: tuple[EvaluationChunk, ...]
    examples: tuple[RetrievalEvaluationExample, ...]


def load_retrieval_evaluation_dataset(
    path: Path = DEFAULT_DATASET_PATH,
) -> RetrievalEvaluationDataset:
    """Load and validate a versioned offline retrieval evaluation dataset."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetrievalEvaluationDatasetError(
            "Unable to read the retrieval evaluation dataset."
        ) from error
    return validate_retrieval_evaluation_dataset(payload)


def validate_retrieval_evaluation_dataset(
    payload: object,
) -> RetrievalEvaluationDataset:
    dataset = _object(payload, "dataset")
    _require_fields(
        dataset,
        {"dataset_version", "schema_version", "chunks", "examples"},
        "dataset",
    )
    dataset_version = _nonempty_string(
        dataset["dataset_version"],
        "dataset_version",
    )
    if not _DATASET_VERSION_PATTERN.fullmatch(dataset_version):
        raise RetrievalEvaluationDatasetError(
            "dataset_version must use MAJOR.MINOR.PATCH format."
        )

    schema_version = _nonempty_string(dataset["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise RetrievalEvaluationDatasetError(
            f"Unsupported schema_version: {schema_version}."
        )

    chunks_value = dataset["chunks"]
    examples_value = dataset["examples"]
    if not isinstance(chunks_value, list) or not chunks_value:
        raise RetrievalEvaluationDatasetError("chunks must be a non-empty array.")
    if not isinstance(examples_value, list) or not examples_value:
        raise RetrievalEvaluationDatasetError("examples must be a non-empty array.")

    chunks = tuple(_parse_chunk(chunk, index) for index, chunk in enumerate(chunks_value))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise RetrievalEvaluationDatasetError("chunk_id values must be unique.")

    examples = tuple(
        _parse_example(example, index)
        for index, example in enumerate(examples_value)
    )
    query_ids = [example.query_id for example in examples]
    if len(set(query_ids)) != len(query_ids):
        raise RetrievalEvaluationDatasetError("query_id values must be unique.")

    available_chunk_ids = set(chunk_ids)
    for example in examples:
        if not set(example.relevant_chunk_ids) <= available_chunk_ids:
            raise RetrievalEvaluationDatasetError(
                f"Example {example.query_id} references an unknown chunk_id."
            )

    return RetrievalEvaluationDataset(
        dataset_version=dataset_version,
        schema_version=schema_version,
        chunks=chunks,
        examples=examples,
    )


def _parse_chunk(value: object, index: int) -> EvaluationChunk:
    chunk = _object(value, f"chunks[{index}]")
    _require_fields(chunk, {"chunk_id", "text"}, f"chunks[{index}]")
    return EvaluationChunk(
        chunk_id=_nonempty_string(chunk["chunk_id"], f"chunks[{index}].chunk_id"),
        text=_nonempty_string(chunk["text"], f"chunks[{index}].text"),
    )


def _parse_example(value: object, index: int) -> RetrievalEvaluationExample:
    label = f"examples[{index}]"
    example = _object(value, label)
    required_fields = {
        "query_id",
        "query",
        "relevant_chunk_ids",
        "sufficient_evidence",
    }
    _require_fields(example, required_fields, label)

    query_id = _nonempty_string(example["query_id"], f"{label}.query_id")
    query = _nonempty_string(example["query"], f"{label}.query")
    relevant_value = example["relevant_chunk_ids"]
    if (
        not isinstance(relevant_value, list)
        or any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in relevant_value)
        or len(set(relevant_value)) != len(relevant_value)
    ):
        raise RetrievalEvaluationDatasetError(
            f"{label}.relevant_chunk_ids must contain unique non-empty strings."
        )

    sufficient = example["sufficient_evidence"]
    if not isinstance(sufficient, bool):
        raise RetrievalEvaluationDatasetError(
            f"{label}.sufficient_evidence must be a boolean."
        )
    if sufficient and not relevant_value:
        raise RetrievalEvaluationDatasetError(
            f"{label} must include relevant chunks when evidence is sufficient."
        )

    reference_answer = example.get("reference_answer")
    if reference_answer is not None:
        reference_answer = _nonempty_string(
            reference_answer,
            f"{label}.reference_answer",
        )
    key_facts_value = example.get("key_facts", [])
    if (
        not isinstance(key_facts_value, list)
        or any(not isinstance(fact, str) or not fact.strip() for fact in key_facts_value)
    ):
        raise RetrievalEvaluationDatasetError(
            f"{label}.key_facts must contain non-empty strings."
        )
    key_facts = tuple(key_facts_value)

    if sufficient and not (reference_answer or key_facts):
        raise RetrievalEvaluationDatasetError(
            f"{label} needs a reference_answer or key_facts when evidence is sufficient."
        )
    if not sufficient and (reference_answer is not None or key_facts):
        raise RetrievalEvaluationDatasetError(
            f"{label} must not include answer facts when evidence is insufficient."
        )
    if not sufficient and relevant_value:
        raise RetrievalEvaluationDatasetError(
            f"{label} must not have relevant chunks when evidence is insufficient."
        )

    return RetrievalEvaluationExample(
        query_id=query_id,
        query=query,
        relevant_chunk_ids=tuple(relevant_value),
        sufficient_evidence=sufficient,
        reference_answer=reference_answer,
        key_facts=key_facts,
    )


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise RetrievalEvaluationDatasetError(f"{label} must be an object.")
    return value


def _require_fields(
    value: JsonObject,
    required: set[str],
    label: str,
) -> None:
    missing = required - value.keys()
    if missing:
        raise RetrievalEvaluationDatasetError(
            f"{label} is missing required fields: {', '.join(sorted(missing))}."
        )


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalEvaluationDatasetError(f"{label} must be a non-empty string.")
    return value
