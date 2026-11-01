from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TypeAlias
from uuid import UUID

from app.services.abstention_evaluation import (
    AbstentionEvaluation,
    AbstentionPrediction,
    AnswerDecision,
    evaluate_abstentions,
)
from app.services.citation_evaluation import (
    CitationEvaluation,
    evaluate_citations,
)
from app.services.claim_support_evaluation import (
    AnswerClaim,
    ClaimSupportEvaluation,
    evaluate_claim_support,
)
from app.services.grounded_answer_service import GroundedAnswer
from app.services.retrieval_dataset_evaluator import (
    RetrievalDatasetEvaluation,
    evaluate_retrieval_dataset,
)
from app.services.retrieval_evaluation_dataset import (
    DEFAULT_DATASET_PATH,
    RetrievalEvaluationDataset,
    load_retrieval_evaluation_dataset,
)

ChunkId: TypeAlias = str | UUID


@dataclass(frozen=True, slots=True)
class QueryEvaluationInput:
    query_id: str
    retrieved_chunk_ids: tuple[str, ...] | None = None
    available_chunk_ids: tuple[ChunkId, ...] | None = None
    answer: GroundedAnswer | None = None
    claims: tuple[AnswerClaim, ...] | None = None
    decision: AnswerDecision | None = None


@dataclass(frozen=True, slots=True)
class OfflineRAGEvaluationInput:
    queries: tuple[QueryEvaluationInput, ...]
    k: int = 10


@dataclass(frozen=True, slots=True)
class MissingEvaluationInputs:
    query_ids: tuple[str, ...]
    retrieval_query_ids: tuple[str, ...]
    citation_availability_query_ids: tuple[str, ...]
    answer_query_ids: tuple[str, ...]
    claim_query_ids: tuple[str, ...]
    decision_query_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryClaimEvaluation:
    query_id: str
    evaluation: ClaimSupportEvaluation


@dataclass(frozen=True, slots=True)
class OfflineRAGEvaluationResult:
    dataset_version: str
    retrieval: RetrievalDatasetEvaluation
    citations: CitationEvaluation
    claim_support: ClaimSupportEvaluation
    claim_support_by_query: tuple[QueryClaimEvaluation, ...]
    abstention: AbstentionEvaluation
    missing_inputs: MissingEvaluationInputs

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible result data."""
        value = _json_value(self)
        if not isinstance(value, dict):
            raise TypeError("Evaluation result serialization did not produce an object.")
        return value


def evaluate_offline_rag(
    evaluation_input: OfflineRAGEvaluationInput,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> OfflineRAGEvaluationResult:
    """Load the versioned dataset and orchestrate its offline evaluators."""
    dataset = load_retrieval_evaluation_dataset(dataset_path)
    queries = _validate_query_inputs(evaluation_input, dataset)
    input_by_id = {query.query_id: query for query in queries}
    ordered_queries = tuple(
        input_by_id[example.query_id]
        for example in dataset.examples
        if example.query_id in input_by_id
    )

    retrieval_rankings = {
        example.query_id: (
            input_by_id[example.query_id].retrieved_chunk_ids or ()
            if example.query_id in input_by_id
            else ()
        )
        for example in dataset.examples
    }
    retrieval = evaluate_retrieval_dataset(
        dataset,
        retrieval_rankings,
        k=evaluation_input.k,
    )

    available_chunk_ids = {
        query.query_id: query.available_chunk_ids or ()
        for query in ordered_queries
    }
    answers = {
        query.query_id: query.answer
        for query in ordered_queries
        if query.answer is not None
    }
    citations = evaluate_citations(dataset, available_chunk_ids, answers)

    claim_support_by_query = tuple(
        QueryClaimEvaluation(
            query_id=query.query_id,
            evaluation=evaluate_claim_support(query.claims or ()),
        )
        for query in ordered_queries
        if query.claims is not None
    )
    namespaced_claims = tuple(
        replace(
            claim,
            claim_id=f"{len(query.query_id)}:{query.query_id}:{claim.claim_id}",
        )
        for query in ordered_queries
        for claim in (query.claims or ())
    )
    aggregate_claim_support = evaluate_claim_support(namespaced_claims)
    claim_support = replace(
        aggregate_claim_support,
        claim_results=tuple(
            result
            for query_result in claim_support_by_query
            for result in query_result.evaluation.claim_results
        ),
    )

    predictions = tuple(
        AbstentionPrediction(query.query_id, query.decision)
        for query in ordered_queries
        if query.decision is not None
    )
    abstention = evaluate_abstentions(dataset, predictions)

    missing_query_ids = tuple(
        example.query_id
        for example in dataset.examples
        if example.query_id not in input_by_id
    )
    return OfflineRAGEvaluationResult(
        dataset_version=dataset.dataset_version,
        retrieval=retrieval,
        citations=citations,
        claim_support=claim_support,
        claim_support_by_query=claim_support_by_query,
        abstention=abstention,
        missing_inputs=MissingEvaluationInputs(
            query_ids=missing_query_ids,
            retrieval_query_ids=_missing_component_inputs(
                dataset,
                input_by_id,
                "retrieved_chunk_ids",
            ),
            citation_availability_query_ids=_missing_component_inputs(
                dataset,
                input_by_id,
                "available_chunk_ids",
            ),
            answer_query_ids=_missing_component_inputs(
                dataset,
                input_by_id,
                "answer",
            ),
            claim_query_ids=_missing_component_inputs(
                dataset,
                input_by_id,
                "claims",
            ),
            decision_query_ids=_missing_component_inputs(
                dataset,
                input_by_id,
                "decision",
            ),
        ),
    )


def _validate_query_inputs(
    evaluation_input: OfflineRAGEvaluationInput,
    dataset: RetrievalEvaluationDataset,
) -> tuple[QueryEvaluationInput, ...]:
    if not isinstance(evaluation_input, OfflineRAGEvaluationInput):
        raise ValueError("evaluation_input must be an OfflineRAGEvaluationInput.")
    if (
        isinstance(evaluation_input.k, bool)
        or not isinstance(evaluation_input.k, int)
        or evaluation_input.k < 1
    ):
        raise ValueError("k must be a positive integer.")
    if isinstance(evaluation_input.queries, (str, bytes)) or not isinstance(
        evaluation_input.queries,
        Sequence,
    ):
        raise ValueError("queries must be a sequence of QueryEvaluationInput values.")

    known_ids = {example.query_id for example in dataset.examples}
    seen_ids: set[str] = set()
    queries: list[QueryEvaluationInput] = []
    dataset_chunk_ids = {chunk.chunk_id for chunk in dataset.chunks}
    for index, query in enumerate(evaluation_input.queries):
        if not isinstance(query, QueryEvaluationInput):
            raise ValueError(f"queries[{index}] must be a QueryEvaluationInput.")
        if not isinstance(query.query_id, str) or not query.query_id.strip():
            raise ValueError(f"queries[{index}].query_id must be non-empty.")
        if query.query_id not in known_ids:
            raise ValueError(f"Unknown query ID: {query.query_id}.")
        if query.query_id in seen_ids:
            raise ValueError(f"Duplicate query ID: {query.query_id}.")
        seen_ids.add(query.query_id)

        if query.retrieved_chunk_ids is not None:
            if isinstance(query.retrieved_chunk_ids, (str, bytes)) or not isinstance(
                query.retrieved_chunk_ids,
                Sequence,
            ):
                raise ValueError(
                    f"queries[{index}].retrieved_chunk_ids must be a sequence."
                )
            if any(
                not isinstance(chunk_id, str) or not chunk_id.strip()
                for chunk_id in query.retrieved_chunk_ids
            ):
                raise ValueError(
                    f"queries[{index}].retrieved_chunk_ids must contain non-empty strings."
                )
            unknown_chunks = set(query.retrieved_chunk_ids) - dataset_chunk_ids
            if unknown_chunks:
                raise ValueError(
                    f"Query {query.query_id} contains unknown retrieved chunk IDs."
                )
        if query.available_chunk_ids is not None:
            if isinstance(query.available_chunk_ids, (str, bytes)) or not isinstance(
                query.available_chunk_ids,
                Sequence,
            ):
                raise ValueError(
                    f"queries[{index}].available_chunk_ids must be a sequence."
                )
            _validate_chunk_ids(
                query.available_chunk_ids,
                f"queries[{index}].available_chunk_ids",
            )
        if query.answer is not None and not isinstance(query.answer, GroundedAnswer):
            raise ValueError(f"queries[{index}].answer must be a GroundedAnswer.")
        if query.claims is not None:
            if not isinstance(query.claims, tuple):
                raise ValueError(f"queries[{index}].claims must be a tuple.")
            evaluate_claim_support(query.claims)
        if query.decision is not None:
            try:
                AnswerDecision(query.decision)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"queries[{index}].decision must be answered or abstained."
                ) from error
        queries.append(query)
    return tuple(queries)


def _validate_chunk_ids(chunk_ids: tuple[ChunkId, ...], label: str) -> None:
    try:
        set(chunk_ids)
    except TypeError as error:
        raise ValueError(f"{label} must contain hashable chunk IDs.") from error
    if any(not isinstance(chunk_id, (str, UUID)) for chunk_id in chunk_ids):
        raise ValueError(f"{label} must contain string or UUID chunk IDs.")


def _missing_component_inputs(
    dataset: RetrievalEvaluationDataset,
    input_by_id: Mapping[str, QueryEvaluationInput],
    field_name: str,
) -> tuple[str, ...]:
    return tuple(
        example.query_id
        for example in dataset.examples
        if example.query_id not in input_by_id
        or getattr(input_by_id[example.query_id], field_name) is None
    )


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported evaluation result value: {type(value).__name__}.")
