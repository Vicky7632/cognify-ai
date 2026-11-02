from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.services.retrieval_evaluation import (
    mean_reciprocal_rank_at_k,
    recall_at_k,
)
from app.services.retrieval_evaluation_dataset import (
    DEFAULT_DATASET_PATH,
    RetrievalEvaluationDataset,
    load_retrieval_evaluation_dataset,
)


@dataclass(frozen=True, slots=True)
class RetrievalDatasetEvaluation:
    k: int
    answerable_query_count: int
    total_query_count: int
    recall_at_k: float
    mean_reciprocal_rank_at_k: float


def evaluate_retrieval_dataset(
    dataset: RetrievalEvaluationDataset,
    ranked_chunk_ids: Mapping[str, Sequence[str]],
    *,
    k: int,
) -> RetrievalDatasetEvaluation:
    """Score supplied offline rankings against answerable dataset examples."""
    if not isinstance(ranked_chunk_ids, Mapping):
        raise ValueError("Rankings must map query IDs to ranked chunk IDs.")

    known_chunk_ids = {chunk.chunk_id for chunk in dataset.chunks}
    for query_id, ranking in ranked_chunk_ids.items():
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Ranking query IDs must be non-empty strings.")
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
            raise ValueError(
                f"Ranking for query {query_id} must be a sequence of chunk IDs."
            )
        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in ranking
        ):
            raise ValueError(
                f"Ranking for query {query_id} must contain non-empty chunk IDs."
            )
        if len(set(ranking)) != len(ranking):
            raise ValueError(
                f"Ranking for query {query_id} must contain unique chunk IDs."
            )
        unknown_chunk_ids = set(ranking) - known_chunk_ids
        if unknown_chunk_ids:
            raise ValueError(
                f"Ranking for query {query_id} contains unknown chunk IDs: "
                + ", ".join(sorted(unknown_chunk_ids))
            )

    known_query_ids = {example.query_id for example in dataset.examples}
    unknown_query_ids = ranked_chunk_ids.keys() - known_query_ids
    if unknown_query_ids:
        raise ValueError(
            "Rankings contain unknown query IDs: "
            + ", ".join(sorted(unknown_query_ids))
        )

    for example in dataset.examples:
        if example.sufficient_evidence and not example.relevant_chunk_ids:
            raise ValueError(
                f"Answerable query {example.query_id} must have at least one "
                "relevant chunk ID."
            )

    answerable = tuple(
        example for example in dataset.examples if example.sufficient_evidence
    )
    if not answerable:
        raise ValueError("At least one answerable query is required.")

    rankings = tuple(
        ranked_chunk_ids.get(example.query_id, ())
        for example in answerable
    )
    relevant = tuple(example.relevant_chunk_ids for example in answerable)

    return RetrievalDatasetEvaluation(
        k=k,
        answerable_query_count=len(answerable),
        total_query_count=len(dataset.examples),
        recall_at_k=recall_at_k(rankings, relevant, k=k),
        mean_reciprocal_rank_at_k=mean_reciprocal_rank_at_k(
            rankings,
            relevant,
            k=k,
        ),
    )


def run_retrieval_dataset_evaluation(
    ranked_chunk_ids: Mapping[str, Sequence[str]],
    *,
    k: int,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> RetrievalDatasetEvaluation:
    """Load the versioned dataset and evaluate supplied offline rankings."""
    dataset = load_retrieval_evaluation_dataset(dataset_path)
    return evaluate_retrieval_dataset(dataset, ranked_chunk_ids, k=k)
