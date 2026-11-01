from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.services.retrieval_evaluation import (
    mean_reciprocal_rank_at_k,
    recall_at_k,
)
from app.services.retrieval_evaluation_dataset import RetrievalEvaluationDataset


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
    known_query_ids = {example.query_id for example in dataset.examples}
    unknown_query_ids = ranked_chunk_ids.keys() - known_query_ids
    if unknown_query_ids:
        raise ValueError(
            "Rankings contain unknown query IDs: "
            + ", ".join(sorted(unknown_query_ids))
        )

    answerable = tuple(
        example for example in dataset.examples if example.sufficient_evidence
    )
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
