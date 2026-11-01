import pytest

from app.services.retrieval_dataset_evaluator import evaluate_retrieval_dataset
from app.services.retrieval_evaluation_dataset import (
    load_retrieval_evaluation_dataset,
)


def test_evaluates_ranked_chunks_for_answerable_dataset_queries() -> None:
    dataset = load_retrieval_evaluation_dataset()

    result = evaluate_retrieval_dataset(
        dataset,
        {
            "study-break-duration": [
                "chunk-study-breaks",
                "chunk-break-timing",
            ],
            "study-break-benefit": ["chunk-study-breaks"],
            "study-techniques-comparison": [
                "chunk-active-recall",
                "chunk-spaced-review",
            ],
            "unsupported-exam-date": ["chunk-break-timing"],
        },
        k=2,
    )

    assert result.k == 2
    assert result.answerable_query_count == 3
    assert result.total_query_count == 4
    assert result.recall_at_k == 1.0
    assert result.mean_reciprocal_rank_at_k == pytest.approx(5 / 6)


def test_missing_rankings_count_as_misses_and_unanswerable_queries_are_excluded() -> None:
    dataset = load_retrieval_evaluation_dataset()

    result = evaluate_retrieval_dataset(dataset, {}, k=3)

    assert result.answerable_query_count == 3
    assert result.total_query_count == 4
    assert result.recall_at_k == 0.0
    assert result.mean_reciprocal_rank_at_k == 0.0


def test_rejects_rankings_for_unknown_query_ids() -> None:
    dataset = load_retrieval_evaluation_dataset()

    with pytest.raises(ValueError, match="unknown query IDs"):
        evaluate_retrieval_dataset(dataset, {"not-in-dataset": ["chunk-study-breaks"]}, k=1)
