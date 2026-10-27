import pytest

from app.services.retrieval_evaluation import (
    mean_reciprocal_rank_at_k,
    recall_at_k,
)


def test_recall_at_k_uses_answerable_query_hit_rate() -> None:
    assert recall_at_k(
        [["a", "b"], ["c", "d"], ["x"]],
        [{"b"}, {"z"}, set()],
        k=2,
    ) == 0.5


def test_mean_reciprocal_rank_at_k_scores_first_relevant_rank_and_misses() -> None:
    assert mean_reciprocal_rank_at_k(
        [["a", "b"], ["c", "d"], ["x"]],
        [{"b"}, {"z"}, set()],
        k=2,
    ) == 0.25


@pytest.mark.parametrize("k", (0, -1, True, 1.5))
def test_metrics_reject_invalid_k(k: object) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        recall_at_k([["a"]], [{"a"}], k=k)


def test_metrics_reject_mismatched_query_counts() -> None:
    with pytest.raises(ValueError, match="query counts must match"):
        mean_reciprocal_rank_at_k([["a"]], [], k=1)


def test_metrics_require_an_answerable_query() -> None:
    with pytest.raises(ValueError, match="answerable query"):
        recall_at_k([["a"]], [set()], k=1)
