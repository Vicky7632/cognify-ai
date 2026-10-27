from collections.abc import Collection, Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def recall_at_k(
    retrieved_ids: Sequence[Sequence[T]],
    relevant_ids: Sequence[Collection[T]],
    *,
    k: int,
) -> float:
    """Return answerable queries with a relevant result in the top k."""
    queries = _answerable_queries(retrieved_ids, relevant_ids, k)
    return sum(
        bool(set(ranked[:k]) & relevant)
        for ranked, relevant in queries
    ) / len(queries)


def mean_reciprocal_rank_at_k(
    retrieved_ids: Sequence[Sequence[T]],
    relevant_ids: Sequence[Collection[T]],
    *,
    k: int,
) -> float:
    """Return mean reciprocal rank of the first relevant result in top k."""
    queries = _answerable_queries(retrieved_ids, relevant_ids, k)
    reciprocal_ranks = []
    for ranked, relevant in queries:
        rank = next(
            (
                position
                for position, result_id in enumerate(ranked[:k], start=1)
                if result_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _answerable_queries(
    retrieved_ids: Sequence[Sequence[T]],
    relevant_ids: Sequence[Collection[T]],
    k: int,
) -> tuple[tuple[Sequence[T], set[T]], ...]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError("Retrieved and relevant query counts must match.")

    queries = tuple(
        (ranked, set(relevant))
        for ranked, relevant in zip(retrieved_ids, relevant_ids, strict=True)
        if relevant
    )
    if not queries:
        raise ValueError("At least one answerable query is required.")
    return queries
