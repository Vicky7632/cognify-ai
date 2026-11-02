from dataclasses import replace

import pytest

from app.services.retrieval_dataset_evaluator import (
    evaluate_retrieval_dataset,
    run_retrieval_dataset_evaluation,
)
from app.services.retrieval_evaluation_dataset import (
    DEFAULT_DATASET_PATH,
    EvaluationChunk,
    RetrievalEvaluationDataset,
    RetrievalEvaluationExample,
    load_retrieval_evaluation_dataset,
)


def _dataset_with_examples(
    *examples: RetrievalEvaluationExample,
) -> RetrievalEvaluationDataset:
    return RetrievalEvaluationDataset(
        dataset_version="1.0.0",
        schema_version="1.0",
        chunks=(EvaluationChunk(chunk_id="chunk-1", text="Relevant evidence."),),
        examples=examples,
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


def test_runner_loads_versioned_dataset_and_returns_deterministic_metrics() -> None:
    ranked_chunk_ids = {
        "study-break-duration": [
            "chunk-study-breaks",
            "chunk-break-timing",
        ],
        "study-break-benefit": ["chunk-study-breaks"],
        "study-techniques-comparison": [
            "chunk-active-recall",
            "chunk-spaced-review",
        ],
    }

    first = run_retrieval_dataset_evaluation(ranked_chunk_ids, k=2)
    second = run_retrieval_dataset_evaluation(
        ranked_chunk_ids,
        k=2,
        dataset_path=DEFAULT_DATASET_PATH,
    )

    assert first == second
    assert first.k == 2
    assert first.answerable_query_count == 3
    assert first.total_query_count == 4
    assert first.recall_at_k == 1.0
    assert first.mean_reciprocal_rank_at_k == pytest.approx(5 / 6)


def test_empty_rankings_and_empty_results_are_scored_as_misses() -> None:
    result = run_retrieval_dataset_evaluation(
        {
            "study-break-duration": [],
            "study-break-benefit": (),
        },
        k=3,
    )

    assert result.answerable_query_count == 3
    assert result.total_query_count == 4
    assert result.recall_at_k == 0.0
    assert result.mean_reciprocal_rank_at_k == 0.0


def test_rankings_without_relevant_chunks_count_as_misses() -> None:
    result = run_retrieval_dataset_evaluation(
        {"study-break-duration": ["chunk-study-breaks"]},
        k=1,
    )

    assert result.recall_at_k == 0.0
    assert result.mean_reciprocal_rank_at_k == 0.0


def test_rejects_rankings_for_unknown_query_ids() -> None:
    dataset = load_retrieval_evaluation_dataset()

    with pytest.raises(ValueError, match="unknown query IDs"):
        evaluate_retrieval_dataset(dataset, {"not-in-dataset": ["chunk-study-breaks"]}, k=1)


@pytest.mark.parametrize(
    ("ranking", "message"),
    [
        (["not-in-dataset"], "unknown chunk IDs"),
        ([""], "non-empty chunk IDs"),
        ([None], "non-empty chunk IDs"),
        (["chunk-study-breaks", "chunk-study-breaks"], "unique chunk IDs"),
        ("chunk-study-breaks", "sequence of chunk IDs"),
    ],
)
def test_rejects_invalid_ranked_records(
    ranking: object,
    message: str,
) -> None:
    dataset = load_retrieval_evaluation_dataset()

    with pytest.raises(ValueError, match=message):
        evaluate_retrieval_dataset(
            dataset,
            {"study-break-duration": ranking},
            k=1,
        )


def test_rejects_dataset_without_answerable_queries() -> None:
    dataset = load_retrieval_evaluation_dataset()
    unanswerable_dataset = replace(
        dataset,
        examples=tuple(
            replace(
                example,
                relevant_chunk_ids=(),
                sufficient_evidence=False,
                reference_answer=None,
                key_facts=(),
            )
            for example in dataset.examples
        ),
    )

    with pytest.raises(ValueError, match="At least one answerable query"):
        evaluate_retrieval_dataset(unanswerable_dataset, {}, k=1)


def test_rejects_directly_constructed_answerable_example_without_relevant_chunks() -> None:
    dataset = _dataset_with_examples(
        RetrievalEvaluationExample(
            query_id="answerable",
            query="What is relevant?",
            relevant_chunk_ids=(),
            sufficient_evidence=True,
            reference_answer="Relevant evidence.",
            key_facts=(),
        )
    )

    with pytest.raises(
        ValueError,
        match="Answerable query answerable must have at least one relevant chunk ID",
    ):
        evaluate_retrieval_dataset(dataset, {}, k=1)


def test_accepts_directly_constructed_answerable_example_with_relevant_chunks() -> None:
    dataset = _dataset_with_examples(
        RetrievalEvaluationExample(
            query_id="answerable",
            query="What is relevant?",
            relevant_chunk_ids=("chunk-1",),
            sufficient_evidence=True,
            reference_answer="Relevant evidence.",
            key_facts=(),
        )
    )

    result = evaluate_retrieval_dataset(dataset, {}, k=1)

    assert result.answerable_query_count == 1
    assert result.recall_at_k == 0.0
    assert result.mean_reciprocal_rank_at_k == 0.0


def test_accepts_directly_constructed_unanswerable_example_without_relevant_chunks() -> None:
    answerable = RetrievalEvaluationExample(
        query_id="answerable",
        query="What is relevant?",
        relevant_chunk_ids=("chunk-1",),
        sufficient_evidence=True,
        reference_answer="Relevant evidence.",
        key_facts=(),
    )
    unanswerable = RetrievalEvaluationExample(
        query_id="unanswerable",
        query="What is unavailable?",
        relevant_chunk_ids=(),
        sufficient_evidence=False,
        reference_answer=None,
        key_facts=(),
    )
    dataset = _dataset_with_examples(answerable, unanswerable)

    result = evaluate_retrieval_dataset(dataset, {}, k=1)

    assert result.answerable_query_count == 1
    assert result.total_query_count == 2
