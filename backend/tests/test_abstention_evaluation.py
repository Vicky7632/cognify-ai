import pytest

from app.services.abstention_evaluation import (
    AbstentionPrediction,
    AnswerDecision,
    QueryOutcome,
    evaluate_abstentions,
)
from app.services.retrieval_evaluation_dataset import (
    RetrievalEvaluationExample,
    RetrievalEvaluationDataset,
    load_retrieval_evaluation_dataset,
)


def test_evaluates_correct_answers_and_correct_abstention() -> None:
    dataset = load_retrieval_evaluation_dataset()
    predictions = [
        AbstentionPrediction("study-break-duration", AnswerDecision.ANSWERED),
        AbstentionPrediction("study-break-benefit", AnswerDecision.ANSWERED),
        AbstentionPrediction(
            "study-techniques-comparison",
            AnswerDecision.ANSWERED,
        ),
        AbstentionPrediction("unsupported-exam-date", AnswerDecision.ABSTAINED),
    ]

    result = evaluate_abstentions(dataset, predictions)

    assert result.abstention_precision == 1.0
    assert result.abstention_precision_numerator == 1
    assert result.abstention_precision_denominator == 1
    assert result.abstention_recall == 1.0
    assert result.abstention_recall_numerator == 1
    assert result.abstention_recall_denominator == 1
    assert result.answer_coverage == 1.0
    assert result.answer_coverage_numerator == 3
    assert result.answer_coverage_denominator == 3
    assert result.incorrect_answer_rate == 0.0
    assert result.incorrect_answer_rate_numerator == 0
    assert result.incorrect_answer_rate_denominator == 1
    assert result.query_results[-1].outcome is QueryOutcome.CORRECT_ABSTENTION


def test_answering_unanswerable_query_increases_incorrect_answer_rate() -> None:
    dataset = load_retrieval_evaluation_dataset()
    result = evaluate_abstentions(
        dataset,
        [AbstentionPrediction("unsupported-exam-date", AnswerDecision.ANSWERED)],
    )

    assert result.abstention_precision is None
    assert result.abstention_recall == 0.0
    assert result.answer_coverage == 0.0
    assert result.incorrect_answer_rate == 1.0
    assert result.query_results[-1].outcome is QueryOutcome.INCORRECT_ANSWER


def test_unnecessary_abstention_is_not_answer_coverage() -> None:
    dataset = load_retrieval_evaluation_dataset()
    result = evaluate_abstentions(
        dataset,
        [
            AbstentionPrediction(
                "study-break-duration",
                AnswerDecision.ABSTAINED,
            )
        ],
    )

    assert result.abstention_precision == 0.0
    assert result.abstention_recall == 0.0
    assert result.answer_coverage == 0.0
    assert result.incorrect_answer_rate == 0.0
    assert (
        result.query_results[0].outcome
        is QueryOutcome.UNNECESSARY_ABSTENTION
    )


def test_missing_prediction_is_explicit_and_never_a_correct_abstention() -> None:
    dataset = load_retrieval_evaluation_dataset()
    result = evaluate_abstentions(dataset, [])

    assert all(
        query.outcome is QueryOutcome.MISSING_PREDICTION
        for query in result.query_results
    )
    assert result.abstention_precision is None
    assert result.abstention_recall == 0.0
    assert result.answer_coverage == 0.0
    assert result.incorrect_answer_rate == 0.0


def test_rejects_duplicate_and_unknown_query_ids() -> None:
    dataset = load_retrieval_evaluation_dataset()
    prediction = AbstentionPrediction(
        "study-break-duration",
        AnswerDecision.ANSWERED,
    )

    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate_abstentions(dataset, [prediction, prediction])
    with pytest.raises(ValueError, match="unknown query ID"):
        evaluate_abstentions(
            dataset,
            [AbstentionPrediction("not-in-dataset", AnswerDecision.ABSTAINED)],
        )


def test_rejects_invalid_decisions_and_prediction_types() -> None:
    dataset = load_retrieval_evaluation_dataset()

    with pytest.raises(ValueError, match="must be answered or abstained"):
        evaluate_abstentions(
            dataset,
            [AbstentionPrediction("study-break-duration", "maybe")],
        )
    with pytest.raises(ValueError, match="must be an AbstentionPrediction"):
        evaluate_abstentions(dataset, [object()])


def test_empty_dataset_returns_none_for_zero_denominator_metrics() -> None:
    dataset = RetrievalEvaluationDataset(
        dataset_version="1.0.0",
        schema_version="1.0",
        chunks=(),
        examples=(),
    )

    result = evaluate_abstentions(dataset, [])

    assert result.query_results == ()
    assert result.abstention_precision is None
    assert result.abstention_recall is None
    assert result.answer_coverage is None
    assert result.incorrect_answer_rate is None
    assert result.abstention_precision_denominator == 0
    assert result.abstention_recall_denominator == 0
    assert result.answer_coverage_denominator == 0
    assert result.incorrect_answer_rate_denominator == 0


def test_metrics_are_none_when_their_population_is_absent() -> None:
    answerable_dataset = RetrievalEvaluationDataset(
        dataset_version="1.0.0",
        schema_version="1.0",
        chunks=(),
        examples=(
            RetrievalEvaluationExample(
                query_id="answerable",
                query="A query with sufficient evidence.",
                relevant_chunk_ids=("chunk-a",),
                sufficient_evidence=True,
                reference_answer="A synthetic answer.",
                key_facts=(),
            ),
        ),
    )
    unanswerable_dataset = RetrievalEvaluationDataset(
        dataset_version="1.0.0",
        schema_version="1.0",
        chunks=(),
        examples=(
            RetrievalEvaluationExample(
                query_id="unanswerable",
                query="A query without sufficient evidence.",
                relevant_chunk_ids=(),
                sufficient_evidence=False,
                reference_answer=None,
                key_facts=(),
            ),
        ),
    )

    answerable_result = evaluate_abstentions(
        answerable_dataset,
        [AbstentionPrediction("answerable", AnswerDecision.ANSWERED)],
    )
    unanswerable_result = evaluate_abstentions(
        unanswerable_dataset,
        [AbstentionPrediction("unanswerable", AnswerDecision.ABSTAINED)],
    )

    assert answerable_result.abstention_precision is None
    assert answerable_result.abstention_recall is None
    assert answerable_result.answer_coverage == 1.0
    assert answerable_result.incorrect_answer_rate is None
    assert unanswerable_result.abstention_precision == 1.0
    assert unanswerable_result.abstention_recall == 1.0
    assert unanswerable_result.answer_coverage is None
    assert unanswerable_result.incorrect_answer_rate == 0.0
