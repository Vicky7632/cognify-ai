from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.services.retrieval_evaluation_dataset import RetrievalEvaluationDataset


class AnswerDecision(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


class QueryOutcome(StrEnum):
    CORRECT_ANSWER = "correct_answer"
    CORRECT_ABSTENTION = "correct_abstention"
    INCORRECT_ANSWER = "incorrect_answer"
    UNNECESSARY_ABSTENTION = "unnecessary_abstention"
    MISSING_PREDICTION = "missing_prediction"


@dataclass(frozen=True, slots=True)
class AbstentionPrediction:
    query_id: str
    decision: AnswerDecision


@dataclass(frozen=True, slots=True)
class AbstentionQueryResult:
    query_id: str
    sufficient_evidence: bool
    decision: AnswerDecision | None
    outcome: QueryOutcome


@dataclass(frozen=True, slots=True)
class AbstentionEvaluation:
    query_results: tuple[AbstentionQueryResult, ...]
    abstention_precision: float | None
    abstention_precision_numerator: int
    abstention_precision_denominator: int
    abstention_recall: float | None
    abstention_recall_numerator: int
    abstention_recall_denominator: int
    answer_coverage: float | None
    answer_coverage_numerator: int
    answer_coverage_denominator: int
    incorrect_answer_rate: float | None
    incorrect_answer_rate_numerator: int
    incorrect_answer_rate_denominator: int


def evaluate_abstentions(
    dataset: RetrievalEvaluationDataset,
    predictions: Sequence[AbstentionPrediction],
) -> AbstentionEvaluation:
    """Evaluate answer/abstain decisions against dataset answerability labels."""
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise ValueError("predictions must be a sequence of AbstentionPrediction values.")

    known_query_ids = {example.query_id for example in dataset.examples}
    predictions_by_id: dict[str, AnswerDecision] = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, AbstentionPrediction):
            raise ValueError(
                f"predictions[{index}] must be an AbstentionPrediction."
            )
        if (
            not isinstance(prediction.query_id, str)
            or not prediction.query_id.strip()
        ):
            raise ValueError(f"predictions[{index}].query_id must be non-empty.")
        if prediction.query_id not in known_query_ids:
            raise ValueError(
                f"Prediction contains unknown query ID: {prediction.query_id}."
            )
        if prediction.query_id in predictions_by_id:
            raise ValueError(
                f"Duplicate prediction for query ID: {prediction.query_id}."
            )
        try:
            decision = AnswerDecision(prediction.decision)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"predictions[{index}].decision must be answered or abstained."
            ) from error
        predictions_by_id[prediction.query_id] = decision

    results: list[AbstentionQueryResult] = []
    for example in dataset.examples:
        decision = predictions_by_id.get(example.query_id)
        if decision is None:
            outcome = QueryOutcome.MISSING_PREDICTION
        elif example.sufficient_evidence:
            outcome = (
                QueryOutcome.CORRECT_ANSWER
                if decision is AnswerDecision.ANSWERED
                else QueryOutcome.UNNECESSARY_ABSTENTION
            )
        else:
            outcome = (
                QueryOutcome.CORRECT_ABSTENTION
                if decision is AnswerDecision.ABSTAINED
                else QueryOutcome.INCORRECT_ANSWER
            )
        results.append(
            AbstentionQueryResult(
                query_id=example.query_id,
                sufficient_evidence=example.sufficient_evidence,
                decision=decision,
                outcome=outcome,
            )
        )

    answerable_count = sum(example.sufficient_evidence for example in dataset.examples)
    unanswerable_count = len(dataset.examples) - answerable_count
    abstained_count = sum(
        result.decision is AnswerDecision.ABSTAINED for result in results
    )
    correctly_abstained_count = sum(
        result.outcome is QueryOutcome.CORRECT_ABSTENTION for result in results
    )
    incorrectly_answered_count = sum(
        result.outcome is QueryOutcome.INCORRECT_ANSWER for result in results
    )

    return AbstentionEvaluation(
        query_results=tuple(results),
        abstention_precision=(
            correctly_abstained_count / abstained_count
            if abstained_count
            else None
        ),
        abstention_precision_numerator=correctly_abstained_count,
        abstention_precision_denominator=abstained_count,
        abstention_recall=(
            correctly_abstained_count / unanswerable_count
            if unanswerable_count
            else None
        ),
        abstention_recall_numerator=correctly_abstained_count,
        abstention_recall_denominator=unanswerable_count,
        answer_coverage=(
            sum(
                result.outcome is QueryOutcome.CORRECT_ANSWER
                for result in results
            )
            / answerable_count
            if answerable_count
            else None
        ),
        answer_coverage_numerator=sum(
            result.outcome is QueryOutcome.CORRECT_ANSWER for result in results
        ),
        answer_coverage_denominator=answerable_count,
        incorrect_answer_rate=(
            incorrectly_answered_count / unanswerable_count
            if unanswerable_count
            else None
        ),
        incorrect_answer_rate_numerator=incorrectly_answered_count,
        incorrect_answer_rate_denominator=unanswerable_count,
    )
