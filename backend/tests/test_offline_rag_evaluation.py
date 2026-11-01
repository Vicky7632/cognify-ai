import json
from uuid import UUID

import pytest

from app.services.abstention_evaluation import AnswerDecision, QueryOutcome
from app.services.claim_support_evaluation import AnswerClaim
from app.services.grounded_answer_service import (
    GroundedAnswer,
    GroundedCitation,
)
from app.services.offline_rag_evaluation import (
    OfflineRAGEvaluationInput,
    QueryEvaluationInput,
    evaluate_offline_rag,
)

VALID_SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
UNKNOWN_SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000002")
DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000003")


def _citation(segment_id: UUID) -> GroundedCitation:
    return GroundedCitation(
        document_id=DOCUMENT_ID,
        segment_id=segment_id,
        ordinal=0,
        page_number=None,
        original_filename="synthetic.txt",
        file_type="txt",
        score=0.9,
    )


def test_orchestrates_all_evaluators_and_serializes_deterministically() -> None:
    result = evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="study-break-duration",
                    retrieved_chunk_ids=(
                        "chunk-study-breaks",
                        "chunk-break-timing",
                    ),
                    available_chunk_ids=(VALID_SEGMENT_ID,),
                    answer=GroundedAnswer(
                        answer="A synthetic answer.",
                        citations=(
                            _citation(VALID_SEGMENT_ID),
                            _citation(UNKNOWN_SEGMENT_ID),
                        ),
                        insufficient_evidence=False,
                    ),
                    claims=(
                        AnswerClaim(
                            claim_id="claim-one",
                            text="A supported synthetic claim.",
                            expected_supporting_chunk_ids=("chunk-break-timing",),
                            actual_cited_chunk_ids=("chunk-break-timing",),
                        ),
                    ),
                    decision=AnswerDecision.ANSWERED,
                ),
                QueryEvaluationInput(
                    query_id="study-break-benefit",
                    retrieved_chunk_ids=("chunk-study-breaks",),
                    claims=(
                        AnswerClaim(
                            claim_id="claim-one",
                            text="An unsupported synthetic claim.",
                            expected_supporting_chunk_ids=("chunk-study-breaks",),
                            actual_cited_chunk_ids=(),
                        ),
                    ),
                    decision=AnswerDecision.ANSWERED,
                ),
                QueryEvaluationInput(
                    query_id="study-techniques-comparison",
                    retrieved_chunk_ids=(
                        "chunk-active-recall",
                        "chunk-spaced-review",
                    ),
                    decision=AnswerDecision.ANSWERED,
                ),
                QueryEvaluationInput(
                    query_id="unsupported-exam-date",
                    decision=AnswerDecision.ABSTAINED,
                ),
            ),
            k=2,
        )
    )

    assert result.dataset_version == "1.0.0"
    assert result.retrieval.recall_at_k == 1.0
    assert result.retrieval.mean_reciprocal_rank_at_k == pytest.approx(5 / 6)
    assert result.citations.citation_validity == 0.5
    assert result.citations.citation_coverage == pytest.approx(1 / 3)
    assert result.citations.cited_id_count == 2
    assert result.citations.answerable_query_count == 3
    assert result.claim_support.claim_support_rate == 0.5
    assert result.claim_support.claim_count == 2
    assert result.claim_support.claim_results[0].claim_id == "claim-one"
    assert result.claim_support.claim_results[1].claim_id == "claim-one"
    assert len(result.claim_support_by_query) == 2
    assert result.abstention.abstention_precision == 1.0
    assert result.abstention.abstention_recall == 1.0
    assert result.abstention.answer_coverage == 1.0
    assert result.abstention.incorrect_answer_rate == 0.0
    assert result.missing_inputs.query_ids == ()
    assert result.missing_inputs.answer_query_ids == (
        "study-break-benefit",
        "study-techniques-comparison",
        "unsupported-exam-date",
    )

    serialized = result.to_dict()
    assert json.dumps(serialized, sort_keys=True) == json.dumps(
        result.to_dict(),
        sort_keys=True,
    )
    assert serialized["citations"]["query_results"][0]["invalid_citation_ids"] == [
        str(UNKNOWN_SEGMENT_ID)
    ]


def test_partial_inputs_are_reported_and_missing_decisions_stay_missing() -> None:
    result = evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="unsupported-exam-date",
                    decision=AnswerDecision.ANSWERED,
                ),
            )
        )
    )

    assert result.missing_inputs.query_ids == (
        "study-break-duration",
        "study-break-benefit",
        "study-techniques-comparison",
    )
    assert result.missing_inputs.decision_query_ids == (
        "study-break-duration",
        "study-break-benefit",
        "study-techniques-comparison",
    )
    assert result.abstention.query_results[-1].outcome is QueryOutcome.INCORRECT_ANSWER
    assert all(
        query.outcome is QueryOutcome.MISSING_PREDICTION
        for query in result.abstention.query_results[:-1]
    )
    assert result.abstention.incorrect_answer_rate == 1.0
    assert result.claim_support.claim_support_rate is None


def test_empty_input_is_explicit_across_all_evaluators() -> None:
    result = evaluate_offline_rag(OfflineRAGEvaluationInput(queries=()))

    assert result.missing_inputs.query_ids == (
        "study-break-duration",
        "study-break-benefit",
        "study-techniques-comparison",
        "unsupported-exam-date",
    )
    assert result.retrieval.recall_at_k == 0.0
    assert result.citations.citation_validity is None
    assert result.citations.citation_coverage == 0.0
    assert result.claim_support.claim_support_rate is None
    assert all(
        query.outcome is QueryOutcome.MISSING_PREDICTION
        for query in result.abstention.query_results
    )
    assert result.abstention.abstention_precision is None


def test_rejects_duplicate_and_unknown_query_ids() -> None:
    query = QueryEvaluationInput(query_id="study-break-duration")

    with pytest.raises(ValueError, match="Duplicate query ID"):
        evaluate_offline_rag(OfflineRAGEvaluationInput(queries=(query, query)))
    with pytest.raises(ValueError, match="Unknown query ID"):
        evaluate_offline_rag(
            OfflineRAGEvaluationInput(
                queries=(QueryEvaluationInput(query_id="not-in-dataset"),)
            )
        )


def test_rejects_invalid_retrieval_ids_and_k() -> None:
    with pytest.raises(ValueError, match="unknown retrieved chunk IDs"):
        evaluate_offline_rag(
            OfflineRAGEvaluationInput(
                queries=(
                    QueryEvaluationInput(
                        query_id="study-break-duration",
                        retrieved_chunk_ids=("not-in-dataset",),
                    ),
                )
            )
        )
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluate_offline_rag(OfflineRAGEvaluationInput(queries=(), k=0))
