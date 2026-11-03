import json
from pathlib import Path
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
    OfflineRAGEvaluationResult,
    QueryEvaluationInput,
    evaluate_offline_rag,
)
from app.services.retrieval_evaluation_dataset import DEFAULT_DATASET_PATH

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


def _claim(
    actual_cited_chunk_ids: tuple[str, ...] = ("chunk-break-timing",),
) -> AnswerClaim:
    return AnswerClaim(
        claim_id="claim-one",
        text="A synthetic claim.",
        expected_supporting_chunk_ids=("chunk-break-timing",),
        actual_cited_chunk_ids=actual_cited_chunk_ids,
    )


def _evaluate_source_consistency(
    mapping: dict[str, UUID],
    *,
    citation_ids: tuple[UUID, ...],
    actual_cited_chunk_ids: tuple[str, ...] = ("chunk-break-timing",),
) -> OfflineRAGEvaluationResult:
    return evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="study-break-duration",
                    answer=GroundedAnswer(
                        answer="A synthetic answer.",
                        citations=tuple(
                            _citation(segment_id) for segment_id in citation_ids
                        ),
                        insufficient_evidence=False,
                    ),
                    claims=(_claim(actual_cited_chunk_ids),),
                ),
            ),
            chunk_id_to_segment_id=mapping,
        )
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
    assert result.citation_source_consistency is None
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


@pytest.mark.parametrize(
    ("decision", "insufficient_evidence"),
    [
        (AnswerDecision.ANSWERED, True),
        (AnswerDecision.ABSTAINED, False),
    ],
)
def test_rejects_decision_conflicting_with_answer_evidence(
    decision: AnswerDecision,
    insufficient_evidence: bool,
) -> None:
    with pytest.raises(ValueError, match="decision conflicts with answer"):
        evaluate_offline_rag(
            OfflineRAGEvaluationInput(
                queries=(
                    QueryEvaluationInput(
                        query_id="study-break-duration",
                        answer=GroundedAnswer(
                            answer="A synthetic answer.",
                            citations=(),
                            insufficient_evidence=insufficient_evidence,
                        ),
                        decision=decision,
                    ),
                )
            )
        )


@pytest.mark.parametrize("insufficient_evidence", [None, 1, "true"])
def test_rejects_non_boolean_answer_insufficient_evidence(
    insufficient_evidence: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="answer.insufficient_evidence must be a boolean",
    ):
        evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="study-break-duration",
                    answer=GroundedAnswer(
                        answer="A synthetic answer.",
                        citations=(),
                        insufficient_evidence=insufficient_evidence,
                    ),
                ),
            )
        )
        )


@pytest.mark.parametrize(
    ("decision", "insufficient_evidence", "expected_outcome"),
    [
        (AnswerDecision.ANSWERED, False, QueryOutcome.CORRECT_ANSWER),
        (AnswerDecision.ABSTAINED, True, QueryOutcome.UNNECESSARY_ABSTENTION),
    ],
)
def test_accepts_decision_consistent_with_answer_evidence(
    decision: AnswerDecision,
    insufficient_evidence: bool,
    expected_outcome: QueryOutcome,
) -> None:
    result = evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="study-break-duration",
                    answer=GroundedAnswer(
                        answer="A synthetic answer.",
                        citations=(),
                        insufficient_evidence=insufficient_evidence,
                    ),
                    decision=decision,
                ),
            )
        )
    )

    assert result.abstention.query_results[0].outcome is expected_outcome


def test_missing_decision_remains_missing_with_insufficient_evidence_answer() -> None:
    result = evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="study-break-duration",
                    answer=GroundedAnswer(
                        answer="Insufficient evidence.",
                        citations=(),
                        insufficient_evidence=True,
                    ),
                ),
            )
        )
    )

    assert (
        result.abstention.query_results[0].outcome
        is QueryOutcome.MISSING_PREDICTION
    )
    assert result.missing_inputs.decision_query_ids == (
        "study-break-duration",
        "study-break-benefit",
        "study-techniques-comparison",
        "unsupported-exam-date",
    )


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


def test_unified_evaluation_supports_all_unanswerable_dataset(
    tmp_path: Path,
) -> None:
    payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    for example in payload["examples"]:
        example["sufficient_evidence"] = False
        example["relevant_chunk_ids"] = []
        example.pop("reference_answer", None)
        example.pop("key_facts", None)
    dataset_path = tmp_path / "all-unanswerable.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="study-break-duration",
                    decision=AnswerDecision.ABSTAINED,
                ),
                QueryEvaluationInput(
                    query_id="study-break-benefit",
                    decision=AnswerDecision.ABSTAINED,
                ),
                QueryEvaluationInput(
                    query_id="study-techniques-comparison",
                    decision=AnswerDecision.ANSWERED,
                ),
                QueryEvaluationInput(
                    query_id="unsupported-exam-date",
                    decision=AnswerDecision.ABSTAINED,
                ),
            )
        ),
        dataset_path=dataset_path,
    )

    assert result.retrieval.total_query_count == 4
    assert result.retrieval.answerable_query_count == 0
    assert result.retrieval.unanswerable_query_count == 4
    assert result.retrieval.recall_at_k is None
    assert result.retrieval.mean_reciprocal_rank_at_k is None
    assert result.abstention.abstention_precision == 1.0
    assert result.abstention.abstention_precision_numerator == 3
    assert result.abstention.abstention_precision_denominator == 3
    assert result.abstention.abstention_recall == 0.75
    assert result.abstention.abstention_recall_numerator == 3
    assert result.abstention.abstention_recall_denominator == 4
    assert result.abstention.incorrect_answer_rate == 0.25
    assert result.abstention.incorrect_answer_rate_numerator == 1
    assert result.abstention.incorrect_answer_rate_denominator == 4

    serialized = result.to_dict()
    assert json.loads(json.dumps(serialized))["retrieval"]["recall_at_k"] is None
    assert serialized["abstention"]["abstention_precision"] == 1.0


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


def test_source_consistency_matches_mapped_claim_sources_to_answer_citations() -> None:
    result = _evaluate_source_consistency(
        {"chunk-break-timing": VALID_SEGMENT_ID},
        citation_ids=(VALID_SEGMENT_ID,),
    )

    consistency = result.citation_source_consistency
    assert consistency is not None
    assert consistency.checked_query_count == 1
    assert consistency.mismatched_query_count == 0
    assert consistency.unmapped_citation_count == 0
    assert consistency.query_results[0].annotated_segment_ids == (VALID_SEGMENT_ID,)
    assert consistency.query_results[0].answer_segment_ids == (VALID_SEGMENT_ID,)


def test_source_consistency_reports_claim_source_missing_from_answer() -> None:
    result = _evaluate_source_consistency(
        {"chunk-break-timing": VALID_SEGMENT_ID},
        citation_ids=(UNKNOWN_SEGMENT_ID,),
    )

    consistency = result.citation_source_consistency
    assert consistency is not None
    query = consistency.query_results[0]
    assert query.missing_answer_segment_ids == (VALID_SEGMENT_ID,)
    assert query.unannotated_answer_segment_ids == (UNKNOWN_SEGMENT_ID,)
    assert query.unmapped_answer_segment_ids == (UNKNOWN_SEGMENT_ID,)
    assert consistency.mismatched_query_count == 1


def test_source_consistency_requires_mapping_for_claim_source_ids() -> None:
    with pytest.raises(ValueError, match="missing from chunk_id_to_segment_id"):
        _evaluate_source_consistency(
            {"chunk-study-breaks": VALID_SEGMENT_ID},
            citation_ids=(),
        )


def test_source_consistency_rejects_unknown_mapping_keys() -> None:
    with pytest.raises(ValueError, match="unknown chunk IDs"):
        _evaluate_source_consistency(
            {
                "chunk-break-timing": VALID_SEGMENT_ID,
                "not-a-dataset-chunk": UNKNOWN_SEGMENT_ID,
            },
            citation_ids=(VALID_SEGMENT_ID,),
        )


def test_source_consistency_rejects_duplicate_segment_uuid_targets() -> None:
    with pytest.raises(ValueError, match="duplicate segment UUIDs"):
        _evaluate_source_consistency(
            {
                "chunk-break-timing": VALID_SEGMENT_ID,
                "chunk-study-breaks": VALID_SEGMENT_ID,
            },
            citation_ids=(VALID_SEGMENT_ID,),
        )


def test_source_consistency_reports_unmapped_answer_citation_uuid() -> None:
    result = _evaluate_source_consistency(
        {"chunk-break-timing": VALID_SEGMENT_ID},
        citation_ids=(VALID_SEGMENT_ID, UNKNOWN_SEGMENT_ID),
    )

    consistency = result.citation_source_consistency
    assert consistency is not None
    query = consistency.query_results[0]
    assert query.unmapped_answer_segment_ids == (UNKNOWN_SEGMENT_ID,)
    assert consistency.unmapped_citation_count == 1


def test_source_consistency_deduplicates_citations_and_serializes_deterministically() -> None:
    mapping = {"chunk-break-timing": VALID_SEGMENT_ID}
    first = _evaluate_source_consistency(
        mapping,
        citation_ids=(VALID_SEGMENT_ID, VALID_SEGMENT_ID),
    ).to_dict()
    second = _evaluate_source_consistency(
        mapping,
        citation_ids=(VALID_SEGMENT_ID, VALID_SEGMENT_ID),
    ).to_dict()

    assert first == second
    assert first["citation_source_consistency"]["query_results"][0][
        "answer_segment_ids"
    ] == [str(VALID_SEGMENT_ID)]


def test_source_consistency_rejects_non_uuid_mapping_values() -> None:
    with pytest.raises(ValueError, match="values must be segment UUIDs"):
        _evaluate_source_consistency(
            {"chunk-break-timing": "not-a-uuid"},
            citation_ids=(VALID_SEGMENT_ID,),
        )


def test_source_consistency_rejects_non_mapping_source_id_mapping() -> None:
    with pytest.raises(ValueError, match="chunk_id_to_segment_id must be a mapping"):
        evaluate_offline_rag(
            OfflineRAGEvaluationInput(
                queries=(),
                chunk_id_to_segment_id=[],
            )
        )


def test_source_consistency_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(ValueError, match="keys must be dataset chunk ID strings"):
        evaluate_offline_rag(
            OfflineRAGEvaluationInput(
                queries=(),
                chunk_id_to_segment_id={1: VALID_SEGMENT_ID},
            )
        )


def test_source_consistency_rejects_non_uuid_answer_citation_segment_id() -> None:
    with pytest.raises(ValueError, match="must have a segment UUID"):
        _evaluate_source_consistency(
            {"chunk-break-timing": VALID_SEGMENT_ID},
            citation_ids=("not-a-uuid",),
        )


def test_source_consistency_reports_mapped_unannotated_citation() -> None:
    result = _evaluate_source_consistency(
        {"chunk-break-timing": VALID_SEGMENT_ID},
        citation_ids=(VALID_SEGMENT_ID,),
        actual_cited_chunk_ids=(),
    )

    consistency = result.citation_source_consistency
    assert consistency is not None
    query = consistency.query_results[0]
    assert query.unannotated_answer_segment_ids == (VALID_SEGMENT_ID,)
    assert query.unmapped_answer_segment_ids == ()
    assert consistency.mismatched_query_count == 1


def test_source_consistency_deduplicates_duplicate_claim_source_ids() -> None:
    result = _evaluate_source_consistency(
        {"chunk-break-timing": VALID_SEGMENT_ID},
        citation_ids=(),
        actual_cited_chunk_ids=("chunk-break-timing", "chunk-break-timing"),
    )

    consistency = result.citation_source_consistency
    assert consistency is not None
    query = consistency.query_results[0]
    assert consistency.checked_query_count == 1
    assert consistency.mismatched_query_count == 1
    assert query.annotated_segment_ids == (VALID_SEGMENT_ID,)
    assert query.missing_answer_segment_ids == (VALID_SEGMENT_ID,)


def test_unified_report_includes_all_metrics_and_source_id_consistency() -> None:
    result = evaluate_offline_rag(
        OfflineRAGEvaluationInput(
            queries=(
                QueryEvaluationInput(
                    query_id="study-break-duration",
                    retrieved_chunk_ids=("chunk-break-timing",),
                    available_chunk_ids=(VALID_SEGMENT_ID,),
                    answer=GroundedAnswer(
                        answer="A synthetic answer.",
                        citations=(
                            _citation(VALID_SEGMENT_ID),
                            _citation(UNKNOWN_SEGMENT_ID),
                        ),
                        insufficient_evidence=False,
                    ),
                    claims=(_claim(),),
                    decision=AnswerDecision.ANSWERED,
                ),
                QueryEvaluationInput(
                    query_id="unsupported-exam-date",
                    decision=AnswerDecision.ABSTAINED,
                ),
                QueryEvaluationInput(
                    query_id="study-break-benefit",
                    retrieved_chunk_ids=("chunk-study-breaks",),
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
            ),
            k=1,
            chunk_id_to_segment_id={
                "chunk-break-timing": VALID_SEGMENT_ID,
            },
        )
    )

    consistency = result.citation_source_consistency
    assert consistency is not None
    assert result.retrieval.recall_at_k == 1.0
    assert result.retrieval.mean_reciprocal_rank_at_k == 1.0
    assert result.citations.citation_validity == 0.5
    assert result.citations.citation_coverage == pytest.approx(1 / 3)
    assert result.claim_support.claim_support_rate == 1.0
    assert result.abstention.abstention_precision == 1.0
    assert result.abstention.abstention_recall == 1.0
    assert consistency.checked_query_count == 1
    assert consistency.mismatched_query_count == 1
    assert consistency.unmapped_citation_count == 1
    assert consistency.query_results[0].query_id == "study-break-duration"
    assert consistency.query_results[0].annotated_segment_ids == (VALID_SEGMENT_ID,)
    assert consistency.query_results[0].unannotated_answer_segment_ids == (
        UNKNOWN_SEGMENT_ID,
    )

    serialized = result.to_dict()
    assert json.loads(json.dumps(serialized)) == serialized
    assert serialized["citation_source_consistency"]["checked_query_count"] == 1
