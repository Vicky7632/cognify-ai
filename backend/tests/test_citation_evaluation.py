from uuid import UUID

import pytest

from app.services.citation_evaluation import evaluate_citations
from app.services.grounded_answer_service import (
    GroundedAnswer,
    GroundedCitation,
)
from app.services.retrieval_evaluation_dataset import (
    RetrievalEvaluationDataset,
    load_retrieval_evaluation_dataset,
)

VALID_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_ID = UUID("00000000-0000-0000-0000-000000000002")
UNKNOWN_ID = UUID("00000000-0000-0000-0000-000000000003")


def _citation(segment_id: UUID) -> GroundedCitation:
    return GroundedCitation(
        document_id=UUID("00000000-0000-0000-0000-000000000010"),
        segment_id=segment_id,
        ordinal=0,
        page_number=None,
        original_filename="synthetic.txt",
        file_type="txt",
        score=0.9,
    )


def _answer(*citations: GroundedCitation) -> GroundedAnswer:
    return GroundedAnswer(
        answer="Synthetic answer.",
        citations=tuple(citations),
        insufficient_evidence=False,
    )


def test_evaluates_valid_ids_and_answerable_query_coverage() -> None:
    dataset = load_retrieval_evaluation_dataset()
    result = evaluate_citations(
        dataset,
        {"study-break-duration": [VALID_ID, OTHER_ID]},
        {"study-break-duration": _answer(_citation(VALID_ID))},
    )

    assert result.citation_validity == 1.0
    assert result.valid_citation_count == 1
    assert result.cited_id_count == 1
    assert result.citation_coverage == pytest.approx(1 / 3)
    assert result.covered_answerable_query_count == 1
    assert result.answerable_query_count == 3
    query = result.query_results[0]
    assert query.expected_relevant_chunk_ids == ("chunk-break-timing",)
    assert query.available_chunk_count == 2
    assert query.has_valid_citation is True


def test_unknown_ids_are_invalid_and_duplicate_citations_count_once() -> None:
    dataset = load_retrieval_evaluation_dataset()
    result = evaluate_citations(
        dataset,
        {"study-break-duration": [VALID_ID]},
        {
            "study-break-duration": _answer(
                _citation(VALID_ID),
                _citation(VALID_ID),
                _citation(UNKNOWN_ID),
            )
        },
    )

    query = result.query_results[0]
    assert query.unique_citation_count == 2
    assert query.valid_citation_count == 1
    assert query.invalid_citation_ids == (UNKNOWN_ID,)
    assert query.citation_validity == 0.5
    assert result.citation_validity == 0.5


def test_empty_citations_have_no_validity_denominator_and_no_coverage() -> None:
    dataset = load_retrieval_evaluation_dataset()
    result = evaluate_citations(dataset, {}, {})

    assert result.citation_validity is None
    assert result.cited_id_count == 0
    assert result.citation_coverage == 0.0
    assert result.covered_answerable_query_count == 0
    assert result.answerable_query_count == 3
    assert all(query.citation_validity is None for query in result.query_results)


def test_coverage_is_undefined_when_dataset_has_no_answerable_queries() -> None:
    dataset = RetrievalEvaluationDataset(
        dataset_version="1.0.0",
        schema_version="1.0",
        chunks=(),
        examples=(),
    )

    result = evaluate_citations(dataset, {}, {})

    assert result.citation_validity is None
    assert result.citation_coverage is None
    assert result.answerable_query_count == 0


def test_unanswerable_query_citations_do_not_count_toward_coverage() -> None:
    dataset = load_retrieval_evaluation_dataset()
    result = evaluate_citations(
        dataset,
        {"unsupported-exam-date": [VALID_ID]},
        {"unsupported-exam-date": _answer(_citation(VALID_ID))},
    )

    unanswerable = result.query_results[-1]
    assert unanswerable.sufficient_evidence is False
    assert unanswerable.has_valid_citation is True
    assert result.citation_validity == 1.0
    assert result.citation_coverage == 0.0
    assert result.covered_answerable_query_count == 0


def test_rejects_answer_for_unknown_query_id() -> None:
    dataset = load_retrieval_evaluation_dataset()

    with pytest.raises(ValueError, match="unknown query IDs"):
        evaluate_citations(dataset, {}, {"missing-query": _answer()})


def test_rejects_malformed_citation_without_segment_id() -> None:
    dataset = load_retrieval_evaluation_dataset()
    malformed_answer = GroundedAnswer(
        answer="Synthetic answer.",
        citations=(object(),),
        insufficient_evidence=False,
    )

    with pytest.raises(ValueError, match="no segment_id"):
        evaluate_citations(
            dataset,
            {},
            {"study-break-duration": malformed_answer},
        )


def test_rejects_unhashable_available_chunk_ids() -> None:
    dataset = load_retrieval_evaluation_dataset()

    with pytest.raises(ValueError, match="must be hashable"):
        evaluate_citations(
            dataset,
            {"study-break-duration": [["not-a-hashable-id"]]},
            {},
        )
