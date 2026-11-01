from uuid import UUID

import pytest

from app.services.claim_support_evaluation import (
    AnswerClaim,
    evaluate_claim_support,
)


def test_evaluates_claims_independently_and_reports_support_rate() -> None:
    claims = (
        AnswerClaim(
            claim_id="claim-one",
            text="First claim.",
            expected_supporting_chunk_ids=("chunk-a",),
            actual_cited_chunk_ids=("chunk-a", "chunk-extra"),
        ),
        AnswerClaim(
            claim_id="claim-two",
            text="Second claim.",
            expected_supporting_chunk_ids=("chunk-b", "chunk-c"),
            actual_cited_chunk_ids=("chunk-b",),
        ),
    )

    result = evaluate_claim_support(claims)

    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 1
    assert result.claim_count == 2
    assert result.claim_support_rate == 0.5
    assert result.claim_results[0].supported is True
    assert result.claim_results[0].missing_expected_chunk_ids == ()
    assert result.claim_results[1].supported is False
    assert result.claim_results[1].missing_expected_chunk_ids == ("chunk-c",)


def test_multiple_expected_sources_require_all_and_duplicates_are_deduplicated() -> None:
    claim = AnswerClaim(
        claim_id="multi-source",
        text="This claim requires both sources.",
        expected_supporting_chunk_ids=("chunk-a", "chunk-a", "chunk-b"),
        actual_cited_chunk_ids=("chunk-a", "chunk-a"),
    )

    result = evaluate_claim_support([claim])
    detail = result.claim_results[0]

    assert detail.expected_supporting_chunk_ids == ("chunk-a", "chunk-b")
    assert detail.actual_cited_chunk_ids == ("chunk-a",)
    assert detail.missing_expected_chunk_ids == ("chunk-b",)
    assert detail.supported is False


def test_missing_citations_make_claim_unsupported() -> None:
    result = evaluate_claim_support(
        [
            AnswerClaim(
                claim_id="uncited",
                text="An uncited claim.",
                expected_supporting_chunk_ids=("chunk-a",),
                actual_cited_chunk_ids=(),
            )
        ]
    )

    assert result.unsupported_claim_count == 1
    assert result.claim_support_rate == 0.0
    assert result.claim_results[0].missing_expected_chunk_ids == ("chunk-a",)


def test_empty_claims_return_undefined_rate_without_division_by_zero() -> None:
    result = evaluate_claim_support([])

    assert result.claim_results == ()
    assert result.claim_count == 0
    assert result.supported_claim_count == 0
    assert result.unsupported_claim_count == 0
    assert result.claim_support_rate is None


def test_accepts_existing_grounded_citation_segment_id_type() -> None:
    chunk_id = UUID("00000000-0000-0000-0000-000000000001")
    result = evaluate_claim_support(
        [
            AnswerClaim(
                claim_id="uuid-claim",
                text="A claim with a production-style segment ID.",
                expected_supporting_chunk_ids=(chunk_id,),
                actual_cited_chunk_ids=(chunk_id,),
            )
        ]
    )

    assert result.claim_results[0].supported is True
    assert result.claim_support_rate == 1.0


@pytest.mark.parametrize(
    "claims",
    [
        "not-a-sequence",
        [object()],
        [
            AnswerClaim(
                claim_id="",
                text="Claim.",
                expected_supporting_chunk_ids=("chunk-a",),
                actual_cited_chunk_ids=("chunk-a",),
            )
        ],
        [
            AnswerClaim(
                claim_id="claim",
                text=" ",
                expected_supporting_chunk_ids=("chunk-a",),
                actual_cited_chunk_ids=("chunk-a",),
            )
        ],
        [
            AnswerClaim(
                claim_id="claim",
                text="Claim.",
                expected_supporting_chunk_ids=(),
                actual_cited_chunk_ids=(),
            )
        ],
        [
            AnswerClaim(
                claim_id="claim",
                text="Claim.",
                expected_supporting_chunk_ids=(["not-hashable"],),
                actual_cited_chunk_ids=(),
            )
        ],
    ],
)
def test_rejects_malformed_claim_inputs(claims: object) -> None:
    with pytest.raises(ValueError):
        evaluate_claim_support(claims)


def test_rejects_duplicate_claim_ids() -> None:
    claim = AnswerClaim(
        claim_id="duplicate",
        text="Claim.",
        expected_supporting_chunk_ids=("chunk-a",),
        actual_cited_chunk_ids=("chunk-a",),
    )

    with pytest.raises(ValueError, match="claim_id values must be unique"):
        evaluate_claim_support([claim, claim])
