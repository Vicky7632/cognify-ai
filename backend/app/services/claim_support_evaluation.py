from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

ChunkId: TypeAlias = Hashable


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    claim_id: str
    text: str
    expected_supporting_chunk_ids: tuple[ChunkId, ...]
    actual_cited_chunk_ids: tuple[ChunkId, ...]


@dataclass(frozen=True, slots=True)
class ClaimSupportResult:
    claim_id: str
    text: str
    expected_supporting_chunk_ids: tuple[ChunkId, ...]
    actual_cited_chunk_ids: tuple[ChunkId, ...]
    missing_expected_chunk_ids: tuple[ChunkId, ...]
    supported: bool


@dataclass(frozen=True, slots=True)
class ClaimSupportEvaluation:
    claim_results: tuple[ClaimSupportResult, ...]
    supported_claim_count: int
    unsupported_claim_count: int
    claim_count: int
    claim_support_rate: float | None


def evaluate_claim_support(
    claims: Sequence[AnswerClaim],
) -> ClaimSupportEvaluation:
    """Compare annotated expected source IDs with supplied cited IDs.

    This deterministic ID comparison does not assess textual entailment or
    factual correctness.
    """
    if isinstance(claims, (str, bytes)) or not isinstance(claims, Sequence):
        raise ValueError("claims must be a sequence of AnswerClaim values.")

    claim_ids: set[str] = set()
    results: list[ClaimSupportResult] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, AnswerClaim):
            raise ValueError(f"claims[{index}] must be an AnswerClaim.")
        if not isinstance(claim.claim_id, str) or not claim.claim_id.strip():
            raise ValueError(f"claims[{index}].claim_id must be non-empty.")
        if claim.claim_id in claim_ids:
            raise ValueError("claim_id values must be unique.")
        claim_ids.add(claim.claim_id)
        if not isinstance(claim.text, str) or not claim.text.strip():
            raise ValueError(f"claims[{index}].text must be non-empty.")

        expected = _unique_ids(
            claim.expected_supporting_chunk_ids,
            f"claims[{index}].expected_supporting_chunk_ids",
        )
        if not expected:
            raise ValueError(
                f"claims[{index}] must have at least one expected supporting chunk ID."
            )
        cited = _unique_ids(
            claim.actual_cited_chunk_ids,
            f"claims[{index}].actual_cited_chunk_ids",
        )
        cited_set = set(cited)
        missing = tuple(chunk_id for chunk_id in expected if chunk_id not in cited_set)
        results.append(
            ClaimSupportResult(
                claim_id=claim.claim_id,
                text=claim.text,
                expected_supporting_chunk_ids=expected,
                actual_cited_chunk_ids=cited,
                missing_expected_chunk_ids=missing,
                supported=not missing,
            )
        )

    supported_count = sum(result.supported for result in results)
    claim_count = len(results)
    return ClaimSupportEvaluation(
        claim_results=tuple(results),
        supported_claim_count=supported_count,
        unsupported_claim_count=claim_count - supported_count,
        claim_count=claim_count,
        claim_support_rate=(
            supported_count / claim_count if claim_count else None
        ),
    )


def _unique_ids(values: object, label: str) -> tuple[ChunkId, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{label} must be a sequence of chunk IDs.")
    unique: dict[ChunkId, None] = {}
    for index, value in enumerate(values):
        try:
            unique.setdefault(value, None)
        except TypeError as error:
            raise ValueError(f"{label}[{index}] must be hashable.") from error
    return tuple(unique)
