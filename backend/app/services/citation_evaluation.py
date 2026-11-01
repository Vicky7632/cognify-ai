from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from app.services.grounded_answer_service import GroundedAnswer
from app.services.retrieval_evaluation_dataset import RetrievalEvaluationDataset

ChunkId: TypeAlias = Hashable


@dataclass(frozen=True, slots=True)
class CitationQueryEvaluation:
    query_id: str
    expected_relevant_chunk_ids: tuple[str, ...]
    available_chunk_count: int
    unique_citation_count: int
    valid_citation_count: int
    invalid_citation_ids: tuple[ChunkId, ...]
    citation_validity: float | None
    has_valid_citation: bool
    sufficient_evidence: bool


@dataclass(frozen=True, slots=True)
class CitationEvaluation:
    query_results: tuple[CitationQueryEvaluation, ...]
    citation_validity: float | None
    valid_citation_count: int
    cited_id_count: int
    citation_coverage: float | None
    covered_answerable_query_count: int
    answerable_query_count: int


def evaluate_citations(
    dataset: RetrievalEvaluationDataset,
    available_chunk_ids: Mapping[str, Sequence[ChunkId]],
    answers: Mapping[str, GroundedAnswer],
) -> CitationEvaluation:
    """Measure citation-ID availability and answerable-query citation coverage.

    Citation validity is identifier membership only; it does not assess whether
    a chunk supports any claim in the answer.
    """
    known_query_ids = {example.query_id for example in dataset.examples}
    unknown_query_ids = (available_chunk_ids.keys() | answers.keys()) - known_query_ids
    if unknown_query_ids:
        raise ValueError(
            "Citation evaluation contains unknown query IDs: "
            + ", ".join(sorted(unknown_query_ids))
        )

    results: list[CitationQueryEvaluation] = []
    for example in dataset.examples:
        available_ids = _available_ids(
            available_chunk_ids.get(example.query_id, ()),
            example.query_id,
        )
        answer = answers.get(example.query_id)
        citations = () if answer is None else answer.citations
        cited_ids = _unique_citation_ids(citations, example.query_id)
        valid_ids = tuple(
            citation_id
            for citation_id in cited_ids
            if citation_id in available_ids
        )
        invalid_ids = tuple(
            citation_id
            for citation_id in cited_ids
            if citation_id not in available_ids
        )
        validity = len(valid_ids) / len(cited_ids) if cited_ids else None
        results.append(
            CitationQueryEvaluation(
                query_id=example.query_id,
                expected_relevant_chunk_ids=example.relevant_chunk_ids,
                available_chunk_count=len(available_ids),
                unique_citation_count=len(cited_ids),
                valid_citation_count=len(valid_ids),
                invalid_citation_ids=invalid_ids,
                citation_validity=validity,
                has_valid_citation=bool(valid_ids),
                sufficient_evidence=example.sufficient_evidence,
            )
        )

    valid_citation_count = sum(result.valid_citation_count for result in results)
    cited_id_count = sum(result.unique_citation_count for result in results)
    answerable_results = tuple(
        result for result in results if result.sufficient_evidence
    )
    covered_query_count = sum(
        result.has_valid_citation for result in answerable_results
    )

    return CitationEvaluation(
        query_results=tuple(results),
        citation_validity=(
            valid_citation_count / cited_id_count if cited_id_count else None
        ),
        valid_citation_count=valid_citation_count,
        cited_id_count=cited_id_count,
        citation_coverage=(
            covered_query_count / len(answerable_results)
            if answerable_results
            else None
        ),
        covered_answerable_query_count=covered_query_count,
        answerable_query_count=len(answerable_results),
    )


def _available_ids(
    chunk_ids: Sequence[ChunkId],
    query_id: str,
) -> set[ChunkId]:
    try:
        return set(chunk_ids)
    except TypeError as error:
        raise ValueError(
            f"Available chunk IDs for query {query_id} must be hashable."
        ) from error


def _unique_citation_ids(
    citations: Sequence[object],
    query_id: str,
) -> tuple[ChunkId, ...]:
    unique_ids: dict[ChunkId, None] = {}
    for citation in citations:
        citation_id = getattr(citation, "segment_id", None)
        if citation_id is None:
            raise ValueError(
                f"Citation for query {query_id} has no segment_id."
            )
        try:
            unique_ids.setdefault(citation_id, None)
        except TypeError as error:
            raise ValueError(
                f"Citation for query {query_id} has an invalid segment_id."
            ) from error
    return tuple(unique_ids)
