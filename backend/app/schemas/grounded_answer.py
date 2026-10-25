from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GroundedAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=10, ge=1, le=100, strict=True)
    document_id: UUID | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be empty or whitespace-only.")
        return value


class GroundedCitationResponse(BaseModel):
    document_id: UUID
    segment_id: UUID
    ordinal: int
    page_number: int | None
    original_filename: str
    file_type: str
    score: float


class GroundedAnswerResponse(BaseModel):
    answer: str
    citations: list[GroundedCitationResponse]
    insufficient_evidence: bool
