from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.vector_store.protocol import VectorMetadataValue


class DocumentRetrievalRequest(BaseModel):
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


class DocumentRetrievalResultResponse(BaseModel):
    vector_id: str
    document_id: UUID
    segment_id: UUID
    ordinal: int
    score: float
    metadata: dict[str, VectorMetadataValue]


class DocumentRetrievalResponse(BaseModel):
    query: str
    top_k: int
    results: list[DocumentRetrievalResultResponse]
