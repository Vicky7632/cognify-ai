from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentSegmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    ordinal: int
    text_content: str
    page_number: int | None
    start_seconds: float | None
    end_seconds: float | None


class DocumentSegmentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    segments: list[DocumentSegmentResponse]
