from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.document import Document, DocumentFileType, DocumentStatus


class DocumentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_filename: str = Field(min_length=1, max_length=255)
    file_type: DocumentFileType

    @field_validator("original_filename")
    @classmethod
    def validate_original_filename(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Filename must not be empty or whitespace-only.")
        if "/" in value or "\\" in value:
            raise ValueError("Filename must not contain path separators.")
        return value


class DocumentStatusUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DocumentStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    file_type: DocumentFileType
    processing_status: DocumentStatus
    created_at: datetime
    updated_at: datetime


def to_document_response(document: Document) -> DocumentResponse:
    return DocumentResponse.model_validate(document)
