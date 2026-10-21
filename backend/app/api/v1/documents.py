from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.document import (
    DocumentCreateRequest,
    DocumentResponse,
    DocumentStatusUpdateRequest,
    to_document_response,
)
from app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    request: DocumentCreateRequest,
    session: Session = Depends(get_db),
) -> DocumentResponse:
    with session.begin():
        document = document_service.create_document(
            session,
            original_filename=request.original_filename,
            file_type=request.file_type.value,
            storage_path=f"pending/{uuid4().hex}",
        )
        response = to_document_response(document)
    return response


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    session: Session = Depends(get_db),
) -> list[DocumentResponse]:
    return [
        to_document_response(document)
        for document in document_service.list_documents(session)
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    session: Session = Depends(get_db),
) -> DocumentResponse:
    document = document_service.get_document(session, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return to_document_response(document)


@router.patch("/{document_id}/status", response_model=DocumentResponse)
def update_document_status(
    document_id: UUID,
    request: DocumentStatusUpdateRequest,
    session: Session = Depends(get_db),
) -> DocumentResponse:
    with session.begin():
        document = document_service.update_document_status(
            session,
            document_id,
            request.status,
        )
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        response = to_document_response(document)
    return response


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: UUID,
    session: Session = Depends(get_db),
) -> Response:
    with session.begin():
        deleted = document_service.delete_document(session, document_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
