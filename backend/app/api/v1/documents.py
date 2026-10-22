import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import DocumentFileType
from app.schemas.document import (
    DocumentCreateRequest,
    DocumentResponse,
    DocumentStatusUpdateRequest,
    to_document_response,
    validate_document_filename,
)
from app.services import document_service
from app.services.file_storage import (
    FileStorageError,
    InvalidUploadError,
    UploadTooLargeError,
    delete_stored_file,
    store_upload,
)

logger = logging.getLogger(__name__)
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


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
) -> DocumentResponse:
    form = await request.form()
    try:
        form_parts = form.multi_items()
        if len(form_parts) != 1 or form_parts[0] != ("file", file):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Exactly one file field is required.",
            )
        try:
            filename = validate_document_filename(file.filename or "")
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        extension = Path(filename).suffix.lower().removeprefix(".")
        try:
            file_type = DocumentFileType(extension)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported document file type.",
            ) from error

        settings = request.app.state.settings
        try:
            storage_key = await store_upload(
                file,
                file_type=file_type,
                upload_root=settings.upload_root,
                max_size_bytes=settings.max_upload_size_bytes,
            )
        except UploadTooLargeError as error:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded file exceeds the maximum allowed size.",
            ) from error
        except InvalidUploadError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except FileStorageError as error:
            logger.error("Unable to store uploaded document.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to store uploaded document.",
            ) from error

        try:
            with session.begin():
                document = document_service.create_document(
                    session,
                    original_filename=filename,
                    file_type=file_type.value,
                    storage_path=storage_key,
                )
                response = to_document_response(document)
        except SQLAlchemyError:
            try:
                delete_stored_file(storage_key, settings.upload_root)
            except FileStorageError:
                logger.warning("Unable to clean up an upload after database failure.")
            logger.error("Unable to persist uploaded document metadata.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to save uploaded document metadata.",
            ) from None
        except Exception:
            try:
                delete_stored_file(storage_key, settings.upload_root)
            except FileStorageError:
                logger.warning("Unable to clean up an upload after request failure.")
            raise
        return response
    finally:
        await form.close()


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
