import logging
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.services.document_retrieval_service import DocumentRetrievalService
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingInputError,
    EmbeddingOutputError,
    EmbeddingProviderError,
    EmbeddingService,
    create_embedding_service,
)
from app.schemas.retrieval import (
    DocumentRetrievalRequest,
    DocumentRetrievalResponse,
    DocumentRetrievalResultResponse,
)
from app.vector_store.pinecone_adapter import create_pinecone_vector_store
from app.vector_store.protocol import (
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreInputError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retrieval", tags=["retrieval"])


def get_document_retrieval_service(
    request: Request,
) -> Generator[DocumentRetrievalService, None, None]:
    settings = request.app.state.settings
    embedding_service: EmbeddingService | None = None
    vector_store: VectorStore | None = None
    try:
        embedding_service = create_embedding_service(settings)
        vector_store = create_pinecone_vector_store(settings)
    except EmbeddingConfigurationError:
        _close_clients(embedding_service, vector_store)
        logger.error("Retrieval embedding provider is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not configured.",
        ) from None
    except VectorStoreConfigurationError:
        _close_clients(embedding_service, vector_store)
        logger.error("Retrieval vector store is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not configured.",
        ) from None

    try:
        yield DocumentRetrievalService(embedding_service, vector_store)
    finally:
        _close_clients(embedding_service, vector_store)


@router.post(
    "/search",
    response_model=DocumentRetrievalResponse,
)
def retrieve_documents(
    request: DocumentRetrievalRequest,
    retrieval_service: Annotated[
        DocumentRetrievalService,
        Depends(get_document_retrieval_service),
    ],
) -> DocumentRetrievalResponse:
    try:
        results = retrieval_service.retrieve(
            request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except (EmbeddingInputError, VectorStoreInputError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except (EmbeddingConfigurationError, VectorStoreConfigurationError):
        logger.error("Document retrieval is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not configured.",
        ) from None
    except (EmbeddingProviderError, EmbeddingOutputError, VectorStoreError):
        logger.error("Document retrieval provider request failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document retrieval failed.",
        ) from None

    return DocumentRetrievalResponse(
        query=request.query,
        top_k=request.top_k,
        results=[
            DocumentRetrievalResultResponse(
                vector_id=result.vector_id,
                document_id=result.document_id,
                segment_id=result.segment_id,
                ordinal=result.ordinal,
                score=result.score,
                metadata=dict(result.metadata),
            )
            for result in results
        ],
    )


def _close_clients(
    embedding_service: EmbeddingService | None,
    vector_store: VectorStore | None,
) -> None:
    for client, label in (
        (vector_store, "Pinecone"),
        (embedding_service, "embedding"),
    ):
        if client is None:
            continue
        try:
            client.close()
        except Exception:
            logger.error("Unable to close the %s retrieval client.", label)
