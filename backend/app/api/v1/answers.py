import logging
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.v1.retrieval import get_document_retrieval_service
from app.db.session import get_db
from app.schemas.grounded_answer import (
    GroundedAnswerRequest,
    GroundedAnswerResponse,
)
from app.services.document_retrieval_service import DocumentRetrievalService
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingInputError,
    EmbeddingOutputError,
    EmbeddingProviderError,
)
from app.services.gemini_grounded_answer_provider import (
    create_gemini_grounded_answer_provider,
)
from app.services.grounded_answer_service import (
    GroundedAnswerConfigurationError,
    GroundedAnswerProvider,
    GroundedAnswerProviderError,
    GroundedAnswerService,
)
from app.vector_store.protocol import (
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreInputError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retrieval", tags=["retrieval"])


def get_grounded_answer_provider(
    request: Request,
) -> Generator[GroundedAnswerProvider, None, None]:
    provider = getattr(request.app.state, "grounded_answer_provider", None)
    owns_provider = provider is None
    if provider is None:
        try:
            provider = create_gemini_grounded_answer_provider(
                request.app.state.settings
            )
        except GroundedAnswerConfigurationError:
            logger.error("Grounded answer provider is not configured.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Grounded answer generation is not configured.",
            ) from None
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grounded answer generation is not configured.",
        )
    try:
        yield provider
    finally:
        if owns_provider:
            try:
                provider.close()
            except Exception:
                logger.error("Unable to close the grounded answer provider.")


def get_grounded_answer_service(
    answer_provider: Annotated[
        GroundedAnswerProvider,
        Depends(get_grounded_answer_provider),
    ],
    retrieval_service: Annotated[
        DocumentRetrievalService,
        Depends(get_document_retrieval_service),
    ],
) -> GroundedAnswerService:
    return GroundedAnswerService(retrieval_service, answer_provider)


@router.post(
    "/answer",
    response_model=GroundedAnswerResponse,
    response_model_exclude_none=True,
)
def generate_grounded_answer(
    request: GroundedAnswerRequest,
    answer_service: Annotated[
        GroundedAnswerService,
        Depends(get_grounded_answer_service),
    ],
    session: Annotated[Session, Depends(get_db)],
) -> GroundedAnswerResponse:
    try:
        result = answer_service.answer(
            session,
            request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except (EmbeddingInputError, VectorStoreInputError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except (
        EmbeddingConfigurationError,
        GroundedAnswerConfigurationError,
        VectorStoreConfigurationError,
    ):
        logger.error("Grounded answer generation is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grounded answer generation is not configured.",
        ) from None
    except (
        EmbeddingOutputError,
        EmbeddingProviderError,
        GroundedAnswerProviderError,
        VectorStoreError,
    ):
        logger.error("Grounded answer provider request failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Grounded answer generation failed.",
        ) from None
    except SQLAlchemyError:
        logger.error("Unable to load retrieved document segments.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load retrieved document segments.",
        ) from None

    return GroundedAnswerResponse(
        answer=result.answer,
        citations=[
            {
                "document_id": citation.document_id,
                "segment_id": citation.segment_id,
                "ordinal": citation.ordinal,
                "page_number": citation.page_number,
                "original_filename": citation.original_filename,
                "file_type": citation.file_type,
                "score": citation.score,
            }
            for citation in result.citations
        ],
        insufficient_evidence=result.insufficient_evidence,
    )
