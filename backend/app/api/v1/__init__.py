from fastapi import APIRouter

from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.answers import router as answers_router
from app.api.v1.retrieval import router as retrieval_router

router = APIRouter()
router.include_router(health_router)
router.include_router(documents_router)
router.include_router(retrieval_router)
router.include_router(answers_router)
