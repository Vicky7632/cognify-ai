from fastapi import FastAPI

from app.api.v1 import router as v1_router
from app.core.config import Settings


def create_app(settings: Settings) -> FastAPI:
    application = FastAPI(title=settings.app_name)
    application.include_router(v1_router, prefix=settings.api_v1_prefix)
    return application
