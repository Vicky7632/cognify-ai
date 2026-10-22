import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import router as v1_router
from app.core.config import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info("Application startup complete")
        yield
        logger.info("Application shutdown complete")

    application = FastAPI(title=settings.app_name, lifespan=lifespan)
    application.state.settings = settings
    application.include_router(v1_router, prefix=settings.api_v1_prefix)
    return application
