import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.config import settings
from app.models import init_db
from app.security import (
    BodySizeLimitMiddleware,
    OriginCheckMiddleware,
    SecurityHeadersMiddleware,
    validation_error_handler,
)
from app.services.transcription import fail_interrupted_jobs

logger = logging.getLogger(__name__)

# Multipart framing and the optional name field on top of the audio file itself.
REQUEST_OVERHEAD_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    if not settings.use_celery:
        # In-process jobs die with the process; don't leave them "processing" forever.
        interrupted = fail_interrupted_jobs()
        if interrupted:
            logger.warning("Marked %d interrupted transcription(s) as failed", interrupted)
    yield


app = FastAPI(
    title="Music Transcriber API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.expose_docs else None,
    redoc_url="/redoc" if settings.expose_docs else None,
    openapi_url="/openapi.json" if settings.expose_docs else None,
)

# Added innermost first: requests pass SecurityHeaders -> CORS -> OriginCheck -> BodySize.
# CORS sits outside the checks so their error responses stay readable by the web app.
app.add_middleware(
    BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + REQUEST_OVERHEAD_BYTES
)
app.add_middleware(OriginCheckMiddleware, allowed_origins=settings.cors_origin_list)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,  # no cookies are used
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.include_router(api_router, prefix="/api")


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
