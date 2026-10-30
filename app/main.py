"""FastAPI application factory.

Usage:
    uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.config import get_settings
from app.logging_config import request_id_ctx, setup_logging
from app.services import Services, build_services

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("app.access")


def create_app(services: Services | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    Args:
        services: Pre-built service container. When ``None``, services are
            constructed from environment settings during lifespan startup.
            Tests pass a container built from fakes.

    Returns:
        A fully-wired FastAPI app with routes, middleware, and metrics.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services(settings)
        yield

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        response.headers["x-request-id"] = request_id
        if request.url.path not in ("/health", "/metrics"):
            logger.info(
                "request",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "ms": elapsed,
                    }
                },
            )
        return response

    app.include_router(router)
    Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
