"""FastAPI application factory for the SecureMail-ML HTTP API.

Creates the application, registers routes, configures CORS,
and loads the ML runtime at startup.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import load_ml_runtime, reset_runtime
from api.routes import router
from api.data_routes import data_router
from api.capture_routes import capture_router
from api.auth_routes import router as auth_router

logger = logging.getLogger("securemailscope.api")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load ML runtime & database at startup, clean up on shutdown."""
    logger.info("Starting SecureMail-ML API...")
    try:
        from api.dependencies import init_db
        await init_db()
        logger.info("Database tables initialized.")
    except Exception:
        logger.exception("Failed to initialize database tables.")

    try:
        load_ml_runtime()
        logger.info("ML runtime loaded successfully.")
    except Exception:
        logger.exception(
            "Failed to load ML runtime. The API will start but /analyses "
            "will return 503 until the bundle is available."
        )
    yield
    reset_runtime()
    logger.info("SecureMail-ML API shut down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="SecureMail-ML API",
        version="0.1.0",
        description=(
            "Authenticated, synchronous session-feature analysis API for "
            "SecureMailScope. Accepts validated session-features.v1 records "
            "and returns ml-result.v1 risk classifications, deterministic "
            "findings, model signals, and feature-level explanations."
        ),
        lifespan=lifespan,
    )

    # --- CORS middleware ---
    cors_origins = os.environ.get("SECUREMAIL_CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in cors_origins.split(",")]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # --- Register routes ---
    application.include_router(router)
    application.include_router(data_router)
    application.include_router(capture_router)
    application.include_router(auth_router)

    return application


app = create_app()
