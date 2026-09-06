"""FastAPI dependency injection for the SecureMail-ML API.

Provides the loaded model bundle + calibration state as a singleton,
request ID generation, authentication, and database sessions.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ml.calibration import CalibrationState
from ml.models import ModelBundle
from ml.product import DEFAULT_BUNDLE, load_runtime

from api.config import settings
from api.database import Base

logger = logging.getLogger("securemailscope.api")


@dataclass
class MLRuntime:
    """Holds the loaded model bundle and calibration state."""

    bundle: ModelBundle
    calibration: CalibrationState
    bundle_path: str


_runtime: MLRuntime | None = None


def load_ml_runtime() -> MLRuntime:
    """Load or return the cached ML runtime singleton."""
    global _runtime
    if _runtime is not None:
        return _runtime

    bundle_path = os.environ.get("SECUREMAIL_BUNDLE_PATH", settings.SECUREMAIL_BUNDLE_PATH)
    logger.info("Loading model bundle from %s", bundle_path)
    bundle, calibration = load_runtime(bundle_path)
    _runtime = MLRuntime(
        bundle=bundle,
        calibration=calibration,
        bundle_path=bundle_path,
    )
    logger.info(
        "Model bundle loaded: version=%s, models=%s",
        bundle.version,
        list(bundle.model_names),
    )
    return _runtime


def get_runtime() -> MLRuntime:
    """FastAPI dependency that returns the loaded runtime."""
    if _runtime is None:
        raise RuntimeError(
            "ML runtime not loaded. The model bundle may be missing or failed to load."
        )
    return _runtime


def reset_runtime() -> None:
    """Clear the cached runtime (useful for tests)."""
    global _runtime
    _runtime = None


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())


def authenticate_request(api_key: str | None = None) -> bool:
    """Placeholder authentication dependency."""
    required_key = os.environ.get("SECUREMAIL_API_KEY")
    if required_key is None:
        return True
    if api_key is None or api_key != required_key:
        return False
    return True


# --- Database Dependencies ---

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
