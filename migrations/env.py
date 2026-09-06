"""Alembic environment configuration for SecureMail-ML.

Reads DATABASE_URL from the .env file via api/config.py (pydantic-settings).
Supports both synchronous (offline) and async (online) migration modes.

Usage:
  # Offline diff against current schema (no live DB needed):
  alembic upgrade head --sql

  # Online migration against live DB:
  alembic upgrade head

  # Downgrade one step:
  alembic downgrade -1

  # Auto-generate a new migration after editing api/database.py:
  alembic revision --autogenerate -m "describe your change"
"""
from __future__ import annotations

import re
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, create_engine

from alembic import context

# ---------------------------------------------------------------------------
# Load our application config (reads .env automatically via pydantic-settings)
# ---------------------------------------------------------------------------
from api.config import settings

# Import our ORM Base so Alembic can compare models against the live schema
from api.database import Base

# ---------------------------------------------------------------------------
# Alembic Config object — provides access to values in alembic.ini
# ---------------------------------------------------------------------------
config = context.config

# Override the sqlalchemy.url from our pydantic settings so the .env file
# is the single source of truth.  Convert async driver to sync for Alembic:
#   postgresql+asyncpg://...  →  postgresql+psycopg2://...
#   sqlite+aiosqlite://...    →  sqlite://...
def _sync_url(url: str) -> str:
    """Strip async driver variants for Alembic's synchronous engine."""
    url = re.sub(r"\+asyncpg", "+psycopg2", url)
    url = re.sub(r"\+aiosqlite", "", url)
    return url

config.set_main_option("sqlalchemy.url", _sync_url(settings.DATABASE_URL))

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Model metadata — Alembic uses this to autogenerate migrations
# ---------------------------------------------------------------------------
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline mode: generate SQL script without a live DB connection
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine, so an
    actual DBAPI connection is never required. Migrations are emitted
    as a SQL script (useful for review or applying manually).
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode: run migrations against a live DB connection
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    """Run migrations in 'online' mode with an active DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
