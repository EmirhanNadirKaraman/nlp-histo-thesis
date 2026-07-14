"""Alembic environment — wired to the project's SQLAlchemy Base and .env config."""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

# ---------------------------------------------------------------------------
# Make the project root importable so `from database import ...` works
# regardless of where alembic is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before importing database (db_connection.py does this too, but
# doing it here ensures the URL is available for the offline mode as well).
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from nlp_histo.database.db_connection import get_database_url  # noqa: E402
from nlp_histo.database.models import Base  # noqa: E402

# ---------------------------------------------------------------------------
# Alembic config object — gives access to alembic.ini values.
# ---------------------------------------------------------------------------
config = context.config

# Override the sqlalchemy.url with the value derived from .env so we never
# hard-code credentials in alembic.ini.
config.set_main_option("sqlalchemy.url", get_database_url())

# Interpret the config file for Python logging (if present).
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata for autogenerate support.
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline mode: emit SQL to stdout without a live connection.
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render column-level server defaults so autogenerate captures them.
        render_as_batch=False,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode: run migrations against a live database connection.
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    from sqlalchemy import engine_from_config, pool

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
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
