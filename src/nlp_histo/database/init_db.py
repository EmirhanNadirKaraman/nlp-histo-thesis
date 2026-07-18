"""Fresh-database initializer:  ``python -m database.init_db``

Creates and verifies the current SQLAlchemy ORM schema on a new PostgreSQL
database. Table creation is delegated to the existing
``DatabaseConnection.create_tables()`` (``Base.metadata.create_all``) — this module
adds configuration validation, state classification, and verification around it.

Alembic is not used for fresh initialization: revision 0001 references
``documents``, which no revision creates, so ``alembic upgrade head`` cannot build
an empty database. See the README's "Schema ownership" section.

Safety: this command only ever creates missing tables. It never drops, truncates,
alters, or deletes anything, and it has no destructive flag.

Importing this module has no side effects — it opens no connection, builds no
engine, parses no arguments, and prints nothing. All behaviour runs through
``main()``.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import sqlalchemy as sa

from .models import Base

# Same .env resolution as db_connection.py — one config source. Searched upward from
# the working directory (not relative to this file: the package is installed), and
# overridable with NLP_HISTO_ENV_FILE.
def _env_path() -> Path:
    from dotenv import find_dotenv

    explicit = os.getenv("NLP_HISTO_ENV_FILE")
    return Path(explicit or find_dotenv(usecwd=True) or ".env")

#: Variables that must be set explicitly. ``DB_CONFIG`` in db_connection.py silently
#: defaults every one of these, so we check the environment, not the resolved config.
REQUIRED_VARS: tuple[str, ...] = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER")
#: Must be present, but may be explicitly empty (local peer/trust auth).
PASSWORD_VAR = "DB_PASSWORD"

# Schema states.
EMPTY = "EMPTY"
CURRENT = "CURRENT"
PARTIAL_SAFE = "PARTIAL_SAFE"
DRIFTED = "DRIFTED"
EXTRA_OBJECTS = "EXTRA_OBJECTS"


class ConfigError(Exception):
    """Database configuration is missing or invalid."""


class ConnectionError_(Exception):
    """Could not reach the configured database."""


@dataclass(frozen=True)
class Problem:
    """One verification failure."""
    table: str
    obj: str
    expected: str
    observed: str

    def __str__(self) -> str:  # pragma: no cover - formatting
        return (f"{self.table}.{self.obj}: expected {self.expected}, "
                f"observed {self.observed}")


@dataclass
class SchemaReport:
    state: str
    missing_tables: list[str] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    extra_tables: list[str] = field(default_factory=list)


# Configuration

def load_env(env_path: Path | None = None) -> None:
    """Load the project's .env using the same convention as db_connection.py."""
    path = _env_path() if env_path is None else env_path
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard dependency in practice
        return
    if path.exists():
        load_dotenv(path)


def validate_config(env: Mapping[str, str]) -> dict[str, str]:
    """Return the explicitly-configured connection settings, or raise ConfigError.

    Never returns or logs the password value.
    """
    missing = [v for v in REQUIRED_VARS if not (env.get(v) or "").strip()]
    if PASSWORD_VAR not in env:                 # present-but-empty is allowed
        missing.append(PASSWORD_VAR)
    if missing:
        raise ConfigError(
            "missing database configuration: " + ", ".join(sorted(missing))
            + f"\n  Copy .env.example to .env and set them ({_env_path()})."
        )

    port = env["DB_PORT"].strip()
    if not port.isdigit():
        raise ConfigError(f"DB_PORT must be an integer, got {port!r}")

    return {
        "host": env["DB_HOST"].strip(),
        "port": port,
        "database": env["DB_NAME"].strip(),
        "user": env["DB_USER"].strip(),
    }


def target_description(cfg: Mapping[str, str]) -> str:
    """Human-readable target — deliberately excludes the password.

    Delegates to the shared formatter so every command renders a target identically and
    no caller can invent a variant that leaks the password (B-113).
    """
    from .env_routing import format_target_config

    return format_target_config(cfg)


# Verification (small, testable units; each takes an Inspector-like object)

def expected_tables() -> list[str]:
    return sorted(Base.metadata.tables)


def _is_text_array(type_: object) -> bool:
    """True for PostgreSQL TEXT[]/VARCHAR[] — i.e. an ARRAY of a text-ish type."""
    if not isinstance(type_, sa.ARRAY):
        return False
    return isinstance(getattr(type_, "item_type", None), (sa.Text, sa.String))


def verify_columns(inspector, present: Iterable[str]) -> list[Problem]:
    """Every ORM-declared column must exist in each present expected table."""
    problems: list[Problem] = []
    for tname in sorted(present):
        observed = {c["name"] for c in inspector.get_columns(tname)}
        for col in Base.metadata.tables[tname].columns:
            if col.name not in observed:
                problems.append(Problem(tname, col.name, "column present", "missing"))
    return problems


def verify_semantic_types(inspector, present: Iterable[str]) -> list[Problem]:
    """entities.semantic_types must be a text ARRAY — not VARCHAR (see B-097)."""
    if "entities" not in set(present):
        return []
    for col in inspector.get_columns("entities"):
        if col["name"] == "semantic_types":
            if not _is_text_array(col["type"]):
                return [Problem("entities", "semantic_types",
                                "ARRAY of text (TEXT[])", str(col["type"]))]
            return []
    return [Problem("entities", "semantic_types", "column present", "missing")]


def verify_narrative_summary(inspector, present: Iterable[str]) -> list[Problem]:
    if "pipeline_runs" not in set(present):
        return []
    observed = {c["name"] for c in inspector.get_columns("pipeline_runs")}
    if "narrative_summary" not in observed:
        return [Problem("pipeline_runs", "narrative_summary", "column present", "missing")]
    return []


def critical_unique_constraints() -> dict[str, set[str]]:
    """Named unique constraints declared by the current ORM, per table."""
    out: dict[str, set[str]] = {}
    for tname, table in Base.metadata.tables.items():
        names = {c.name for c in table.constraints
                 if isinstance(c, sa.UniqueConstraint) and c.name}
        if names:
            out[tname] = names
    return out


def verify_unique_constraints(inspector, present: Iterable[str]) -> list[Problem]:
    problems: list[Problem] = []
    present_set = set(present)
    for tname, expected in sorted(critical_unique_constraints().items()):
        if tname not in present_set:
            continue
        observed = {u["name"] for u in inspector.get_unique_constraints(tname)}
        for name in sorted(expected - observed):
            problems.append(Problem(tname, name, "unique constraint present", "missing"))
    return problems


def verify_schema(inspector, present: Iterable[str]) -> list[Problem]:
    """All bounded structural checks against the tables that currently exist."""
    present = sorted(set(present))
    return (verify_columns(inspector, present)
            + verify_semantic_types(inspector, present)
            + verify_narrative_summary(inspector, present)
            + verify_unique_constraints(inspector, present))


# State classification (must run before any create_tables() call)

def _inspector(engine):
    """Indirection so tests can substitute an Inspector without patching SQLAlchemy."""
    return sa.inspect(engine)


def classify(inspector) -> SchemaReport:
    expected = set(expected_tables())
    observed = set(inspector.get_table_names())

    present = expected & observed
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    problems = verify_schema(inspector, present) if present else []
    if problems:
        return SchemaReport(DRIFTED, missing, problems, extra)
    if not present:
        return SchemaReport(EMPTY, missing, [], extra)
    if missing:
        return SchemaReport(PARTIAL_SAFE, missing, [], extra)
    return SchemaReport(EXTRA_OBJECTS if extra else CURRENT, [], [], extra)


# Smoke test (never commits; rolls back in a finally block)

def run_smoke(engine) -> None:
    """Insert one Document + TextElement, read them back, roll everything back.

    Uses SQLAlchemy's external-transaction pattern rather than
    ``DatabaseConnection.session_scope()``, which commits on exit.
    """
    from sqlalchemy.orm import Session

    from .models import Document, TextElement

    tag = f"SMOKE_{uuid.uuid4().hex[:8]}"
    conn = engine.connect()
    trans = conn.begin()
    session = Session(bind=conn)
    try:
        doc = Document(pmcid=tag, filename=f"{tag}.pdf", file_path=f"/nonexistent/{tag}.pdf")
        session.add(doc)
        session.flush()                       # assigns doc.id; still inside the transaction

        element = TextElement(
            document_id=doc.id,
            unique_path=f"{tag}/Smoke/0",
            path_list=["Smoke"],
            path_string="Smoke",
            depth=1,
            position_in_section=0,
            text_content="init_db smoke test",
        )
        session.add(element)
        session.flush()

        got_doc = session.query(Document).filter_by(pmcid=tag).one()
        got_el = session.query(TextElement).filter_by(document_id=got_doc.id).one()
        if got_el.document_id != got_doc.id:   # pragma: no cover - defensive
            raise AssertionError("smoke: TextElement is not linked to its Document")
    finally:
        session.close()
        trans.rollback()                      # nothing is ever persisted
        conn.close()


# CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m database.init_db",
        description=(
            "Create and verify the current SQLAlchemy ORM schema on a PostgreSQL "
            "database. Only ever creates missing tables — never drops, truncates, "
            "or alters anything. Alembic is not used for fresh initialization."
        ),
    )
    p.add_argument("--check-only", action="store_true",
                   help="Verify the existing schema without creating anything (read-only).")
    p.add_argument("--smoke", action="store_true",
                   help="After initializing, insert and read back one Document + "
                        "TextElement inside a transaction that is rolled back "
                        "(leaves zero rows).")
    return p


def _print_problems(report: SchemaReport) -> None:
    print("ERROR: the existing schema is not compatible with the current models.\n")
    for prob in report.problems:
        print(f"  - {prob}")
    print("\n  This command creates missing tables; it will not alter existing ones.")
    print("  Repair the schema manually, or initialize a fresh, empty database.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_only and args.smoke:
        print("ERROR: --check-only is read-only and cannot be combined with --smoke.",
              file=sys.stderr)
        return 2

    try:
        load_env()
        cfg = validate_config(os.environ)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    target = target_description(cfg)
    print(f"Target: {target}")

    from .db_connection import close_db_connection, get_db_connection

    try:
        db = get_db_connection()
        try:
            inspector = _inspector(db.engine)
            report = classify(inspector)

            if report.extra_tables:
                print(f"Note: {len(report.extra_tables)} unrelated table(s) present and left "
                      f"untouched: {', '.join(report.extra_tables)}")

            if report.state == DRIFTED:
                _print_problems(report)
                return 1

            if args.check_only:
                if report.state in (CURRENT, EXTRA_OBJECTS):
                    print(f"OK: schema is present and valid ({len(expected_tables())} tables).")
                    return 0
                print("ERROR: schema is not fully initialized. Missing table(s): "
                      + ", ".join(report.missing_tables), file=sys.stderr)
                return 1

            if report.state in (CURRENT, EXTRA_OBJECTS):
                print("Schema already initialized — nothing to create.")
            else:
                if report.state == PARTIAL_SAFE:
                    print(f"Creating {len(report.missing_tables)} missing table(s): "
                          + ", ".join(report.missing_tables))
                db.create_tables()                       # existing implementation

                report = classify(_inspector(db.engine))   # verify the result
                if report.state == DRIFTED:
                    _print_problems(report)
                    return 1
                if report.missing_tables:
                    print("ERROR: tables are still missing after creation: "
                          + ", ".join(report.missing_tables), file=sys.stderr)
                    return 1

            if args.smoke:
                run_smoke(db.engine)
                print("Smoke test passed (transaction rolled back — no rows written).")

            print(f"OK: schema verified ({len(expected_tables())} tables).")
            print("\nThis database was initialized from the current SQLAlchemy ORM schema "
                  "(database/models.py).")
            print("Alembic was not used for fresh initialization — do not run "
                  "`alembic upgrade head` merely to initialize this new database.")
            return 0
        finally:
            close_db_connection()
    except sa.exc.SQLAlchemyError as exc:
        origin = getattr(exc, "orig", exc)
        print(f"ERROR: could not connect to {target}\n  {type(origin).__name__}: "
              f"{str(origin).strip().splitlines()[0] if str(origin).strip() else ''}",
              file=sys.stderr)
        print("\n  Check that:\n"
              "    - PostgreSQL is running (pg_isready)\n"
              f"    - the database '{cfg['database']}' exists (createdb {cfg['database']})\n"
              "    - DB_HOST / DB_PORT / DB_USER / DB_PASSWORD in .env are correct",
              file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
