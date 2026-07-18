"""Unit tests for `python -m database.init_db`.

Nothing here connects to PostgreSQL: the SQLAlchemy Inspector, the engine, and the
DatabaseConnection are all replaced with small test doubles.
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

import pytest

from nlp_histo.database import init_db
from nlp_histo.database.init_db import (
    CURRENT,
    DRIFTED,
    EMPTY,
    EXTRA_OBJECTS,
    PARTIAL_SAFE,
    ConfigError,
    Problem,
    classify,
    expected_tables,
    validate_config,
    verify_narrative_summary,
    verify_schema,
    verify_semantic_types,
)


# Fakes

class FakeInspector:
    """Minimal stand-in for sqlalchemy.Inspector, built from the real ORM metadata."""

    def __init__(self, tables=None, drop_columns=None, override_types=None,
                 drop_uniques=None, extra_tables=()):
        self._tables = list(expected_tables()) if tables is None else list(tables)
        self._tables += list(extra_tables)
        self._drop_columns = drop_columns or {}          # {table: {col, ...}}
        self._override_types = override_types or {}      # {(table, col): type}
        self._drop_uniques = drop_uniques or {}          # {table: {name, ...}}

    def get_table_names(self):
        return list(self._tables)

    def get_columns(self, table):
        cols = []
        for c in init_db.Base.metadata.tables[table].columns:
            if c.name in self._drop_columns.get(table, set()):
                continue
            ctype = self._override_types.get((table, c.name), c.type)
            cols.append({"name": c.name, "type": ctype})
        return cols

    def get_unique_constraints(self, table):
        out = []
        for con in init_db.Base.metadata.tables[table].constraints:
            if isinstance(con, sa.UniqueConstraint) and con.name:
                if con.name in self._drop_uniques.get(table, set()):
                    continue
                out.append({"name": con.name,
                            "column_names": [c.name for c in con.columns]})
        return out


class FakeDB:
    """Stand-in for DatabaseConnection: records create_tables() calls."""

    def __init__(self, inspectors):
        self.engine = object()
        self._inspectors = list(inspectors)   # returned in order by sa.inspect
        self.create_calls = 0

    def create_tables(self):
        self.create_calls += 1


@pytest.fixture
def wire(monkeypatch):
    """Wire init_db's collaborators to fakes. Returns a setup(...) callable."""
    def setup(inspectors, env=None):
        db = FakeDB(inspectors)
        seq = iter(db._inspectors)
        # Patch init_db's own indirection — never the global sqlalchemy module.
        monkeypatch.setattr(init_db, "_inspector", lambda _engine: next(seq))
        monkeypatch.setattr(init_db, "load_env", lambda *a, **k: None)
        import nlp_histo.database.db_connection as dbc
        monkeypatch.setattr(dbc, "get_db_connection", lambda *a, **k: db)
        monkeypatch.setattr(dbc, "close_db_connection", lambda: None)
        full_env = {"DB_HOST": "h", "DB_PORT": "5432", "DB_NAME": "d",
                    "DB_USER": "u", "DB_PASSWORD": "secret-pw"}
        full_env.update(env or {})
        monkeypatch.setattr(init_db.os, "environ", full_env)
        return db
    return setup


# 1-4. Configuration validation

def test_explicit_configuration_is_accepted():
    cfg = validate_config({"DB_HOST": "h", "DB_PORT": "5432", "DB_NAME": "d",
                           "DB_USER": "u", "DB_PASSWORD": "pw"})
    assert cfg == {"host": "h", "port": "5432", "database": "d", "user": "u"}
    assert "pw" not in str(cfg)          # the password is never carried around


def test_empty_password_is_accepted_for_trust_auth():
    cfg = validate_config({"DB_HOST": "h", "DB_PORT": "5432", "DB_NAME": "d",
                           "DB_USER": "u", "DB_PASSWORD": ""})
    assert cfg["user"] == "u"


@pytest.mark.parametrize("missing", ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"])
def test_missing_required_configuration_is_rejected(missing):
    env = {"DB_HOST": "h", "DB_PORT": "5432", "DB_NAME": "d",
           "DB_USER": "u", "DB_PASSWORD": "pw"}
    del env[missing]
    with pytest.raises(ConfigError, match=missing):
        validate_config(env)


def test_invalid_port_is_rejected():
    with pytest.raises(ConfigError, match="DB_PORT must be an integer"):
        validate_config({"DB_HOST": "h", "DB_PORT": "not-a-port", "DB_NAME": "d",
                         "DB_USER": "u", "DB_PASSWORD": "pw"})


def test_password_never_appears_in_the_target_description():
    cfg = validate_config({"DB_HOST": "h", "DB_PORT": "5432", "DB_NAME": "d",
                           "DB_USER": "u", "DB_PASSWORD": "super-secret"})
    assert "super-secret" not in init_db.target_description(cfg)


# 5-12. State classification

def test_empty_database_is_classified_empty():
    assert classify(FakeInspector(tables=[])).state == EMPTY


def test_complete_valid_database_is_classified_current():
    assert classify(FakeInspector()).state == CURRENT


def test_safe_subset_is_classified_partial_safe():
    subset = expected_tables()[:3]
    report = classify(FakeInspector(tables=subset))
    assert report.state == PARTIAL_SAFE
    assert set(report.missing_tables) == set(expected_tables()) - set(subset)


def test_table_missing_a_column_is_classified_drifted():
    report = classify(FakeInspector(drop_columns={"documents": {"title"}}))
    assert report.state == DRIFTED
    assert Problem("documents", "title", "column present", "missing") in report.problems


def test_semantic_types_varchar_is_classified_drifted():
    report = classify(FakeInspector(
        override_types={("entities", "semantic_types"): sa.String()}))
    assert report.state == DRIFTED
    assert any(p.obj == "semantic_types" for p in report.problems)


def test_semantic_types_text_array_is_accepted():
    inspector = FakeInspector(
        override_types={("entities", "semantic_types"): postgresql.ARRAY(sa.Text())})
    assert verify_semantic_types(inspector, ["entities"]) == []
    assert classify(inspector).state == CURRENT


def test_missing_narrative_summary_is_classified_drifted():
    report = classify(FakeInspector(drop_columns={"pipeline_runs": {"narrative_summary"}}))
    assert report.state == DRIFTED
    assert verify_narrative_summary(
        FakeInspector(drop_columns={"pipeline_runs": {"narrative_summary"}}),
        ["pipeline_runs"])


def test_missing_unique_constraint_is_classified_drifted():
    report = classify(FakeInspector(
        drop_uniques={"sum_map_findings": {"uq_sum_map_finding_pos"}}))
    assert report.state == DRIFTED
    assert any(p.obj == "uq_sum_map_finding_pos" for p in report.problems)


def test_extra_tables_are_reported_but_not_rejected():
    report = classify(FakeInspector(extra_tables=["alembic_version", "old_stuff"]))
    assert report.state == EXTRA_OBJECTS          # a success state, not an error
    assert report.extra_tables == ["alembic_version", "old_stuff"]
    assert report.problems == []


def test_verify_schema_is_clean_on_a_valid_database():
    assert verify_schema(FakeInspector(), expected_tables()) == []


# 13-19. main() behaviour

def test_check_only_never_calls_create_tables(wire, capsys):
    db = wire([FakeInspector()])
    assert init_db.main(["--check-only"]) == 0
    assert db.create_calls == 0
    assert "present and valid" in capsys.readouterr().out


def test_check_only_fails_when_schema_is_incomplete(wire):
    wire([FakeInspector(tables=expected_tables()[:2])])
    assert init_db.main(["--check-only"]) == 1


def test_current_database_does_not_call_create_tables(wire, capsys):
    db = wire([FakeInspector()])
    assert init_db.main([]) == 0
    assert db.create_calls == 0
    assert "already initialized" in capsys.readouterr().out


def test_empty_database_creates_once_then_verifies(wire):
    db = wire([FakeInspector(tables=[]), FakeInspector()])   # before, after
    assert init_db.main([]) == 0
    assert db.create_calls == 1


def test_partial_safe_creates_once_then_verifies(wire, capsys):
    db = wire([FakeInspector(tables=expected_tables()[:3]), FakeInspector()])
    assert init_db.main([]) == 0
    assert db.create_calls == 1
    assert "Creating" in capsys.readouterr().out


def test_drifted_fails_before_create_tables(wire, capsys):
    db = wire([FakeInspector(drop_columns={"documents": {"title"}})])
    assert init_db.main([]) == 1
    assert db.create_calls == 0                     # preflight, not post-hoc
    assert "not compatible" in capsys.readouterr().out


def test_successful_output_contains_no_credentials(wire, capsys):
    wire([FakeInspector()])
    init_db.main([])
    out = capsys.readouterr().out
    assert "secret-pw" not in out
    assert "postgresql://" not in out
    assert "u@h:5432/d" in out                      # sanitized target only


def test_connection_failure_is_actionable_and_non_zero(wire, monkeypatch, capsys):
    wire([FakeInspector()])

    def boom(_engine):
        raise sa.exc.OperationalError("SELECT 1", {}, Exception("could not connect"))
    monkeypatch.setattr(init_db, "_inspector", boom)

    assert init_db.main([]) == 1
    err = capsys.readouterr().err
    assert "could not connect to u@h:5432/d" in err
    assert "createdb d" in err
    assert "secret-pw" not in err


def test_missing_config_exits_two_without_connecting(monkeypatch, capsys):
    monkeypatch.setattr(init_db, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(init_db.os, "environ", {"DB_HOST": "h"})

    def fail(*_a, **_k):                            # must never be reached
        raise AssertionError("must not connect when configuration is missing")
    import nlp_histo.database.db_connection as dbc
    monkeypatch.setattr(dbc, "get_db_connection", fail)

    assert init_db.main([]) == 2
    assert "missing database configuration" in capsys.readouterr().err


# 20. Flag combination

def test_check_only_with_smoke_is_rejected(capsys):
    assert init_db.main(["--check-only", "--smoke"]) == 2
    assert "read-only" in capsys.readouterr().err


# --smoke transaction control (fakes; no database)

class FakeSession:
    def __init__(self, *_a, **_k):
        self.committed = False
        self.closed = False
        self.flushes = 0

    def add(self, _obj):
        pass

    def flush(self):
        self.flushes += 1

    def commit(self):                                # must never be called
        self.committed = True

    def close(self):
        self.closed = True

    def query(self, model):
        return self

    def filter_by(self, **_kw):
        return self

    def one(self):
        class _Row:
            id = 1
            document_id = 1
        return _Row()


class FakeTrans:
    def __init__(self):
        self.rolled_back = False
        self.committed = False

    def rollback(self):
        self.rolled_back = True

    def commit(self):                                # must never be called
        self.committed = True


class FakeConn:
    def __init__(self, trans):
        self._trans = trans
        self.closed = False

    def begin(self):
        return self._trans

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, conn):
        self._conn = conn

    def connect(self):
        return self._conn


def _wire_smoke(monkeypatch, session):
    trans = FakeTrans()
    conn = FakeConn(trans)
    import sqlalchemy.orm as orm
    monkeypatch.setattr(orm, "Session", lambda **_kw: session)
    return FakeEngine(conn), conn, trans


def test_smoke_rolls_back_on_success_and_never_commits(monkeypatch):
    session = FakeSession()
    engine, conn, trans = _wire_smoke(monkeypatch, session)

    init_db.run_smoke(engine)

    assert trans.rolled_back is True
    assert trans.committed is False
    assert session.committed is False                # no commit anywhere
    assert session.closed is True and conn.closed is True
    assert session.flushes == 2                      # Document, then TextElement


def test_smoke_rolls_back_and_cleans_up_on_failure(monkeypatch):
    session = FakeSession()

    def boom():
        raise RuntimeError("insert exploded")
    session.flush = boom
    engine, conn, trans = _wire_smoke(monkeypatch, session)

    with pytest.raises(RuntimeError, match="insert exploded"):
        init_db.run_smoke(engine)

    assert trans.rolled_back is True
    assert trans.committed is False
    assert session.committed is False
    assert session.closed is True and conn.closed is True


# Import safety

def test_importing_init_db_prints_nothing_and_exits_cleanly():
    """A real, isolated import in a subprocess: no output, no crash, no connection.

    (Deliberately not importlib.reload() — reloading rebinds the module's exception
    classes and would poison every other test that holds a reference to them.)
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(init_db.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import database.init_db"],
        capture_output=True, text=True, cwd=repo, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_init_db_module_level_has_no_operational_side_effects():
    """Statically: no connection, engine, inspection, creation, or arg parsing at import."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(init_db.__file__).read_text(encoding="utf-8"))
    top = [n for n in tree.body
           if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.If, ast.Import,
                                 ast.ImportFrom))]
    called = {
        (c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", ""))
        for node in top for c in ast.walk(node) if isinstance(c, ast.Call)
    }
    forbidden = {"get_db_connection", "create_engine", "create_tables", "inspect",
                 "parse_args", "print", "connect"}
    assert not (called & forbidden), f"module-level side effects: {sorted(called & forbidden)}"
