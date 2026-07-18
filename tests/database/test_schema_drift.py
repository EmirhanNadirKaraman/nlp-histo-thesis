"""Static ORM-vs-migration drift characterization — fail-closed.

Compares the schema surface implied by Alembic revisions 0001–0014 with the
current ``database.models.Base.metadata`` and asserts the *exact* set of
divergences that have been reviewed and approved (B-097 / B-098).

**This is a characterization lint, not proof of PostgreSQL parity.** The
revisions are parsed with :mod:`ast` — they are never imported or executed —
and only the operation surface those revisions actually use is modelled.
Anything else fails.

Real (disposable) PostgreSQL migration testing is still required for:

* operator classes and expression indexes;
* server-default *equivalence* (we compare declarations, not resolved DDL);
* enum evolution;
* dialect-specific implicit behaviour;
* data-dependent migration effects (e.g. the 0009 backfills);
* lock and transaction behaviour.

Nothing here connects to PostgreSQL, builds an engine, runs Alembic, or
executes SQL.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from nlp_histo.database.models import Base

from tests.paths import REPO_ROOT as _REPO_ROOT
_VERSIONS = _REPO_ROOT / "alembic" / "versions"
_DIALECT = postgresql.dialect()


class UnsupportedMigration(AssertionError):
    """Raised when the chain uses something this parser does not model."""


# Type normalization (fail-closed)
# Both sides are normalized by compiling a real SQLAlchemy type with the
# PostgreSQL dialect, so the two inventories are directly comparable.

def _compile_type(t: sa.types.TypeEngine) -> str:
    return t.compile(_DIALECT)


_ZERO_ARG_TYPES = {
    "Integer": sa.Integer,
    "Text": sa.Text,
    "Boolean": sa.Boolean,
    "Float": sa.Float,
    "TIMESTAMP": sa.TIMESTAMP,
    "JSON": sa.JSON,
    "JSONB": postgresql.JSONB,
}


def _type_from_ast(node: ast.expr, where: str) -> str:
    """Map a revision's type expression to a normalized PostgreSQL type string.

    Only the exact expressions used by revisions 0001–0014 are supported.
    """
    # Bare attribute/name:  sa.Integer, sa.Text, JSONB, ...
    if isinstance(node, (ast.Attribute, ast.Name)):
        name = node.attr if isinstance(node, ast.Attribute) else node.id
        if name in _ZERO_ARG_TYPES:
            return _compile_type(_ZERO_ARG_TYPES[name]())
        raise UnsupportedMigration(
            f"{where}: unsupported type expression {ast.unparse(node)!r}"
        )

    if isinstance(node, ast.Call):
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")

        if fname in _ZERO_ARG_TYPES and not node.args and not node.keywords:
            return _compile_type(_ZERO_ARG_TYPES[fname]())

        if fname == "String":
            length = None
            if node.args and isinstance(node.args[0], ast.Constant):
                length = node.args[0].value
            for kw in node.keywords:
                if kw.arg == "length" and isinstance(kw.value, ast.Constant):
                    length = kw.value.value
            if not isinstance(length, int):
                raise UnsupportedMigration(f"{where}: String() without an integer length")
            return _compile_type(sa.String(length))

        if fname == "ARRAY":
            if len(node.args) != 1:
                raise UnsupportedMigration(f"{where}: ARRAY() must take exactly one element type")
            inner = _type_from_ast(node.args[0], where)
            if inner != _compile_type(sa.Text()):
                raise UnsupportedMigration(f"{where}: only ARRAY(Text) is modelled, got {inner}")
            return _compile_type(postgresql.ARRAY(sa.Text()))

    raise UnsupportedMigration(f"{where}: unsupported type expression {ast.unparse(node)!r}")


# Approved raw SQL (each op.execute must match exactly)

def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").lower()


# Schema-affecting raw SQL → its precise modelled effect.
_RAW_SQL_SCHEMA = {
    _norm_sql("ALTER TABLE entities ADD COLUMN IF NOT EXISTS semantic_types VARCHAR;"):
        ("add_column", "entities", "semantic_types", _compile_type(sa.String()), True),
}

# Data-only backfills → approved historical data operations with no schema effect.
# Preserved here so the characterization fails if the SQL ever changes.
_RAW_SQL_DATA_ONLY = {
    _norm_sql("""
        UPDATE sum_canonical_rules
        SET
            is_conflicted  = (canonical_scope = 'conflicted'),
            study_coverage = CASE
                WHEN canonical_scope IN ('multi_study', 'conflicted') THEN 'multi_study'
                WHEN canonical_scope = 'single_study'                 THEN 'single_study'
                ELSE                                                       'unknown'
            END
    """): "0009 backfill: sum_canonical_rules.canonical_scope -> is_conflicted/study_coverage",
    _norm_sql("""
        UPDATE sum_corpus_relations
        SET
            is_conflicted_a  = (canonical_scope_a = 'conflicted'),
            study_coverage_a = CASE
                WHEN canonical_scope_a IN ('multi_study', 'conflicted') THEN 'multi_study'
                WHEN canonical_scope_a = 'single_study'                 THEN 'single_study'
                ELSE                                                         'unknown'
            END,
            is_conflicted_b  = (canonical_scope_b = 'conflicted'),
            study_coverage_b = CASE
                WHEN canonical_scope_b IN ('multi_study', 'conflicted') THEN 'multi_study'
                WHEN canonical_scope_b = 'single_study'                 THEN 'single_study'
                ELSE                                                         'unknown'
            END
    """): "0009 backfill: sum_corpus_relations.canonical_scope_{a,b} -> is_conflicted/study_coverage",
}


def _orm_server_default(col: sa.Column) -> str | None:
    """DDL server default (``server_default=``) only — this *is* schema.

    ``Column(default=...)`` is client-side and is deliberately not returned here;
    see :func:`_orm_client_default`. The two must never compare equal.
    """
    if col.server_default is None:
        return None
    arg = col.server_default.arg
    return str(getattr(arg, "text", arg))


def _orm_client_default(col: sa.Column) -> str | None:
    """Client-side default (``Column(default=...)``) — applied by SQLAlchemy at INSERT.

    This is **not** DDL: a table created from this metadata carries no ``DEFAULT``
    clause for the column. Tracked separately so an ORM client default can never be
    mistaken for an Alembic server default.
    """
    d = col.default
    if d is None:
        return None
    arg = getattr(d, "arg", None)
    if arg is None:
        return None
    if isinstance(arg, (str, bool, int, float)):    # plain Python scalar
        return str(arg)
    if hasattr(arg, "compile"):                     # SQL expression, e.g. func.now()
        return str(arg.compile(dialect=_DIALECT))
    raise UnsupportedMigration(                     # pragma: no cover - fail closed
        f"{col.table.name}.{col.name}: unsupported client default {arg!r}")


# Revision graph

def _parse_revisions() -> dict[str, dict]:
    revs: dict[str, dict] = {}
    for path in sorted(_VERSIONS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rev = down = None
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if not isinstance(target, ast.Name) or node.value is None:
                    continue
                if target.id == "revision" and isinstance(node.value, ast.Constant):
                    rev = node.value.value
                elif target.id == "down_revision":
                    down = node.value.value if isinstance(node.value, ast.Constant) else None
        if rev is None:
            raise UnsupportedMigration(f"{path.name}: no revision identifier found")
        if rev in revs:
            raise UnsupportedMigration(f"duplicate revision identifier {rev!r}")
        revs[rev] = {"down": down, "path": path, "tree": tree}
    return revs


def _ordered_chain(revs: dict[str, dict]) -> list[str]:
    """Validate the graph and return revisions in dependency order."""
    bases = [r for r, m in revs.items() if m["down"] is None]
    if len(bases) != 1:
        raise UnsupportedMigration(f"expected exactly one base revision, found {sorted(bases)}")

    children: dict[str, list[str]] = {}
    for r, m in revs.items():
        if m["down"] is not None:
            if m["down"] not in revs:
                raise UnsupportedMigration(f"revision {r!r} references unknown down_revision {m['down']!r}")
            children.setdefault(m["down"], []).append(r)

    heads = [r for r in revs if r not in children]
    if len(heads) != 1:
        raise UnsupportedMigration(f"expected exactly one head, found {sorted(heads)}")

    order, cur = [], bases[0]
    while cur is not None:
        order.append(cur)
        kids = children.get(cur, [])
        if len(kids) > 1:
            raise UnsupportedMigration(f"branch point at {cur!r}: {sorted(kids)}")
        cur = kids[0] if kids else None
    if len(order) != len(revs):
        raise UnsupportedMigration("revision graph is not a single continuous path")
    return order


# Replay the chain (AST only — nothing is imported or executed)

_SUPPORTED_OPS = {
    "create_table", "add_column", "drop_column", "alter_column",
    "create_index", "create_unique_constraint", "execute",
}

# Column() kwargs the parser models. `default=` is deliberately absent: a client-side
# default in a migration would not be DDL, and silently ignoring it would hide drift.
_COLUMN_KWARGS = {"nullable", "primary_key", "server_default", "unique", "autoincrement"}


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _const(node, default=None):
    return node.value if isinstance(node, ast.Constant) else default


def _server_default_from_ast(node: ast.expr | None, where: str) -> str | None:
    """Normalize a revision's ``server_default=`` to canonical SQL text.

    ``sa.text('now()')`` and ``sa.func.now()`` both render as ``now()``; a literal
    string renders as itself. Anything else fails closed.
    """
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call):
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if fname == "text" and node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value
        if fname == "now" and not node.args:          # sa.func.now()
            return "now()"
    raise UnsupportedMigration(
        f"{where}: unsupported server_default expression {ast.unparse(node)!r}")


def _replay() -> dict:
    revs = _parse_revisions()
    order = _ordered_chain(revs)

    tables: dict[str, dict] = {}          # created by the chain
    altered_cols: dict[tuple[str, str], dict] = {}   # (table, col) for tables the chain only ALTERs
    indexes: dict[str, dict] = {}
    uniques: dict[str, dict] = {}
    data_ops: list[str] = []

    created = set()

    def _column_from_ast(call: ast.Call, where: str) -> dict:
        if not call.args or not isinstance(call.args[0], ast.Constant):
            raise UnsupportedMigration(f"{where}: Column() without a literal name")
        name = call.args[0].value
        if len(call.args) < 2:
            raise UnsupportedMigration(f"{where}: Column({name!r}) without a type")
        given = {k.arg for k in call.keywords}
        if not given <= _COLUMN_KWARGS:
            raise UnsupportedMigration(
                f"{where}: Column({name!r}) uses unmodelled kwargs "
                f"{sorted(given - _COLUMN_KWARGS)} — only {sorted(_COLUMN_KWARGS)} are modelled")
        col = {
            "type": _type_from_ast(call.args[1], where),
            "nullable": _const(_kw(call, "nullable"), True),
            "primary_key": bool(_const(_kw(call, "primary_key"), False)),
            "server_default": _server_default_from_ast(_kw(call, "server_default"), where),
            # Revisions never use Column(default=...) — a client default in a migration
            # would be meaningless. _COLUMN_KWARGS rejects it, so this is always None.
            "client_default": None,
            "unique": bool(_const(_kw(call, "unique"), False)),
            "fk": None,
        }
        if col["primary_key"]:
            col["nullable"] = False
        for extra in call.args[2:]:
            if isinstance(extra, ast.Call) and getattr(extra.func, "attr", "") == "ForeignKey":
                target = _const(extra.args[0]) if extra.args else None
                ondelete = _const(_kw(extra, "ondelete"))
                col["fk"] = {"target": target, "ondelete": ondelete}
            else:
                raise UnsupportedMigration(f"{where}: unsupported Column arg {ast.unparse(extra)!r}")
        return name, col

    for rev in order:
        tree = revs[rev]["tree"]
        where_rev = f"revision {rev}"
        upgrade = next((n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "upgrade"), None)
        if upgrade is None:
            raise UnsupportedMigration(f"{where_rev}: no upgrade() function")

        for node in ast.walk(upgrade):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and getattr(node.func.value, "id", None) == "op"):
                continue
            kind = node.func.attr
            where = f"{where_rev}: op.{kind}"
            if kind not in _SUPPORTED_OPS:
                raise UnsupportedMigration(f"{where}: unsupported operation — {ast.unparse(node)[:90]!r}")

            if kind == "create_table":
                tname = _const(node.args[0])
                created.add(tname)
                tables[tname] = {"columns": {}, "uniques": {}}
                for arg in node.args[1:]:
                    if not isinstance(arg, ast.Call):
                        raise UnsupportedMigration(f"{where}({tname}): unsupported element {ast.unparse(arg)!r}")
                    fname = getattr(arg.func, "attr", "")
                    if fname == "Column":
                        cname, col = _column_from_ast(arg, f"{where}({tname})")
                        tables[tname]["columns"][cname] = col
                    elif fname == "UniqueConstraint":      # inline — must be captured
                        cols = tuple(_const(a) for a in arg.args if isinstance(a, ast.Constant))
                        uname = _const(_kw(arg, "name"))
                        if not uname:
                            raise UnsupportedMigration(f"{where}({tname}): unnamed inline UniqueConstraint")
                        uniques[uname] = {"table": tname, "columns": cols}
                    else:
                        raise UnsupportedMigration(
                            f"{where}({tname}): unsupported table element {ast.unparse(arg)[:70]!r}")

            elif kind == "add_column":
                tname = _const(node.args[0])
                cname, col = _column_from_ast(node.args[1], where)
                if tname in created:
                    tables[tname]["columns"][cname] = col
                else:
                    altered_cols[(tname, cname)] = col

            elif kind == "drop_column":
                tname, cname = _const(node.args[0]), _const(node.args[1])
                if tname in created:
                    tables[tname]["columns"].pop(cname, None)
                else:
                    altered_cols.pop((tname, cname), None)

            elif kind == "alter_column":
                tname, cname = _const(node.args[0]), _const(node.args[1])
                supported = {"nullable"}
                given = {k.arg for k in node.keywords}
                if not given <= supported:
                    raise UnsupportedMigration(
                        f"{where}({tname}.{cname}): only {sorted(supported)} modelled, got {sorted(given)}")
                nullable = _const(_kw(node, "nullable"))
                if tname in created and cname in tables[tname]["columns"]:
                    tables[tname]["columns"][cname]["nullable"] = nullable
                elif (tname, cname) in altered_cols:
                    altered_cols[(tname, cname)]["nullable"] = nullable
                else:
                    raise UnsupportedMigration(f"{where}: alter of unknown column {tname}.{cname}")

            elif kind == "create_index":
                iname, tname = _const(node.args[0]), _const(node.args[1])
                if not (len(node.args) > 2 and isinstance(node.args[2], ast.List)):
                    raise UnsupportedMigration(f"{where}({iname}): index columns must be a literal list")
                cols = tuple(_const(c) for c in node.args[2].elts)
                extra = {k.arg for k in node.keywords} - {"unique"}
                if extra:
                    raise UnsupportedMigration(f"{where}({iname}): unmodelled kwargs {sorted(extra)}")
                indexes[iname] = {
                    "table": tname, "columns": cols,
                    "unique": bool(_const(_kw(node, "unique"), False)),
                }

            elif kind == "create_unique_constraint":
                uname, tname = _const(node.args[0]), _const(node.args[1])
                if not (len(node.args) > 2 and isinstance(node.args[2], ast.List)):
                    raise UnsupportedMigration(f"{where}({uname}): columns must be a literal list")
                uniques[uname] = {
                    "table": tname,
                    "columns": tuple(_const(c) for c in node.args[2].elts),
                }

            elif kind == "execute":
                if not (node.args and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    raise UnsupportedMigration(f"{where}: only literal SQL strings are modelled")
                sql = _norm_sql(node.args[0].value)
                if sql in _RAW_SQL_SCHEMA:
                    kind_, tname, cname, ctype, nullable = _RAW_SQL_SCHEMA[sql]
                    assert kind_ == "add_column"
                    altered_cols[(tname, cname)] = {
                        "type": ctype, "nullable": nullable, "primary_key": False,
                        "server_default": None, "unique": False, "fk": None,
                    }
                elif sql in _RAW_SQL_DATA_ONLY:
                    data_ops.append(_RAW_SQL_DATA_ONLY[sql])   # no schema effect
                else:
                    raise UnsupportedMigration(
                        f"{where}: unrecognised raw SQL — every op.execute must be reviewed and "
                        f"modelled explicitly. Normalized: {sql[:120]!r}")

    return {
        "tables": tables, "altered_cols": altered_cols,
        "indexes": indexes, "uniques": uniques,
        "data_ops": data_ops, "order": order,
    }


# ORM normalization

def _orm() -> dict:
    tables, indexes, uniques = {}, {}, {}
    for tname, t in Base.metadata.tables.items():
        cols = {}
        for c in t.columns:
            fk = None
            for f in c.foreign_keys:
                fk = {"target": f.target_fullname, "ondelete": f.ondelete}
            cols[c.name] = {
                "type": _compile_type(c.type),
                "nullable": c.nullable,
                "primary_key": c.primary_key,
                # server_default is DDL; client_default (Column(default=...)) is not.
                # They are tracked separately and never compared as equivalent.
                "server_default": _orm_server_default(c),
                "client_default": _orm_client_default(c),
                "unique": bool(c.unique),
                "fk": fk,
            }
        tables[tname] = {"columns": cols}
        for ix in t.indexes:
            indexes[ix.name] = {
                "table": tname,
                "columns": tuple(c.name for c in ix.columns),
                "unique": bool(ix.unique),
                # keep auto (Column(index=True)) distinct from explicit Index(...)
                "auto": bool(getattr(ix, "_column_flag", False)),
            }
        for con in t.constraints:
            if isinstance(con, sa.UniqueConstraint) and con.name:
                uniques[con.name] = {"table": tname, "columns": tuple(c.name for c in con.columns)}
    return {"tables": tables, "indexes": indexes, "uniques": uniques}


# The approved divergence set (B-097 / B-098)

APPROVED_ORM_ONLY_COLUMNS = {("pipeline_runs", "narrative_summary")}

APPROVED_TYPE_DIFFS = {
    ("entities", "semantic_types"): (
        _compile_type(postgresql.ARRAY(sa.Text())),   # ORM
        _compile_type(sa.String()),                   # Alembic (0006 raw SQL)
    ),
}

# Divergence categories. Kept distinct so a failure says *what kind* of drift it is.
ORM_ONLY_COLUMN = "orm_only_column"
ALEMBIC_ONLY_COLUMN = "alembic_only_column"
TYPE_MISMATCH = "type_mismatch"
NULLABILITY_MISMATCH = "nullability_mismatch"
SERVER_DEFAULT_MISMATCH = "server_default_mismatch"
CLIENT_VS_SERVER_DEFAULT = "client_default_vs_server_default"

# The exact column-level divergences that currently exist (B-097 / B-098).
# Characterization only — not an endorsement. Each remains a pending defect:
#   * scope_note's nullability difference is substantive (an ORM-created table accepts
#     NULL; a migrated one rejects it);
#   * the three client-vs-server default rows mean a migrated table carries real DDL
#     DEFAULT clauses that an ORM-created table does not.
EXPECTED_COLUMN_DIVERGENCES = frozenset({
    # (category, table, column, alembic_value, orm_value)
    (ORM_ONLY_COLUMN, "pipeline_runs", "narrative_summary", None, "present"),

    (CLIENT_VS_SERVER_DEFAULT, "pipeline_runs", "status", "running", "running"),
    (CLIENT_VS_SERVER_DEFAULT, "pipeline_runs", "started_at", "now()", "now()"),
    (CLIENT_VS_SERVER_DEFAULT, "sum_corpus_relations", "scope_check_result",
     "scope_unknown", "scope_unknown"),

    (NULLABILITY_MISMATCH, "sum_corpus_relations", "scope_note", False, True),
    (SERVER_DEFAULT_MISMATCH, "sum_corpus_relations", "scope_note", "", None),
})

# ORM name -> Alembic name, semantically equivalent apart from the name.
APPROVED_INDEX_RENAMES = {
    "ix_screl_run":     "ix_cor_corpus_run",
    "ix_screl_scope":   "ix_cor_scope",
    "ix_screl_pmcid_a": "ix_cor_pmcid_a",
    "ix_screl_pmcid_b": "ix_cor_pmcid_b",
    "ix_screl_rule_a":  "ix_cor_rule_a",
    "ix_screl_rule_b":  "ix_cor_rule_b",
}

APPROVED_UNIQUE_RENAMES = {"uq_sum_corpus_relation": "uq_corpus_relation"}

# Genuinely missing from the chain (never deployed).
APPROVED_ORM_ONLY_INDEXES = {
    "ix_screl_type",                          # sum_corpus_relations.relation_type
    "ix_sum_corpus_relations_corpus_run_id",  # duplicate of ix_screl_run (Column(index=True))
}


# Tests

def test_revision_graph_is_a_single_continuous_chain():
    revs = _parse_revisions()
    order = _ordered_chain(revs)
    assert order[0] == "0001"
    assert order[-1] == "0014"
    assert order == sorted(order), "chain must be replayed in dependency order"
    assert len(order) == len(revs) == 14


def test_every_op_execute_is_explicitly_modelled():
    """The chain's raw SQL is enumerated; anything else fails (no blanket exemption)."""
    chain = _replay()
    assert sorted(chain["data_ops"]) == sorted(_RAW_SQL_DATA_ONLY.values())
    assert ("entities", "semantic_types") in chain["altered_cols"], \
        "0006's raw ALTER TABLE must be modelled as a schema effect"


def _column_divergences() -> set[tuple]:
    """Every column-level difference on the tables Alembic creates, categorized."""
    chain, orm = _replay(), _orm()
    found: set[tuple] = set()

    for tname, tdef in chain["tables"].items():
        assert tname in orm["tables"], f"table {tname}: created by Alembic but absent from the ORM"
        acols, ocols = tdef["columns"], orm["tables"][tname]["columns"]

        for cname in sorted(set(acols) - set(ocols)):
            found.add((ALEMBIC_ONLY_COLUMN, tname, cname, "present", None))
        for cname in sorted(set(ocols) - set(acols)):
            found.add((ORM_ONLY_COLUMN, tname, cname, None, "present"))

        for cname in sorted(set(acols) & set(ocols)):
            a, o = acols[cname], ocols[cname]

            if a["type"] != o["type"]:
                found.add((TYPE_MISMATCH, tname, cname, a["type"], o["type"]))
            if a["nullable"] != o["nullable"]:
                found.add((NULLABILITY_MISMATCH, tname, cname, a["nullable"], o["nullable"]))

            # Defaults. A client-side ORM default is never equivalent to a server default:
            # when the values coincide we record the *ownership* mismatch; otherwise it is a
            # plain server-default mismatch.
            if a["server_default"] != o["server_default"]:
                if o["server_default"] is None and o["client_default"] is not None \
                        and o["client_default"] == a["server_default"]:
                    found.add((CLIENT_VS_SERVER_DEFAULT, tname, cname,
                               a["server_default"], o["client_default"]))
                else:
                    found.add((SERVER_DEFAULT_MISMATCH, tname, cname,
                               a["server_default"], o["server_default"]))

            assert a["primary_key"] == o["primary_key"], f"{tname}.{cname}: primary-key drift"
            assert (a["fk"] is None) == (o["fk"] is None), f"{tname}.{cname}: foreign-key drift"
            if a["fk"]:
                assert (a["fk"]["target"], a["fk"]["ondelete"]) \
                    == (o["fk"]["target"], o["fk"]["ondelete"]), f"{tname}.{cname}: foreign-key drift"
    return found


def test_column_divergences_are_exactly_the_characterized_set():
    found = _column_divergences()
    new = found - EXPECTED_COLUMN_DIVERGENCES
    gone = EXPECTED_COLUMN_DIVERGENCES - found
    msg = []
    if new:
        msg.append("NEW, uncharacterized drift (stop and review — do not just add it):\n    "
                   + "\n    ".join(repr(d) for d in sorted(map(str, new))))
    if gone:
        msg.append("characterized drift has DISAPPEARED (update the expected set):\n    "
                   + "\n    ".join(repr(d) for d in sorted(map(str, gone))))
    assert not msg, "\n  " + "\n  ".join(msg)


def test_columns_altered_on_orm_owned_tables_match_except_approved_type_diffs():
    """The chain only ALTERs the ORM-created core tables (entities, figures)."""
    chain, orm = _replay(), _orm()
    problems = []
    for (tname, cname), a in chain["altered_cols"].items():
        o = orm["tables"][tname]["columns"].get(cname)
        if o is None:
            problems.append(f"{tname}.{cname}: added by Alembic but absent from the ORM")
            continue
        approved = APPROVED_TYPE_DIFFS.get((tname, cname))
        if approved:
            assert (o["type"], a["type"]) == approved, (
                f"{tname}.{cname}: type divergence changed — orm={o['type']!r} "
                f"alembic={a['type']!r}, approved={approved!r}")
            continue
        if a["type"] != o["type"]:
            problems.append(f"{tname}.{cname}.type: alembic={a['type']!r} orm={o['type']!r}")
        if a["nullable"] != o["nullable"]:
            problems.append(f"{tname}.{cname}.nullable: alembic={a['nullable']} orm={o['nullable']}")
    assert not problems, "unapproved drift on ORM-owned tables:\n  " + "\n  ".join(problems)


def test_indexes_match_except_approved_renames_and_orm_only():
    chain, orm = _replay(), _orm()
    alembic_ix = chain["indexes"]
    # Only compare indexes on tables the chain creates.
    orm_ix = {n: d for n, d in orm["indexes"].items() if d["table"] in chain["tables"]}

    problems = []
    for oname, odef in sorted(orm_ix.items()):
        if oname in alembic_ix:
            a = alembic_ix[oname]
            if (a["columns"], a["unique"]) != (odef["columns"], odef["unique"]):
                problems.append(f"index {oname}: alembic={a} orm={odef}")
        elif oname in APPROVED_INDEX_RENAMES:
            aname = APPROVED_INDEX_RENAMES[oname]
            assert aname in alembic_ix, f"approved rename target {aname} missing from the chain"
            a = alembic_ix[aname]
            # semantically equivalent apart from the name
            assert (a["table"], a["columns"], a["unique"]) == (odef["table"], odef["columns"], odef["unique"]), (
                f"approved rename {oname}<->{aname} is no longer semantically equivalent: "
                f"alembic={a} orm={odef}")
        elif oname in APPROVED_ORM_ONLY_INDEXES:
            continue
        else:
            problems.append(f"index {oname}: in ORM only (unapproved) {odef}")

    renamed_targets = set(APPROVED_INDEX_RENAMES.values())
    for aname in sorted(set(alembic_ix) - set(orm_ix) - renamed_targets):
        problems.append(f"index {aname}: in Alembic only (unapproved) {alembic_ix[aname]}")

    assert not problems, "unapproved index drift:\n  " + "\n  ".join(problems)


def test_the_duplicate_corpus_run_id_index_is_still_visible():
    """The ORM declares BOTH Column(index=True) and an explicit Index on corpus_run_id."""
    orm = _orm()
    dup = orm["indexes"]["ix_sum_corpus_relations_corpus_run_id"]
    explicit = orm["indexes"]["ix_screl_run"]
    assert dup["auto"] is True and explicit["auto"] is False
    assert dup["columns"] == explicit["columns"] == ("corpus_run_id",), \
        "the redundancy must remain visible, not collapsed"


def test_unique_constraints_match_except_the_approved_rename():
    chain, orm = _replay(), _orm()
    a_uq = chain["uniques"]
    o_uq = {n: d for n, d in orm["uniques"].items() if d["table"] in chain["tables"]}

    problems = []
    for oname, odef in sorted(o_uq.items()):
        if oname in a_uq:
            if a_uq[oname]["columns"] != odef["columns"]:
                problems.append(f"unique {oname}: alembic={a_uq[oname]} orm={odef}")
        elif oname in APPROVED_UNIQUE_RENAMES:
            aname = APPROVED_UNIQUE_RENAMES[oname]
            assert aname in a_uq, f"approved rename target {aname} missing from the chain"
            assert a_uq[aname]["columns"] == odef["columns"] and a_uq[aname]["table"] == odef["table"], (
                f"approved rename {oname}<->{aname} no longer equivalent: "
                f"alembic={a_uq[aname]} orm={odef}")
        else:
            problems.append(f"unique {oname}: in ORM only (unapproved) {odef}")

    for aname in sorted(set(a_uq) - set(o_uq) - set(APPROVED_UNIQUE_RENAMES.values())):
        problems.append(f"unique {aname}: in Alembic only (unapproved) {a_uq[aname]}")

    assert not problems, "unapproved unique-constraint drift:\n  " + "\n  ".join(problems)


# Parser self-tests: it must fail closed

def _upgrade_module(body: str) -> ast.Module:
    return ast.parse(f"def upgrade() -> None:\n{body}\n")


def _run_ops(body: str):
    """Replay a synthetic upgrade() body through the same op dispatch."""
    tree = _upgrade_module(body)
    fake = {"0001": {"down": None, "path": pathlib.Path("synthetic.py"), "tree": tree}}
    import unittest.mock as mock
    with mock.patch(f"{__name__}._parse_revisions", return_value=fake):
        return _replay()


def test_client_default_is_never_equivalent_to_a_server_default():
    """`Column(default=...)` is client-side; `server_default=` is DDL. Never equal."""
    orm = _orm()
    status = orm["tables"]["pipeline_runs"]["columns"]["status"]
    started = orm["tables"]["pipeline_runs"]["columns"]["started_at"]

    # The ORM declares client defaults and NO server defaults …
    assert status["client_default"] == "running" and status["server_default"] is None
    assert started["client_default"] == "now()" and started["server_default"] is None

    # … while the chain declares server defaults with the same values.
    chain = _replay()
    a_status = chain["tables"]["pipeline_runs"]["columns"]["status"]
    a_started = chain["tables"]["pipeline_runs"]["columns"]["started_at"]
    assert a_status["server_default"] == "running" and a_status["client_default"] is None
    assert a_started["server_default"] == "now()" and a_started["client_default"] is None

    # Equal *values* must still be reported as an ownership divergence, not folded away.
    for col in ("status", "started_at"):
        assert (CLIENT_VS_SERVER_DEFAULT, "pipeline_runs", col,
                orm["tables"]["pipeline_runs"]["columns"][col]["client_default"],
                orm["tables"]["pipeline_runs"]["columns"][col]["client_default"]
                ) in _column_divergences()


def test_scope_note_records_both_nullability_and_server_default():
    """The substantive scope_note defect must not collapse into one vague record."""
    found = _column_divergences()
    assert (NULLABILITY_MISMATCH, "sum_corpus_relations", "scope_note", False, True) in found
    assert (SERVER_DEFAULT_MISMATCH, "sum_corpus_relations", "scope_note", "", None) in found


def test_parser_rejects_unmodelled_column_kwargs():
    with pytest.raises(UnsupportedMigration, match="unmodelled kwargs"):
        _run_ops('    op.create_table("t", sa.Column("c", sa.Text, default="x"))')


def test_parser_rejects_an_unknown_op():
    with pytest.raises(UnsupportedMigration, match="unsupported operation"):
        _run_ops('    op.rename_table("a", "b")')


def test_parser_rejects_unrecognised_raw_sql():
    with pytest.raises(UnsupportedMigration, match="unrecognised raw SQL"):
        _run_ops('    op.execute("DROP TABLE entities")')


def test_parser_rejects_non_literal_raw_sql():
    with pytest.raises(UnsupportedMigration, match="literal SQL strings"):
        _run_ops("    op.execute(some_variable)")


def test_parser_captures_inline_unique_constraints():
    chain = _run_ops(
        '    op.create_table("t",\n'
        '        sa.Column("id", sa.Integer, primary_key=True),\n'
        '        sa.Column("a", sa.Text),\n'
        '        sa.UniqueConstraint("a", name="uq_inline"),\n'
        "    )"
    )
    assert chain["uniques"]["uq_inline"] == {"table": "t", "columns": ("a",)}


def test_parser_rejects_unsupported_type_expression():
    with pytest.raises(UnsupportedMigration, match="unsupported type expression"):
        _run_ops(
            '    op.create_table("t", sa.Column("c", sa.LargeBinary))'
        )


def test_parser_rejects_unmodelled_alter_column_kwargs():
    with pytest.raises(UnsupportedMigration, match="only \\['nullable'\\] modelled"):
        _run_ops(
            '    op.create_table("t", sa.Column("c", sa.Text))\n'
            '    op.alter_column("t", "c", type_=sa.Integer)'
        )


def test_indexes_are_keyed_by_name_not_collapsed_by_columns():
    """Two identically-defined indexes with different names must stay distinct."""
    chain = _run_ops(
        '    op.create_table("t", sa.Column("c", sa.Text))\n'
        '    op.create_index("ix_one", "t", ["c"])\n'
        '    op.create_index("ix_two", "t", ["c"])'
    )
    assert set(chain["indexes"]) == {"ix_one", "ix_two"}
