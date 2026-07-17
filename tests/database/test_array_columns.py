"""B-121 — the array columns must carry PostgreSQL's operators, not the generic stubs.

`path_list` was declared with SQLAlchemy's *generic* `ARRAY`, whose `.contains()` raises
`NotImplementedError` by design: without a dialect it cannot know the containment syntax.
So the query CLAUDE.md documents as a Critical Pattern — every paragraph under a "Methods"
heading — could not run at all.

Nothing caught it because nothing in the pipeline queries by path: `path_list` is written
by ingest and read back whole, so the array operators only matter to a human exploring the
corpus. That is the one surface with no automated coverage, and it is exactly what the
corpus is handed to a reader to do.

These tests need no database. They assert what the *type* compiles to, which is where the
bug lived — a live query would also have caught it, but only where PostgreSQL is running.
"""
from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql

from nlp_histo.database.models import Entity, TextElement

_PG = postgresql.dialect()


def _sql(expr) -> str:
    return str(expr.compile(dialect=_PG))


def test_the_documented_hierarchical_query_compiles() -> None:
    """The exact pattern from CLAUDE.md. Before the fix this raised NotImplementedError."""
    sql = _sql(TextElement.path_list.contains(["Methods"]))
    assert "@>" in sql, f"containment must compile to the PostgreSQL @> operator: {sql}"


def test_contains_does_not_raise_not_implemented() -> None:
    """Pin the failure mode itself: the generic type raises rather than mis-compiling, so
    a regression is an exception, not wrong SQL."""
    try:
        TextElement.path_list.contains(["Methods"])
    except NotImplementedError as exc:  # pragma: no cover — the bug being pinned
        pytest.fail(f"path_list is back on the generic ARRAY type: {exc}")


def test_array_columns_are_dialect_typed() -> None:
    """Not just `path_list` — the provenance arrays gained the operators too, and a future
    column added with the generic import would be the same bug again."""
    for col in (TextElement.path_list, Entity.semantic_types):
        assert isinstance(col.type, postgresql.ARRAY), (
            f"{col} must use the postgresql ARRAY; the generic one has no operators"
        )


def test_ddl_is_unchanged_by_the_dialect_type() -> None:
    """Why B-121 needed no migration: both types emit TEXT[]. If this ever fails, the
    hosted corpus dump and every existing database are affected and a migration is due.
    """
    from sqlalchemy import ARRAY as GenericARRAY
    from sqlalchemy import Column, MetaData, Table, Text
    from sqlalchemy.schema import CreateTable

    def ddl(array_type) -> str:
        t = Table("t", MetaData(), Column("p", array_type))
        return " ".join(str(CreateTable(t).compile(dialect=_PG)).split())

    assert ddl(GenericARRAY(Text)) == ddl(postgresql.ARRAY(Text)) == "CREATE TABLE t ( p TEXT[] )"
