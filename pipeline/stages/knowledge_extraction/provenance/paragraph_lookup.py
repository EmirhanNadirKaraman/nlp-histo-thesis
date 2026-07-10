"""Lazy paragraph lookup for CanonicalRule.

A CanonicalRule carries ``representative_text_element_id`` as a pointer to
the source paragraph (``text_elements.text_content``). The paragraph itself
is not baked into per-paper JSON because it's ~10× the size of the rest of
the rule data — for 15 papers, that's the difference between ~3 MB and
~30 MB of artifacts.

This helper resolves that pointer lazily, given an active DB session. It is
the entry point for downstream consumers (LLM-judge prompts, inspector HTML
context-popovers, audit tools) that want the full paragraph for a rule.

For NLI input, prefer ``CanonicalRule.representative_verbatim`` (already in
the rule object, no DB call) — verbatim is the exact source sentence,
paragraph is the surrounding context.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from pipeline.stages.knowledge_extraction.models import CanonicalRule

logger = logging.getLogger(__name__)


def get_paragraph_for_rule(rule: "CanonicalRule", db_session) -> str | None:
    """Return the full ``text_elements.text_content`` for one rule.

    Returns ``None`` when:
      - ``rule.representative_text_element_id`` is None (rule pre-dates the
        field, or canonicalize had no SourceSpan to attach)
      - the text_element_id does not resolve in this DB (paper not ingested,
        wrong DB connection, …)
      - the lookup raises (logged as a warning; callers should treat None
        as "not available", not a hard error)

    Parameters
    ----------
    rule:
        The CanonicalRule whose paragraph we want.
    db_session:
        An active SQLAlchemy session (from
        ``DatabaseConnection.session_scope()``).
    """
    te_id = getattr(rule, "representative_text_element_id", None)
    if te_id is None:
        return None
    try:
        from sqlalchemy import text  # noqa: PLC0415
        row = db_session.execute(
            text(
                "SELECT text_content FROM text_elements WHERE id = :te_id LIMIT 1"
            ),
            {"te_id": int(te_id)},
        ).fetchone()
        return row.text_content if row is not None else None
    except Exception as exc:  # noqa: BLE001 — non-fatal, observability-only
        logger.warning(
            "get_paragraph_for_rule(%s): lookup failed for "
            "text_element_id=%s — returning None (%s)",
            getattr(rule, "canonical_id", "?"), te_id, exc,
        )
        return None


def get_paragraphs_for_rules(
    rules: Iterable["CanonicalRule"], db_session,
) -> dict[str, str | None]:
    """Batch variant: ``canonical_id → paragraph text or None``.

    Uses a single ``WHERE id IN (...)`` query so N rules cost 1 round-trip
    instead of N. Order-independent; callers reindex by canonical_id.

    De-duplicates ``text_element_id``s before the SQL call — multiple rules
    grounded in the same paragraph each get the same string without
    re-fetching.
    """
    results: dict[str, str | None] = {}
    rules_list = list(rules)
    if not rules_list:
        return results

    te_ids = {
        int(r.representative_text_element_id)
        for r in rules_list
        if getattr(r, "representative_text_element_id", None) is not None
    }
    if not te_ids:
        return {r.canonical_id: None for r in rules_list}

    try:
        from sqlalchemy import text  # noqa: PLC0415
        rows = db_session.execute(
            text(
                "SELECT id, text_content FROM text_elements WHERE id = ANY(:ids)"
            ),
            {"ids": list(te_ids)},
        ).fetchall()
        by_id: dict[int, str] = {row.id: row.text_content for row in rows}
    except Exception as exc:  # noqa: BLE001 — non-fatal
        logger.warning(
            "get_paragraphs_for_rules: batch lookup failed (%d ids) — "
            "returning all-None map (%s)", len(te_ids), exc,
        )
        return {r.canonical_id: None for r in rules_list}

    for r in rules_list:
        te_id = getattr(r, "representative_text_element_id", None)
        results[r.canonical_id] = by_id.get(int(te_id)) if te_id is not None else None
    return results
