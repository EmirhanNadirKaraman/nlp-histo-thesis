"""Regression test for sync MAP per-voter persistence (Phase 0 audit).

Asserts ``SummarizationRunner._persist_voter_outputs`` constructs the
expected ``SumMapVoterOutput`` rows from buffered per-voter records and
hands them to ``db.session_scope().bulk_save_objects``. No LLM, no real
DB, no scispaCy load.

Companion to [`docs/CALIBRATION_EXECUTION_PLAN.md`](../../docs/CALIBRATION_EXECUTION_PLAN.md)
§10 Phase 0; complements the existing ``test_map_voter_cache.py`` (which
covers the in-run dedup cache, not persistence).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

# Skip scispaCy/UMLS load — never touched by this test.
os.environ.setdefault("NLP_HISTO_DISABLE_UMLS", "1")
os.environ.setdefault("NLP_HISTO_SKIP_UMLS_ENRICHMENT", "1")

from database.models import SumMapVoterOutput
from pipeline.stages.summarization.runner import SummarizationRunner


PMCID = "PMC_TEST_VOTER_PERSIST"


_SENTINEL = object()


def _record(
    chunk_id: str = "C0",
    level: str = "l1",
    voter_index: int = 0,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    is_selected: bool = False,
    failed: bool = False,
    finding_count: int = 3,
    latency_ms: float | None = 123.4,
    raw_output=_SENTINEL,
) -> dict:
    """Matches ``MapStage._buffer_voter_outputs`` row shape exactly.

    ``raw_output`` honours an explicit ``None`` (failed voters) by using a
    sentinel default — a plain ``None`` default with ``or`` would silently
    coerce ``None`` to the fallback dict.
    """
    if raw_output is _SENTINEL:
        raw_output = {"chunk_id": chunk_id, "findings": []}
    return {
        "pmcid":         PMCID,
        "chunk_id":      chunk_id,
        "level":         level,
        "voter_index":   voter_index,
        "provider":      provider,
        "model":         model,
        "is_selected":   is_selected,
        "failed":        failed,
        "error_message": None,
        "finding_count": finding_count,
        "latency_ms":    latency_ms,
        "raw_output":    raw_output,
    }


def _make_runner(records: list[dict]) -> tuple[SummarizationRunner, MagicMock]:
    """Build a bare runner stub wired to a mock DB session.

    ``SummarizationRunner.__init__`` requires LLMs that we don't want to
    construct here, so the stub is allocated with ``__new__`` and only the
    attributes ``_persist_voter_outputs`` touches are populated:
    ``self._db.session_scope()`` and ``self._map.voter_output_records()``.
    """
    runner = object.__new__(SummarizationRunner)

    map_stub = MagicMock(name="MapStage")
    map_stub.voter_output_records.return_value = records
    runner._map = map_stub

    session = MagicMock(name="Session")
    cm = MagicMock(name="session_scope_cm")
    cm.__enter__ = MagicMock(return_value=session)
    cm.__exit__ = MagicMock(return_value=False)
    db_stub = MagicMock(name="DB")
    db_stub.session_scope = MagicMock(return_value=cm)
    runner._db = db_stub

    return runner, session


def test_persist_voter_outputs_writes_one_row_per_record() -> None:
    """End-to-end: records from MapStage → SumMapVoterOutput rows via bulk_save_objects."""
    records = [
        _record(chunk_id="C0", level="l1", voter_index=0, provider="openai",
                model="gpt-4o-mini", is_selected=True,  finding_count=4),
        _record(chunk_id="C0", level="l1", voter_index=1, provider="gemini",
                model="gemini-2.5-flash-lite", is_selected=False, finding_count=2),
        _record(chunk_id="C0", level="l2", voter_index=0, provider="claude",
                model="claude-haiku-4-5", is_selected=False, failed=True,
                finding_count=0, raw_output=None, latency_ms=None),
        _record(chunk_id="C1", level="l1", voter_index=0, provider="openai",
                model="gpt-4o-mini", is_selected=True,  finding_count=1),
    ]
    runner, session = _make_runner(records)

    runner._persist_voter_outputs(db_id=42, pmcid=PMCID)

    assert session.bulk_save_objects.call_count == 1
    rows = session.bulk_save_objects.call_args[0][0]
    assert len(rows) == len(records)
    assert all(isinstance(r, SumMapVoterOutput) for r in rows)

    # Spot-check the first row carries every record field through to ORM.
    r0 = rows[0]
    assert r0.pipeline_run_id == 42
    assert r0.pmcid == PMCID
    assert r0.chunk_id == "C0"
    assert r0.level == "l1"
    assert r0.voter_index == 0
    assert r0.provider == "openai"
    assert r0.model == "gpt-4o-mini"
    assert r0.is_selected is True
    assert r0.failed is False
    assert r0.finding_count == 4
    assert r0.latency_ms == 123.4
    assert isinstance(r0.raw_output, dict)

    # Verify is_selected polarity preserved across rows.
    selected = [(r.chunk_id, r.level, r.voter_index) for r in rows if r.is_selected]
    assert selected == [("C0", "l1", 0), ("C1", "l1", 0)]

    # Failed voter row: raw_output None, latency None, finding_count 0.
    failed_rows = [r for r in rows if r.failed]
    assert len(failed_rows) == 1
    fr = failed_rows[0]
    assert fr.raw_output is None
    assert fr.latency_ms is None
    assert fr.finding_count == 0


def test_persist_voter_outputs_noop_when_db_id_none() -> None:
    """Sync runs without DB ingestion must not crash and must not query records."""
    records = [_record()]
    runner, session = _make_runner(records)

    runner._persist_voter_outputs(db_id=None, pmcid=PMCID)

    session.bulk_save_objects.assert_not_called()
    # Should not even reach voter_output_records lookup.
    runner._map.voter_output_records.assert_not_called()


def test_persist_voter_outputs_noop_when_records_empty() -> None:
    """Cache-hit-only paper with no buffered voter rows: no DB write."""
    runner, session = _make_runner(records=[])

    runner._persist_voter_outputs(db_id=99, pmcid=PMCID)

    session.bulk_save_objects.assert_not_called()


def test_persist_voter_outputs_swallows_db_exception() -> None:
    """Voter persistence is observability-only; failure must not raise."""
    records = [_record()]
    runner, session = _make_runner(records)
    session.bulk_save_objects.side_effect = RuntimeError("simulated DB outage")

    # Must not raise. If it ever does, the entire paper run would abort
    # over a non-load-bearing audit row.
    runner._persist_voter_outputs(db_id=42, pmcid=PMCID)
