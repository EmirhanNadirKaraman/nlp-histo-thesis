"""Regression test for Issue A (this session): failed batch jobs must
surface loudly.

Pre-fix the runner silently skipped jobs with `status="failed"` — a single
log line per failed job, no aggregation, no exception, no impact on the cost
report. B-020 hid for weeks because half the OpenAI batches were rejected
without anything in stdout or the cost report mentioning it.

This test patches `BatchKnowledgeExtractionRunner.advance()`'s provider-check step
so a chosen subset of jobs come back with `status="failed"`, then asserts:

* All-jobs-failed → RuntimeError raised (cascade can't proceed on empty
  results; better to fail loudly than to silently bill an L3-only re-run).
* Partial-failure → WARNING log emitted naming provider / model / job_id /
  request_count for each failed job, plus a summary line.
"""
from __future__ import annotations

import logging

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.batch import runner as runner_mod
from nlp_histo.pipeline.stages.knowledge_extraction.batch.models import (
    BatchHandle,
    BatchPhase,
    ProviderJob,
)


class _FakeProvider:
    """Stub that does nothing on check() — we set statuses directly on the jobs
    before calling advance(). retrieve() returns an empty list for the
    non-failed jobs so the post-collect path is a no-op."""

    def check(self, job: ProviderJob) -> ProviderJob:  # pragma: no cover — pass-through
        return job

    def retrieve(self, job: ProviderJob):
        return []


def _make_handle(pmcid: str, jobs: list[ProviderJob]) -> BatchHandle:
    return BatchHandle(
        pmcid=pmcid,
        phase=BatchPhase.L1_SUBMITTED,
        jobs=jobs,
    )


def _patch_providers(monkeypatch):
    """Make build_providers return our fake regardless of provider name."""
    fake = _FakeProvider()
    monkeypatch.setattr(
        runner_mod, "build_providers",
        lambda providers_needed: {p: fake for p in providers_needed},
    )


def test_all_jobs_failed_raises_runtime_error(monkeypatch, tmp_path):
    """If every job in the current phase failed, advance() must raise
    rather than silently producing empty results."""
    _patch_providers(monkeypatch)

    jobs = [
        ProviderJob(provider="openai", job_id="batch_a", status="failed",
                    model="gpt-4o-mini",   request_count=15),
        ProviderJob(provider="openai", job_id="batch_b", status="failed",
                    model="gpt-4.1-nano",  request_count=15),
    ]
    handle = _make_handle("PMC_TEST", jobs)

    # Minimal runner stub — we only need advance() to reach the failure check.
    class _MiniRunner:
        handle_path = lambda self, _pmcid: tmp_path / "handle.json"  # noqa: E731
    mini = _MiniRunner.__new__(runner_mod.BatchKnowledgeExtractionRunner)
    mini.handle_path = lambda _pmcid: tmp_path / "handle.json"

    with pytest.raises(RuntimeError) as exc_info:
        runner_mod.BatchKnowledgeExtractionRunner.advance(mini, handle)
    assert "ALL" in str(exc_info.value)
    assert "PMC_TEST" in str(exc_info.value)
    # The error message must name the failed jobs so an operator can debug
    # without rummaging through provider dashboards.
    assert "gpt-4o-mini" in str(exc_info.value)
    assert "gpt-4.1-nano" in str(exc_info.value)


def test_partial_failure_emits_per_job_warning(monkeypatch, tmp_path, caplog):
    """When some-but-not-all jobs fail, each failure produces a WARNING
    line naming provider / model / request_count + one summary line."""
    _patch_providers(monkeypatch)

    jobs = [
        ProviderJob(provider="gemini", job_id="job_gem", status="completed",
                    model="gemini-2.5-flash-lite", request_count=15),
        ProviderJob(provider="openai", job_id="job_oai", status="failed",
                    model="gpt-4o-mini",  request_count=15),
    ]
    handle = _make_handle("PMC_PARTIAL", jobs)

    # Make `_process_level` a no-op so the WARNING is the only side effect
    # we observe — the deeper agreement / cascade machinery would need real
    # LLM clients otherwise. We also need stubs for the attributes that
    # advance()'s call-site reads when constructing `_process_level` args
    # (the call-site evaluates `self._l3` before the stubbed method runs).
    monkeypatch.setattr(
        runner_mod.BatchKnowledgeExtractionRunner, "_process_level",
        lambda self, *a, **kw: None,
    )
    handle.save = lambda _path: None

    mini = runner_mod.BatchKnowledgeExtractionRunner.__new__(runner_mod.BatchKnowledgeExtractionRunner)
    mini.handle_path = lambda _pmcid: tmp_path / "handle.json"
    mini._enable_router = True
    mini._l3 = None
    mini._l2 = []

    with caplog.at_level(logging.WARNING, logger="nlp_histo.pipeline.stages.knowledge_extraction.batch.runner"):
        runner_mod.BatchKnowledgeExtractionRunner.advance(mini, handle)

    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    # Per-job WARNING with full context.
    assert "Batch job FAILED" in messages
    assert "gpt-4o-mini" in messages
    assert "request_count=15" in messages
    assert "job_oai" in messages
    # Summary line.
    assert "1 of 2 batch jobs failed" in messages


def test_completed_jobs_only_emits_no_warnings(monkeypatch, tmp_path, caplog):
    """Sanity: a fully-successful set of jobs must NOT emit the failure
    warnings — guards against false positives that would normalize noise."""
    _patch_providers(monkeypatch)
    monkeypatch.setattr(
        runner_mod.BatchKnowledgeExtractionRunner, "_process_level",
        lambda self, *a, **kw: None,
    )

    jobs = [
        ProviderJob(provider="gemini", job_id="g1", status="completed",
                    model="gemini-2.5-flash-lite", request_count=15),
        ProviderJob(provider="openai", job_id="o1", status="completed",
                    model="gpt-4o-mini", request_count=15),
    ]
    handle = _make_handle("PMC_OK", jobs)
    handle.save = lambda _path: None

    mini = runner_mod.BatchKnowledgeExtractionRunner.__new__(runner_mod.BatchKnowledgeExtractionRunner)
    mini.handle_path = lambda _pmcid: tmp_path / "handle.json"
    mini._enable_router = True
    mini._l3 = None
    mini._l2 = []

    with caplog.at_level(logging.WARNING, logger="nlp_histo.pipeline.stages.knowledge_extraction.batch.runner"):
        runner_mod.BatchKnowledgeExtractionRunner.advance(mini, handle)

    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Batch job FAILED" not in messages
    assert "batch jobs failed" not in messages
