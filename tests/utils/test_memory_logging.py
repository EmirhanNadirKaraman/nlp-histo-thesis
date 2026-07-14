"""Smoke tests for pipeline.utils.memory_logging."""
from __future__ import annotations

import logging
import re

import pytest

from pipeline.utils import memory_logging
from pipeline.utils.memory_logging import MemoryLogger


_MEMORY_PREFIX = "MEMORY "


def _records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith(_MEMORY_PREFIX) and "stage=" in r.getMessage()
    ]


def test_checkpoint_emits_memory_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pipeline.memory")
    mem = MemoryLogger()
    mem.checkpoint("NORMALIZE", "before")

    lines = _records(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert "stage=NORMALIZE" in line
    assert "event=before" in line
    assert "elapsed_s=" in line


def test_checkpoint_pair_reports_delta(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pipeline.memory")
    mem = MemoryLogger()
    mem.checkpoint("MAP", "before")
    mem.checkpoint("MAP", "after")

    lines = _records(caplog)
    assert len(lines) == 2
    # First call has no previous checkpoint so no delta_rss_mb is emitted.
    assert "delta_rss_mb" not in lines[0]
    # Second call should include a delta when psutil is available.
    if memory_logging._PSUTIL_AVAILABLE:
        assert "delta_rss_mb=" in lines[1]


def test_stage_context_manager_records_before_and_after(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pipeline.memory")
    mem = MemoryLogger()
    with mem.stage("GROUP"):
        pass

    lines = _records(caplog)
    events = [re.search(r"event=(\S+)", line).group(1) for line in lines]  # type: ignore[union-attr]
    assert events == ["before", "after"]
    assert all("stage=GROUP" in line for line in lines)


def test_stage_context_manager_records_failure_and_reraises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline.memory")
    mem = MemoryLogger()
    with pytest.raises(RuntimeError):
        with mem.stage("RELATE"):
            raise RuntimeError("boom")

    lines = _records(caplog)
    events = [re.search(r"event=(\S+)", line).group(1) for line in lines]  # type: ignore[union-attr]
    assert events == ["before", "failed"]
    assert mem.last_stage == "RELATE"


def test_pmcid_included_when_provided(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pipeline.memory")
    mem = MemoryLogger(pmcid="PMC1448691")
    mem.checkpoint("MAP", "before")

    lines = _records(caplog)
    assert "pmcid=PMC1448691" in lines[0]


def test_psutil_missing_is_graceful(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When psutil is unavailable, the logger still emits MEMORY lines (with
    no rss/vms fields) and warns at most once."""
    monkeypatch.setattr(memory_logging, "_PSUTIL_AVAILABLE", False)
    monkeypatch.setattr(memory_logging, "_PSUTIL_WARNED", False)

    caplog.set_level(logging.INFO, logger="pipeline.memory")
    mem = MemoryLogger()
    mem.checkpoint("MAP", "before")
    mem.checkpoint("MAP", "after")

    lines = _records(caplog)
    assert len(lines) == 2
    for line in lines:
        assert "rss_mb=" not in line
        assert "vms_mb=" not in line
        assert "delta_rss_mb" not in line

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
