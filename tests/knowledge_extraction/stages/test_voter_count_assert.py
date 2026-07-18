"""Regression test for Issue B (this session): cascade decisions must warn
when voter_count is below the profile's declared voter list.

A shortfall almost always indicates an upstream silent failure:
* Router classified one or more voters as UNUSABLE (B-019 class).
* Batch submission rejected one or more voters (B-020 class).
* Live LLM call exceeded the 2-attempt retry budget.

Either way, a single WARNING per occurrence is the cheapest tripwire we
have. This test patches the cascade-decision path with stubbed voter lists
and asserts the WARNING fires under the right conditions.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage import MapStage


def _make_stub_stage(l1_voter_count: int, l2_voter_count: int) -> MapStage:
    """Build a bare MapStage instance without invoking __init__ — we only
    need the attributes the voter-count assertion reads."""
    stage = MapStage.__new__(MapStage)
    stage._voter_specs = [("gemini", "g")] * l1_voter_count
    stage._l2_specs = [("openai", "o")] * l2_voter_count
    stage._cascade_decision_log = None  # disable JSONL write — the assertion
                                        # runs before the log path is reached
    stage._cascade_run_id = "test"
    stage._cascade_signature = "sig"
    stage._cascade_profile = "test"
    return stage


def _fake_outcome():
    """Minimal stand-in for ChunkOutcome — _record_cascade_decision only
    reads attributes via make_decision_record, which we bypass since the
    log is None."""
    return SimpleNamespace(best=None, keep=True)


def test_l1_shortfall_emits_warning(caplog):
    """L1 with 3 declared voters but only 1 ran → WARNING."""
    stage = _make_stub_stage(l1_voter_count=3, l2_voter_count=3)
    with caplog.at_level(logging.WARNING, logger="nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage"):
        stage._record_cascade_decision(
            _fake_outcome(),
            pmcid="PMC_X", chunk_id="C1", level="l1",
            voter_count=1, voter_specs=None,
        )
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "voter_count=1" in messages
    assert "expected=3" in messages
    assert "L1" in messages


def test_l2_shortfall_emits_warning(caplog):
    """L2 with 3 declared voters but only 2 ran → WARNING."""
    stage = _make_stub_stage(l1_voter_count=3, l2_voter_count=3)
    with caplog.at_level(logging.WARNING, logger="nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage"):
        stage._record_cascade_decision(
            _fake_outcome(),
            pmcid="PMC_X", chunk_id="C5", level="l2",
            voter_count=2, voter_specs=None,
        )
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "voter_count=2" in messages
    assert "expected=3" in messages
    assert "L2" in messages


def test_full_voter_count_emits_no_warning(caplog):
    """Sanity: voter_count == expected → NO warning. Guards against the
    assertion firing as a constant noise source."""
    stage = _make_stub_stage(l1_voter_count=3, l2_voter_count=3)
    with caplog.at_level(logging.WARNING, logger="nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage"):
        stage._record_cascade_decision(
            _fake_outcome(),
            pmcid="PMC_X", chunk_id="C1", level="l1",
            voter_count=3, voter_specs=None,
        )
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "voter_count" not in messages.lower() or "expected" not in messages


def test_l3_expected_is_one(caplog):
    """L3 is single-voter by design — voter_count=1 must not warn,
    voter_count=0 must warn."""
    stage = _make_stub_stage(l1_voter_count=3, l2_voter_count=3)
    # Happy path
    with caplog.at_level(logging.WARNING, logger="nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage"):
        stage._record_cascade_decision(
            _fake_outcome(),
            pmcid="PMC_X", chunk_id="C1", level="l3",
            voter_count=1, voter_specs=None,
        )
    happy_msgs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "voter_count" not in happy_msgs.lower() or "expected" not in happy_msgs

    caplog.clear()
    # Shortfall path
    with caplog.at_level(logging.WARNING, logger="nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage"):
        stage._record_cascade_decision(
            _fake_outcome(),
            pmcid="PMC_X", chunk_id="C1", level="l3",
            voter_count=0, voter_specs=None,
        )
    short_msgs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "voter_count=0" in short_msgs
    assert "expected=1" in short_msgs
