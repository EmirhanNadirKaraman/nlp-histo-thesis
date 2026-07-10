"""Tests for invocation-time MAP usage capture (replaces the broken
LangChain callback path for cost reporting).

The point being tested: when ``MapStage`` invokes a chain built with
``with_structured_output(..., include_raw=True)``, the returned dict's
``raw`` message carries ``usage_metadata``, and ``MapStage`` records a
non-zero ``InvocationUsage`` for that call — even though
``with_structured_output`` strips LangChain callbacks (the historical
reason the sync cost report came out as $0.00000).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: extract_usage_from_message
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_usage_metadata_standard_shape():
    """LangChain ≥0.3 standardised AIMessage.usage_metadata field."""
    from pipeline.stages.summarization.costing.invocation_usage import (
        extract_usage_from_message,
    )
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290},
        response_metadata={},
    )
    inp, out, src = extract_usage_from_message(msg)
    assert inp == 1234
    assert out == 56
    assert src == "usage_metadata"


def test_extract_response_metadata_token_usage_fallback():
    """Older OpenAI shape: response_metadata['token_usage']['prompt_tokens']."""
    from pipeline.stages.summarization.costing.invocation_usage import (
        extract_usage_from_message,
    )
    msg = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    )
    inp, out, src = extract_usage_from_message(msg)
    assert (inp, out, src) == (10, 3, "response_metadata")


def test_extract_returns_missing_when_no_usage():
    from pipeline.stages.summarization.costing.invocation_usage import (
        extract_usage_from_message,
    )
    msg = SimpleNamespace(usage_metadata=None, response_metadata={})
    inp, out, src = extract_usage_from_message(msg)
    assert (inp, out, src) == (0, 0, "missing")


def test_unwrap_handles_dict_and_bare_schema():
    """unwrap_structured_output supports both shapes — chain may or may not
    have been built with include_raw=True."""
    from pipeline.stages.summarization.costing.invocation_usage import (
        unwrap_structured_output,
    )
    raw_msg = SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 2})
    schema_obj = SimpleNamespace(chunk_id="chunk-0")
    # include_raw=True shape
    parsed, raw, err = unwrap_structured_output(
        {"raw": raw_msg, "parsed": schema_obj, "parsing_error": None},
    )
    assert parsed is schema_obj
    assert raw is raw_msg
    assert err is None
    # legacy shape (no include_raw): bare schema, no raw
    parsed, raw, err = unwrap_structured_output(schema_obj)
    assert parsed is schema_obj
    assert raw is None
    assert err is None


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test: MapStage._invoke_l3 records non-zero usage from a fake chain
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_auditable_summary():
    from pipeline.stages.summarization.models import AuditableSummary, AuditMetadata
    return AuditableSummary(
        chunk_id="chunk-0",
        findings=[],
        summary_text="",
        audit_metadata=AuditMetadata(
            sentences_analyzed=0,
            sentences_cited=[],
            pmcids_referenced=[],
            uncited_sentences=[],
        ),
    )


def _make_fake_chain(raw_usage_metadata: dict, parsed_obj):
    """Build a Runnable-shaped object whose .invoke returns include_raw dict."""
    raw_msg = SimpleNamespace(
        usage_metadata=raw_usage_metadata,
        response_metadata={},
    )
    class _FakeChain:
        def invoke(self, _inp, config=None):
            return {"raw": raw_msg, "parsed": parsed_obj, "parsing_error": None}
    return _FakeChain()


def test_invoke_l3_records_usage_from_structured_output_raw(fake_auditable_summary):
    """When build_map_chain returns include_raw dicts, MapStage records a
    non-zero InvocationUsage from the raw AIMessage even though no LangChain
    callbacks fire (with_structured_output strips them)."""
    from pipeline.stages.summarization.stages import map_stage

    fake_chain = _make_fake_chain(
        {"input_tokens": 4242, "output_tokens": 99}, fake_auditable_summary,
    )

    # Patch build_map_chain so MapStage's __init__ doesn't need real LLMs.
    with patch.object(map_stage, "build_map_chain", return_value=fake_chain):
        stage = map_stage.MapStage(
            voter_llms=[object()],
            level2_voter_llms=[object()],
            escalation_llm=object(),
            voter_specs=[("anthropic", "claude-haiku-4-5-20251001")],
            level2_voter_specs=[("anthropic", "claude-haiku-4-5-20251001")],
            escalation_spec=("anthropic", "claude-sonnet-4-6"),
        )

    # _invoke_l3 needs the in-run voter cache initialised (set up in process()).
    import threading
    stage._voter_cache = {}
    stage._voter_cache_lock = threading.Lock()
    stage._voter_cache_hits = 0
    stage._voter_cache_misses = 0

    parsed = stage._invoke_l3({"text": "hello", "chunk_id": "chunk-0"}, "chunk-0")

    # The chain's parsed schema flows through unchanged.
    assert parsed is fake_auditable_summary
    # And a non-zero usage record was appended.
    records = stage.invocation_usage_records()
    assert len(records) == 1
    r = records[0]
    assert r.level == "L3"
    assert r.provider == "anthropic"
    assert r.model == "claude-sonnet-4-6"
    assert r.input_tokens == 4242
    assert r.output_tokens == 99
    assert r.usage_source == "usage_metadata"


def test_invoke_l3_records_missing_when_raw_absent(fake_auditable_summary):
    """Back-compat path: chain returns bare schema (no include_raw). The
    record is still emitted, tagged usage_source='missing' so reports can
    surface the gap rather than silently returning $0."""
    from pipeline.stages.summarization.stages import map_stage

    class _BareChain:
        def invoke(self, _inp, config=None):
            return fake_auditable_summary

    with patch.object(map_stage, "build_map_chain", return_value=_BareChain()):
        stage = map_stage.MapStage(
            voter_llms=[object()],
            level2_voter_llms=[object()],
            escalation_llm=object(),
            voter_specs=[("anthropic", "claude-haiku-4-5-20251001")],
            level2_voter_specs=[("anthropic", "claude-haiku-4-5-20251001")],
            escalation_spec=("anthropic", "claude-sonnet-4-6"),
        )

    import threading
    stage._voter_cache = {}
    stage._voter_cache_lock = threading.Lock()
    stage._voter_cache_hits = 0
    stage._voter_cache_misses = 0

    parsed = stage._invoke_l3({"text": "hello", "chunk_id": "chunk-0"}, "chunk-0")
    assert parsed is fake_auditable_summary

    records = stage.invocation_usage_records()
    assert len(records) == 1
    assert records[0].usage_source == "missing"
    assert records[0].input_tokens == 0
    assert records[0].output_tokens == 0
