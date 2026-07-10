"""Tests for MapStage in-run voter cache.

Same (provider, model, temperature, chunk_text) must not call the LLM twice
within one paper-run. Concretely: when the real cascade profile has
claude-haiku at both L1 and L2 with the same temperature, the second call
must come from the cache rather than the API.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from pipeline.stages.knowledge_extraction.stages.map_stage import MapStage


@pytest.fixture
def stage_with_cache():
    """Construct a MapStage and prime its per-paper cache (process() not entered)."""
    stub_llm = MagicMock()
    stub_llm.temperature = 0.1
    with patch(
        "pipeline.stages.knowledge_extraction.stages.map_stage.build_map_chain"
    ) as mock_build:
        mock_build.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        stage = MapStage(
            voter_llms=[stub_llm],
            level2_voter_llms=[stub_llm],
            escalation_llm=stub_llm,
            voter_specs=[("claude", "claude-haiku-4-5-20251001")],
            level2_voter_specs=[("claude", "claude-haiku-4-5-20251001")],
            escalation_spec=("claude", "claude-sonnet-4-6-20251001"),
        )
    stage._voter_cache = {}
    stage._voter_cache_lock = threading.Lock()
    stage._voter_cache_hits = 0
    return stage


def test_cache_key_includes_provider_model_temp_and_text(stage_with_cache):
    inp = {"text": "hello world"}
    k1 = stage_with_cache._voter_cache_key("claude", "haiku", 0.1, inp)
    k2 = stage_with_cache._voter_cache_key("claude", "haiku", 0.1, inp)
    k3 = stage_with_cache._voter_cache_key("claude", "haiku", 0.3, inp)  # diff temp
    k4 = stage_with_cache._voter_cache_key("openai", "haiku", 0.1, inp)  # diff provider
    k5 = stage_with_cache._voter_cache_key("claude", "haiku", 0.1, {"text": "other"})  # diff text

    assert k1 == k2
    assert k1 != k3
    assert k1 != k4
    assert k1 != k5


def test_cache_key_rounds_float_temperature(stage_with_cache):
    inp = {"text": "x"}
    k1 = stage_with_cache._voter_cache_key("claude", "haiku", 0.1, inp)
    k2 = stage_with_cache._voter_cache_key("claude", "haiku", 0.1 + 1e-9, inp)
    # 1e-9 difference must round to the same bucket (3-decimal rounding).
    assert k1 == k2


def test_cache_get_set_roundtrip(stage_with_cache):
    inp = {"text": "hello"}
    key = stage_with_cache._voter_cache_key("claude", "haiku", 0.1, inp)
    assert stage_with_cache._voter_cache_get(key) is None
    sentinel = MagicMock(name="AuditableSummary")
    stage_with_cache._voter_cache_set(key, sentinel)
    assert stage_with_cache._voter_cache_get(key) is sentinel


def test_cache_get_returns_none_when_cache_uninitialised():
    """If process() never ran (no _voter_cache_lock), get must not crash."""
    stage = object.__new__(MapStage)
    key = ("claude", "haiku", 0.1, "deadbeef")
    assert stage._voter_cache_get(key) is None
    # set is a no-op rather than raising
    stage._voter_cache_set(key, MagicMock())


def test_run_voters_caches_l1_result_and_l2_haiku_reuses_it():
    """L1 and L2 each have one Haiku voter at the same temp.

    After L1 runs, L2's _run_voters call must hit the cache for that voter and
    NOT invoke its chain. Verifies the within-run dedup for the real profile.
    """
    stub_llm = MagicMock()
    stub_llm.temperature = 0.1

    l1_chain = MagicMock()
    l2_chain = MagicMock()
    l3_chain = MagicMock()
    l1_result = MagicMock(name="L1Result")
    # chunk_id round-trip check (2026-05-15) compares parsed.chunk_id against
    # the expected chunk_id. Configure the mock so the equality check matches.
    l1_result.chunk_id = "C0"
    l1_chain.invoke.return_value = l1_result

    with patch(
        "pipeline.stages.knowledge_extraction.stages.map_stage.build_map_chain"
    ) as mock_build:
        mock_build.side_effect = [l1_chain, l2_chain, l3_chain]
        stage = MapStage(
            voter_llms=[stub_llm],
            level2_voter_llms=[stub_llm],
            escalation_llm=stub_llm,
            voter_specs=[("claude", "claude-haiku-4-5-20251001")],
            level2_voter_specs=[("claude", "claude-haiku-4-5-20251001")],
            escalation_spec=("claude", "claude-sonnet-4-6-20251001"),
        )

    # Simulate the bookkeeping that process() does, so _run_voters can use the cache.
    stage._voter_cache = {}
    stage._voter_cache_lock = threading.Lock()
    stage._voter_cache_hits = 0
    stage._usage_collector = None

    inp = {"text": "the same chunk text", "chunk_id": "C0", "pmcid": "PMC1"}

    # First call: L1. Chain is invoked, result cached.
    out_l1, _t1 = stage._run_voters(inp, chains=stage._voter_chains, level="L1")
    assert out_l1 == [l1_result]
    assert l1_chain.invoke.call_count == 1

    # Second call: L2 with same (provider, model, temp). Must NOT invoke L2 chain.
    out_l2, _t2 = stage._run_voters(inp, chains=stage._level2_voter_chains, level="L2")
    assert out_l2 == [l1_result], "L2 must reuse the cached L1 Haiku result"
    assert l2_chain.invoke.call_count == 0, (
        "L2 Haiku voter must hit the cache instead of calling the API"
    )
    assert stage._voter_cache_hits == 1


def test_run_voters_does_not_dedup_different_temperatures():
    """Same model, different temp at L1 and L2 → both must invoke (no cache hit)."""
    l1_llm = MagicMock()
    l1_llm.temperature = 0.1
    l2_llm = MagicMock()
    l2_llm.temperature = 0.5  # deliberately different

    l1_chain = MagicMock()
    l2_chain = MagicMock()
    l3_chain = MagicMock()
    l1_result = MagicMock(name="L1Result")
    l2_result = MagicMock(name="L2Result")
    # chunk_id round-trip check expects parsed.chunk_id to equal the chunk_id
    # passed into _run_voters (test uses "C0").
    l1_result.chunk_id = "C0"
    l2_result.chunk_id = "C0"
    l1_chain.invoke.return_value = l1_result
    l2_chain.invoke.return_value = l2_result

    with patch(
        "pipeline.stages.knowledge_extraction.stages.map_stage.build_map_chain"
    ) as mock_build:
        mock_build.side_effect = [l1_chain, l2_chain, l3_chain]
        stage = MapStage(
            voter_llms=[l1_llm],
            level2_voter_llms=[l2_llm],
            escalation_llm=l2_llm,
            voter_specs=[("claude", "claude-haiku-4-5-20251001")],
            level2_voter_specs=[("claude", "claude-haiku-4-5-20251001")],
            escalation_spec=("claude", "claude-sonnet-4-6-20251001"),
        )

    stage._voter_cache = {}
    stage._voter_cache_lock = threading.Lock()
    stage._voter_cache_hits = 0
    stage._usage_collector = None

    inp = {"text": "same text", "chunk_id": "C0", "pmcid": "PMC1"}

    stage._run_voters(inp, chains=stage._voter_chains, level="L1")
    stage._run_voters(inp, chains=stage._level2_voter_chains, level="L2")

    assert l1_chain.invoke.call_count == 1
    assert l2_chain.invoke.call_count == 1, (
        "different temperatures must not share the voter-cache entry"
    )
    assert stage._voter_cache_hits == 0
