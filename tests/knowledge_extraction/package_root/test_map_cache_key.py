"""Patch 2 — MAP cache invalidation regression tests.

Confirms that the MAP cache key changes when any of the following change:
  * NLI model id (`$NLP_HISTO_NLI_MODEL`)
  * Voter temperature
  * Voter top_p

Also confirms that on-disk entries written before the new fields existed are
treated as misses once the new metadata carries non-empty values.
"""
from __future__ import annotations

from nlp_histo.pipeline.stages.knowledge_extraction.cache import PipelineCache
from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    MAP_PROMPT_VERSION,
    MAP_SCHEMA_VERSION,
    MAP_STAGE_NAME,
    MapRunMetadata,
    compute_cascade_signature,
    compute_voter_config_hash,
)


def _meta(
    *,
    cascade_signature: str = "abc",
    voter_config_hash: str = "vh1",
    nli_model_id: str = "model-X",
) -> MapRunMetadata:
    return MapRunMetadata(
        schema_version=MAP_SCHEMA_VERSION,
        prompt_version=MAP_PROMPT_VERSION,
        stage_name=MAP_STAGE_NAME,
        provider="openai",
        model="gpt-4o-mini",
        cascade_profile="custom",
        cascade_signature=cascade_signature,
        voter_config_hash=voter_config_hash,
        nli_model_id=nli_model_id,
    )


def test_different_nli_model_id_changes_map_key():
    sentences = [{"text_element_id": 101}, {"text_element_id": 102}]
    k1 = PipelineCache._map_key(sentences, _meta(nli_model_id="model-X"))
    k2 = PipelineCache._map_key(sentences, _meta(nli_model_id="model-Y"))
    assert k1 != k2, "MAP cache key must change when NLI model id changes"


def test_different_voter_temperature_changes_voter_config_hash():
    spec_low  = [("openai", "gpt-4o-mini", 0.0, 1.0)]
    spec_high = [("openai", "gpt-4o-mini", 0.2, 1.0)]
    assert compute_voter_config_hash(spec_low) != compute_voter_config_hash(spec_high)


def test_different_voter_top_p_changes_voter_config_hash():
    spec_1 = [("openai", "gpt-4o-mini", 0.0, 1.0)]
    spec_p = [("openai", "gpt-4o-mini", 0.0, 0.9)]
    assert compute_voter_config_hash(spec_1) != compute_voter_config_hash(spec_p)


def test_top_p_none_distinguishable_from_one():
    """A provider that doesn't expose top_p (None) must hash differently from
    an explicit top_p=1.0 so the absence is part of the signature."""
    a = [("anthropic", "claude-haiku", 0.0, None)]
    b = [("anthropic", "claude-haiku", 0.0, 1.0)]
    assert compute_voter_config_hash(a) != compute_voter_config_hash(b)


def test_different_voter_config_hash_changes_map_key():
    sentences = [{"text_element_id": 101}, {"text_element_id": 102}]
    k1 = PipelineCache._map_key(sentences, _meta(voter_config_hash="vh1"))
    k2 = PipelineCache._map_key(sentences, _meta(voter_config_hash="vh2"))
    assert k1 != k2


def test_cascade_signature_unchanged_when_only_provider_model_match():
    """compute_cascade_signature still hashes identity only — temperature must
    not leak into it, so the voter_config_hash carries that signal instead."""
    s1 = compute_cascade_signature([("openai", "gpt-4o-mini")])
    s2 = compute_cascade_signature([("openai", "gpt-4o-mini")])
    assert s1 == s2


def test_old_format_cache_entry_is_miss_after_schema_bump(tmp_path):
    """An on-disk entry written without nli_model_id / voter_config_hash
    must miss when the caller's metadata now carries non-empty values."""
    cache_path = tmp_path / "cache.json"
    cache = PipelineCache(cache_path)
    sentences = [{"text_element_id": 1}]

    # Hand-write an old-format entry (no new fields) under what would have
    # been the old key. We use the new key construction but with empty
    # backwards-compat fields to mimic what a pre-patch run would have written.
    old_meta = _meta(voter_config_hash="", nli_model_id="")
    from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary, AuditMetadata
    fake_result = AuditableSummary(
        chunk_id="C0",
        findings=[],
        summary_text="",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=[],
            pmcids_referenced=[],
            uncited_sentences=[],
        ),
    )
    cache.set_map(sentences, fake_result, old_meta)

    # Now look up with the new-format metadata (non-empty fields).
    new_meta = _meta(voter_config_hash="vh1", nli_model_id="model-X")
    result = cache.get_map(sentences, new_meta)
    assert result is None, (
        "Old-format entry must miss when looked up with a metadata "
        "object whose voter_config_hash / nli_model_id differ from the "
        "stored values."
    )


def test_schema_version_in_current_release_is_new():
    """Pinned so a future MAP_SCHEMA_VERSION bump is an intentional act."""
    assert MAP_SCHEMA_VERSION == "map_v10_nli_and_voter_config_in_key"
