"""Tests for `compute_pipeline_config_hash` (PERSISTENCE_TODOS #8).

Verifies the hash is:
- deterministic for identical inputs
- sensitive to every input dimension
- key-order-independent for dict inputs
- coerces Pydantic / dataclasses / enums via _to_jsonable

No LLM / API calls.
"""
from __future__ import annotations

import dataclasses
from enum import Enum

from nlp_histo.pipeline.stages.knowledge_extraction.persistence import compute_pipeline_config_hash


# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_kwargs() -> dict:
    return dict(
        config={"map": {"theta": 0.7}, "relate": {"entailment_threshold": 0.5}},
        thresholds={
            "grounding_threshold": 0.6,
            "entailment_threshold": 0.5,
            "contradiction_threshold": 0.55,
            "map_theta": 0.7,
            "map_reject_theta": 0.3,
        },
        models={
            "voter_models": ["deepseek", "gemini-flash-lite", "mistral-large"],
            "level2_voter_models": ["gemini-flash", "kimi", "haiku"],
            "escalation_model": "sonnet-4-6",
            "scorer": "embedding",
        },
        schema_version="map_v1_explicit_direction",
        prompt_version="map_prompt_v1_explicit_enums",
        cascade_signature="abc123def456",
    )


# ── Determinism ──────────────────────────────────────────────────────────────

def test_identical_inputs_yield_identical_hash():
    h1 = compute_pipeline_config_hash(**_base_kwargs())
    h2 = compute_pipeline_config_hash(**_base_kwargs())
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 16


def test_hash_is_dict_key_order_independent():
    a = _base_kwargs()
    b = _base_kwargs()
    # Re-pack thresholds in a different key order.
    b["thresholds"] = {k: a["thresholds"][k] for k in reversed(list(a["thresholds"]))}
    assert compute_pipeline_config_hash(**a) == compute_pipeline_config_hash(**b)


# ── Sensitivity ──────────────────────────────────────────────────────────────

def test_changing_threshold_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["thresholds"]["grounding_threshold"] = 0.65  # was 0.6
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_changing_model_changes_hash():
    base = _base_kwargs()
    swapped = _base_kwargs()
    swapped["models"]["escalation_model"] = "opus-4-7"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**swapped)


def test_changing_config_changes_hash():
    base = _base_kwargs()
    edited = _base_kwargs()
    edited["config"]["map"]["theta"] = 0.65  # was 0.7
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**edited)


def test_changing_schema_version_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["schema_version"] = "map_v2_new"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_changing_prompt_version_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["prompt_version"] = "map_prompt_v2"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_changing_cascade_signature_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["cascade_signature"] = "ffffffffffff"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_adding_threshold_key_changes_hash():
    base = _base_kwargs()
    extended = _base_kwargs()
    extended["thresholds"]["new_dial"] = 0.42
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**extended)


# ── Coercion: dataclasses + enums survive through _to_jsonable ───────────────

class _Mode(Enum):
    A = "a"
    B = "b"


@dataclasses.dataclass
class _MiniCfg:
    threshold: float
    mode: _Mode


def test_dataclass_and_enum_inputs_are_coerced_deterministically():
    cfg_a = _MiniCfg(threshold=0.5, mode=_Mode.A)
    cfg_a2 = _MiniCfg(threshold=0.5, mode=_Mode.A)
    cfg_b = _MiniCfg(threshold=0.5, mode=_Mode.B)
    h_a  = compute_pipeline_config_hash(config=cfg_a)
    h_a2 = compute_pipeline_config_hash(config=cfg_a2)
    h_b  = compute_pipeline_config_hash(config=cfg_b)
    assert h_a == h_a2
    assert h_a != h_b


# ── Empty / None inputs still hash deterministically ─────────────────────────

def test_all_none_inputs_hash_deterministically():
    h1 = compute_pipeline_config_hash()
    h2 = compute_pipeline_config_hash()
    assert h1 == h2
    # And differs from anything with real content.
    assert h1 != compute_pipeline_config_hash(schema_version="x")


# ── B-049: CANONICALIZE_DIRECTION_POLICY_VERSION participates in the hash ──────

def test_canonicalize_direction_policy_version_changes_hash():
    """The B-049 stamp must invalidate the cached summary hash when bumped.

    Both runners (`runner.py`, `batch/runner.py`) include the constant under
    the `thresholds["canonicalize_direction_policy_version"]` key. If a future
    refactor accidentally drops the key, this test catches it.
    """
    base = _base_kwargs()
    base["thresholds"]["canonicalize_direction_policy_version"] = (
        "per_direction_no_folding_v2"
    )
    bumped = _base_kwargs()
    bumped["thresholds"]["canonicalize_direction_policy_version"] = (
        "fold_majority_v1"  # imaginary old/new value to prove sensitivity
    )
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_canonicalize_direction_policy_constant_is_defined():
    """The constant must live at the documented import location."""
    from nlp_histo.pipeline.stages.knowledge_extraction.models import (
        CANONICALIZE_DIRECTION_POLICY_VERSION,
    )
    assert isinstance(CANONICALIZE_DIRECTION_POLICY_VERSION, str)
    assert CANONICALIZE_DIRECTION_POLICY_VERSION  # non-empty


def test_map_agreement_policy_version_changes_hash():
    """B-051 stamp must invalidate the cached summary hash when bumped.

    Both runners (`runner.py`, `batch/runner.py`) include the constant under
    the `thresholds["map_agreement_policy_version"]` key. This test pins the
    contract so a future refactor that removes the key gets caught.
    """
    base = _base_kwargs()
    base["thresholds"]["map_agreement_policy_version"] = "polarity_hard_fail_v1"
    bumped = _base_kwargs()
    bumped["thresholds"]["map_agreement_policy_version"] = (
        "polarity_hard_fail_v2_with_absent"  # imaginary future value
    )
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_map_agreement_policy_constant_is_defined():
    """The constant must live at the documented import location."""
    from nlp_histo.pipeline.stages.knowledge_extraction.models import MAP_AGREEMENT_POLICY_VERSION

    assert isinstance(MAP_AGREEMENT_POLICY_VERSION, str)
    assert MAP_AGREEMENT_POLICY_VERSION  # non-empty


def test_both_runners_include_map_agreement_policy_version_in_thresholds():
    """Regression guard: a runner that quietly drops the policy-version key
    re-opens the stale-cache hole that B-051 cache-invalidation closes.
    Greps the runner source for the literal key to enforce parity.
    """
    # Anchored to the installed module, not to a repository-relative path, so the
    # guard keeps pointing at the real source wherever the package lives.
    from pathlib import Path

    from nlp_histo.pipeline.stages.knowledge_extraction import batch, runner

    sync_src = Path(runner.__file__).read_text()
    batch_src = (Path(batch.__file__).parent / "runner.py").read_text()
    assert '"map_agreement_policy_version"' in sync_src, (
        "KnowledgeExtractionRunner._pipeline_config_hash must include "
        "'map_agreement_policy_version' in the thresholds dict (B-051)."
    )
    assert '"map_agreement_policy_version"' in batch_src, (
        "BatchKnowledgeExtractionRunner._pipeline_config_hash must include "
        "'map_agreement_policy_version' in the thresholds dict (B-051)."
    )
