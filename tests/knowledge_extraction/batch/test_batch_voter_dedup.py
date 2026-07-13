"""Tests for in-run voter dedup in the batch knowledge extraction runner.

When an L2 (or L3) voter has the same (provider, model, temperature) as a
voter that already ran at an earlier level for the same chunk, the duplicate
API call is skipped and the earlier voter's content is reused via
``BatchHandle.synthetic_results``.

These tests poke ``_compute_voter_dedup`` directly with a minimally
initialised ``BatchKnowledgeExtractionRunner``; they don't hit any provider API.
"""
from __future__ import annotations

from pathlib import Path


from pipeline.stages.knowledge_extraction.batch.dispatch import build_requests
from pipeline.stages.knowledge_extraction.batch.models import (
    BatchHandle,
    BatchPhase,
    VoterBatchConfig,
)
from pipeline.stages.knowledge_extraction.batch.runner import BatchKnowledgeExtractionRunner
from pipeline.stages.knowledge_extraction.batch.voter_configs import (
    CLAUDE_HAIKU,
    CLAUDE_SONNET,
    GEMINI_L1,
    GEMINI_L2,
    OPENAI_L1,
    OPENAI_L2,
)


PMCID = "PMC1234567"


def _runner_for_dedup(l1, l2, l3) -> BatchKnowledgeExtractionRunner:
    """Build a runner shell with just the cascade attributes set.

    Skips __init__ on purpose — full construction requires LLM/embedding deps
    that are out of scope for these unit tests. _compute_voter_dedup and
    _voter_sig only read self._l1 / self._l2 / self._l3 plus handle data.
    """
    r = object.__new__(BatchKnowledgeExtractionRunner)
    r._l1 = l1
    r._l2 = l2
    r._l3 = l3
    return r


def _empty_handle(escalated: list[str]) -> BatchHandle:
    h = BatchHandle(pmcid=PMCID, phase=BatchPhase.L1_SUBMITTED)
    h.chunk_map = {c: [{"sentence": f"text for {c}", "text_element_id": i}]
                   for i, c in enumerate(escalated)}
    return h


# ── _voter_sig ──────────────────────────────────────────────────────────────

def test_voter_sig_treats_close_temps_as_equal():
    a = VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1)
    b = VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1 + 1e-9)
    assert BatchKnowledgeExtractionRunner._voter_sig(a) == BatchKnowledgeExtractionRunner._voter_sig(b)


def test_voter_sig_separates_different_models_or_providers_or_temps():
    base = VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1)
    diff_model = VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.1)
    diff_temp = VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.3)
    diff_provider = VoterBatchConfig(CLAUDE_HAIKU, provider="bedrock", temperature=0.1)
    sigs = {BatchKnowledgeExtractionRunner._voter_sig(v)
            for v in (base, diff_model, diff_temp, diff_provider)}
    assert len(sigs) == 4


# ── _compute_voter_dedup ────────────────────────────────────────────────────

def test_l2_haiku_at_same_temp_is_deduped_against_l1_haiku():
    """The `real` profile shape: Haiku at L1 (T=0.1) and L2 (T=0.1)."""
    l1 = [
        VoterBatchConfig(GEMINI_L1, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L1, provider="openai", temperature=0.1),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1),
    ]
    l2 = [
        VoterBatchConfig(GEMINI_L2, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.1),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1),
    ]
    l3 = VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.0)
    runner = _runner_for_dedup(l1, l2, l3)

    handle = _empty_handle(["C0", "C1"])
    # Pre-populate L1 raw content so the dedup helper can copy it.
    handle.l1_raw = {
        f"{PMCID}__C0__l1__2": '{"chunk_id": "C0", "findings": [], "summary_text": "x", '
                               '"audit_metadata": {"sentences_analyzed": 1, '
                               '"sentences_cited": [], "pmcids_referenced": [], '
                               '"uncited_sentences": []}}',
        f"{PMCID}__C1__l1__2": '{"chunk_id": "C1", "findings": [], "summary_text": "y", '
                               '"audit_metadata": {"sentences_analyzed": 1, '
                               '"sentences_cited": [], "pmcids_referenced": [], '
                               '"uncited_sentences": []}}',
    }

    skip, synthetic = runner._compute_voter_dedup(
        handle, escalated=["C0", "C1"], next_voters=l2, next_level="l2",
    )

    # Only voter index 2 (Haiku) matches; gemini and openai differ at L1 vs L2.
    assert skip == {"C0": {2}, "C1": {2}}
    # Two synthetic L2 results (one per chunk), pointing at L2 voter slot 2.
    assert sorted(s["custom_id"] for s in synthetic) == [
        f"{PMCID}__C0__l2__2",
        f"{PMCID}__C1__l2__2",
    ]
    # Token usage is zero so the L1 call isn't double-counted.
    assert all(s["input_tokens"] == 0 and s["output_tokens"] == 0 for s in synthetic)


def test_no_dedup_when_temperatures_differ():
    """Same model at L1 and L2 but different temps → no dedup."""
    l1 = [VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1)]
    l2 = [VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.3)]
    l3 = VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.0)
    runner = _runner_for_dedup(l1, l2, l3)

    handle = _empty_handle(["C0"])
    handle.l1_raw = {f"{PMCID}__C0__l1__0": '{"chunk_id": "C0"}'}

    skip, synthetic = runner._compute_voter_dedup(
        handle, escalated=["C0"], next_voters=l2, next_level="l2",
    )
    assert skip == {}
    assert synthetic == []


def test_no_dedup_when_l1_voter_failed_and_produced_no_content():
    """If the prior level's content is missing, we have nothing to reuse."""
    l1 = [VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1)]
    l2 = [VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1)]
    l3 = VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.0)
    runner = _runner_for_dedup(l1, l2, l3)

    handle = _empty_handle(["C0"])
    handle.l1_raw = {}  # L1 voter failed — nothing to copy.

    skip, synthetic = runner._compute_voter_dedup(
        handle, escalated=["C0"], next_voters=l2, next_level="l2",
    )
    assert skip == {}
    assert synthetic == []


def test_l3_dedup_checks_both_l1_and_l2():
    """L3 dedup must check L1 and L2 (L2 first-match overrides L1 only if added later)."""
    l1 = [VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.0)]
    l2 = [VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.0)]
    l3 = VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.0)  # matches L2
    runner = _runner_for_dedup(l1, l2, l3)

    handle = _empty_handle(["C0"])
    handle.l2_raw = {f"{PMCID}__C0__l2__0": '{"chunk_id": "C0"}'}
    handle.l1_raw = {}  # L1 doesn't match L3 anyway

    skip, synthetic = runner._compute_voter_dedup(
        handle, escalated=["C0"], next_voters=[l3], next_level="l3",
    )
    assert skip == {"C0": {0}}
    assert len(synthetic) == 1
    assert synthetic[0]["custom_id"] == f"{PMCID}__C0__l3__0"


# ── build_requests with skip_voters ─────────────────────────────────────────

def test_build_requests_skips_specified_voter_indices():
    voters = [
        VoterBatchConfig("a", provider="openai", temperature=0.1),
        VoterBatchConfig("b", provider="claude", temperature=0.1),
        VoterBatchConfig("c", provider="gemini", temperature=0.1),
    ]
    chunk_map = {"C0": [{"sentence": "s", "text_element_id": 1}]}
    reqs = build_requests(
        chunk_map, PMCID, voters, level="l2",
        chunk_ids=["C0"], skip_voters={"C0": {1}},  # skip 'b'
    )
    submitted_voter_indices = sorted(int(r.custom_id.split("__")[-1]) for r in reqs)
    assert submitted_voter_indices == [0, 2]


def test_build_requests_skip_all_returns_empty_list():
    voters = [VoterBatchConfig("a", provider="openai", temperature=0.1)]
    chunk_map = {"C0": [{"sentence": "s", "text_element_id": 1}]}
    reqs = build_requests(
        chunk_map, PMCID, voters, level="l2",
        chunk_ids=["C0"], skip_voters={"C0": {0}},
    )
    assert reqs == []


# ── BatchHandle serialisation round-trip with synthetic_results ─────────────

def test_handle_serialises_synthetic_results_field(tmp_path: Path):
    h = BatchHandle(pmcid=PMCID, phase=BatchPhase.L2_SUBMITTED)
    h.synthetic_results = {
        "l1": [],
        "l2": [{"custom_id": f"{PMCID}__C0__l2__2", "content": "{}", "input_tokens": 0,
                "output_tokens": 0, "error": None}],
        "l3": [],
    }
    p = tmp_path / "handle.json"
    h.save(p)
    h2 = BatchHandle.load(p)
    assert h2.synthetic_results["l2"] == h.synthetic_results["l2"]


# ── End-to-end advance() → synthetic merge → finalized ────────────────────

def test_advance_merges_synthetic_l2_into_finalized(tmp_path):
    """Full merge path: pre-populated synthetic L2 results survive advance()
    and end up in handle.finalized via _process_level → agreement.

    Sets handle.jobs=[] so advance() doesn't try to retrieve from any provider;
    the only "raw_results" come from synthetic_results["l2"].
    """
    import json as _json
    from pipeline.stages.knowledge_extraction.batch.runner import BatchKnowledgeExtractionRunner
    from pipeline.stages.knowledge_extraction.config import KnowledgeExtractionConfig
    from pipeline.stages.knowledge_extraction.models import (
        AuditableSummary, AuditMetadata, Finding,
        DirectionEnum, RelationTypeEnum, FindingScope,
    )

    cfg = KnowledgeExtractionConfig()
    runner = BatchKnowledgeExtractionRunner(
        l1_voters=[
            VoterBatchConfig(GEMINI_L1, provider="gemini", temperature=0.1),
            VoterBatchConfig(OPENAI_L1, provider="openai", temperature=0.1),
            VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1),
        ],
        l2_voters=[
            VoterBatchConfig(GEMINI_L2, provider="gemini", temperature=0.1),
            VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.1),
            VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1),
        ],
        l3_model=VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.0),
        escalation_llm=None,
        config=cfg,
        output_dir=tmp_path / "out",
        # Non-zero deterministic embedding: identical text → identical vector,
        # which gives cosine similarity 1.0 and AgreementChecker → KEEP.
        # An all-zero embed_fn would NaN the cosine and trigger ESCALATE.
        embed_fn=lambda texts: [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts],
        cascade_profile="real",
        artifact_root=None,
        run_modern_pipeline=False,
        # This test exercises the synthetic-results merge path. The router's
        # ProvenanceValidator would reject the test's synthetic verbatim
        # ("CD30 was associated with OS.") because it isn't a substring of
        # the chunk's lone sentence ("CD30 OS."), forcing escalation to L3
        # and an API call. Disable the router here — router behaviour is
        # covered by tests/knowledge_extraction/routing/.
        enable_router=False,
    )

    # Build a valid AuditableSummary JSON for the synthetic L2 content.
    finding = Finding(
        category="IHC",
        claim="CD30 predicts OS",
        evidence=[f"S1|{PMCID}|101"],
        confidence="high",
        verbatim_support="CD30 was associated with OS.",
        subject_entity="CD30",
        outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.negative,
        scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        grounding_score=0.8,
    )
    audit = AuditableSummary(
        chunk_id="C0",
        findings=[finding],
        summary_text="One finding about CD30.",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=[f"S1|{PMCID}|101"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )
    audit_json = _json.dumps(audit.model_dump())

    # Construct the handle as if L1 escalated C0 to L2 AND every L2 voter for
    # C0 was satisfied by L1 dedup (so jobs=[] and synthetic_results carries
    # all three L2 votes — collapsing the L2 agreement to "one cached opinion
    # per voter slot" for the single chunk).
    handle = BatchHandle(pmcid=PMCID, phase=BatchPhase.L2_SUBMITTED)
    handle.chunk_map = {"C0": [{"sentence_id": "S1", "pmcid": PMCID,
                                "text_element_id": 101,
                                "sentence": "CD30 OS."}]}
    handle.l2_chunk_ids = ["C0"]
    handle.l2_strip = [False, False, False]
    handle.jobs = []
    handle.synthetic_results = {
        "l1": [],
        "l2": [
            {"custom_id": f"{PMCID}__C0__l2__{i}", "content": audit_json,
             "input_tokens": 0, "output_tokens": 0, "error": None}
            for i in range(3)
        ],
        "l3": [],
    }

    handle_path = tmp_path / "handle.json"
    handle.save(handle_path)
    # The runner's handle_path() lookup needs to find our file.
    monkey_handle_path = (
        runner.handle_path(PMCID)
        if hasattr(runner, "handle_path")
        else handle_path
    )
    # Save under the expected location too.
    monkey_handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle.save(monkey_handle_path)

    advanced = runner.advance(handle)

    # The synthetic L2 votes parse, agree (identical content → KEEP), and
    # finalise into handle.finalized for chunk C0.
    assert "C0" in advanced.finalized, (
        f"Expected C0 in finalized, got {list(advanced.finalized)}; "
        f"phase={advanced.phase.value}"
    )
    assert advanced.finalized["C0"]["chunk_id"] == "C0"
    # Synthetic content should also have been recorded in l2_raw (the
    # debugging/audit store) so cost-postmortem tools see it.
    assert f"{PMCID}__C0__l2__0" in advanced.l2_raw
