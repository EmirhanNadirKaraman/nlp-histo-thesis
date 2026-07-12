"""Regression test for B-065 — `_process_level` must not crash on None voter outputs.

The bug:
  ``BatchKnowledgeExtractionRunner._process_level`` Pass 2 (pre-embedding) iterated
  ``chunk_voters`` without filtering ``None`` slots, so ``_claims(None)``
  raised ``AttributeError: 'NoneType' object has no attribute 'findings'``.
  Latent for the ``real`` profile (3 L1 voters; rarely all three fail on the
  same chunk) but certain for ``haiku_only`` (N=1 → any parse failure → the
  voters_full list is ``[None]``).

This test exercises the haiku_only-style cascade in batch mode with one chunk
whose single L1 voter result is structurally an error (empty content → parse
returns None). Pre-fix behaviour: ``advance()`` crashes mid-Pass-2. Post-fix:
the chunk routes to escalation via Pass 3's existing all-None branch, and
``advance()`` completes without raising.

See ``docs/BUGS.md`` §"Bug 65".
"""
from __future__ import annotations

import json as _json
from pathlib import Path

import pytest

from pipeline.stages.knowledge_extraction.batch.models import (
    BatchHandle,
    BatchPhase,
    VoterBatchConfig,
)
from pipeline.stages.knowledge_extraction.batch.runner import BatchKnowledgeExtractionRunner
from pipeline.stages.knowledge_extraction.batch.voter_configs import CLAUDE_HAIKU
from pipeline.stages.knowledge_extraction.config import KnowledgeExtractionConfig


PMCID = "PMC9999991"


def _haiku_only_runner(tmp_path: Path) -> BatchKnowledgeExtractionRunner:
    """Build a minimally-constructed haiku_only cascade for the regression test.

    Real construction skips API clients (we mock submit_level instead so no
    provider calls fire). The embed_fn returns a stable non-zero vector so
    the agreement path (Pass 3) would resolve cleanly *if* a voter did
    parse — this test exercises the None branch, not the keep branch.
    """
    haiku = VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1)
    cfg = KnowledgeExtractionConfig()
    cfg.contradiction_similarity_threshold = None  # type: ignore[misc]
    return BatchKnowledgeExtractionRunner(
        l1_voters=[haiku],
        l2_voters=[haiku],
        l3_model=haiku,
        escalation_llm=None,
        config=cfg,
        output_dir=tmp_path / "out",
        embed_fn=lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts],
        cascade_profile="haiku_only",
        artifact_root=None,
        run_modern_pipeline=False,
        enable_router=False,
    )


def test_advance_does_not_crash_when_lone_l1_voter_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """B-065: a None voter at L1 in the N=1 haiku_only cascade must not crash
    Pass 2's embed-collection. The chunk should fall through to Pass 3's
    all-None branch and be marked for escalation to L2 instead.
    """
    runner = _haiku_only_runner(tmp_path)

    # Mock submit_level so the post-escalation L2 submission is a no-op —
    # the regression we care about is purely _process_level's None handling.
    from pipeline.stages.knowledge_extraction.batch import runner as _runner_mod
    monkeypatch.setattr(_runner_mod, "submit_level", lambda *a, **kw: [])

    handle = BatchHandle(pmcid=PMCID, phase=BatchPhase.L1_SUBMITTED)
    handle.chunk_map = {
        "C0": [{"sentence_id": "S1", "pmcid": PMCID,
                "text_element_id": 101,
                "sentence": "CD30 OS."}],
    }
    handle.l1_strip = [False]
    handle.jobs = []                       # no provider results to retrieve
    handle.synthetic_results = {
        # The single L1 voter "failed": empty content → parse_result returns
        # None → voters_full = [None]. Pre-fix this killed _process_level.
        "l1": [
            {"custom_id": f"{PMCID}__C0__l1__0", "content": "",
             "input_tokens": 0, "output_tokens": 0, "error": "empty"},
        ],
        "l2": [],
        "l3": [],
    }

    handle_path = runner.handle_path(PMCID)
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle.save(handle_path)

    # The actual regression assertion: no AttributeError.
    advanced = runner.advance(handle)

    # Chunk did NOT finalise (no valid voter output to keep) and was routed
    # to L2 escalation by Pass 3's `if not voters: escalated.append(...)` path.
    assert "C0" not in advanced.finalized, (
        f"Expected C0 to escalate (no valid L1 voter to KEEP), but it was "
        f"finalised: {advanced.finalized}"
    )
    assert advanced.l2_chunk_ids == ["C0"], (
        f"Expected C0 in l2_chunk_ids after L1-None escalation, got "
        f"{advanced.l2_chunk_ids}"
    )
    assert advanced.phase == BatchPhase.L2_SUBMITTED, (
        f"Expected phase L2_SUBMITTED after escalation, got {advanced.phase}"
    )


def test_advance_handles_mixed_none_and_valid_voters_in_real_profile_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """B-065 was latent in `real` too — partial-None voter sets must also be
    safe. This builds a 3-voter L1 where one slot fails and the other two
    return valid AuditableSummary JSON; Pass 2 must collect claims only from
    the surviving voters and Pass 3 must run agreement on N=2 (not N=3).
    """
    from pipeline.stages.knowledge_extraction.batch.voter_configs import (
        GEMINI_L1, OPENAI_L1,
    )
    from pipeline.stages.knowledge_extraction.models import (
        AuditableSummary, AuditMetadata, Finding,
        DirectionEnum, RelationTypeEnum, FindingScope,
    )

    cfg = KnowledgeExtractionConfig()
    cfg.contradiction_similarity_threshold = None  # type: ignore[misc]
    runner = BatchKnowledgeExtractionRunner(
        l1_voters=[
            VoterBatchConfig(GEMINI_L1, provider="gemini", temperature=0.1),
            VoterBatchConfig(OPENAI_L1, provider="openai", temperature=0.1),
            VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1),
        ],
        l2_voters=[
            VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1),
        ],
        l3_model=VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1),
        escalation_llm=None,
        config=cfg,
        output_dir=tmp_path / "out",
        embed_fn=lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts],
        cascade_profile="real",
        artifact_root=None,
        run_modern_pipeline=False,
        enable_router=False,
    )

    from pipeline.stages.knowledge_extraction.batch import runner as _runner_mod
    monkeypatch.setattr(_runner_mod, "submit_level", lambda *a, **kw: [])

    # Two valid voters + one failed voter for the same chunk.
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

    handle = BatchHandle(pmcid=PMCID, phase=BatchPhase.L1_SUBMITTED)
    handle.chunk_map = {
        "C0": [{"sentence_id": "S1", "pmcid": PMCID,
                "text_element_id": 101,
                "sentence": "CD30 OS."}],
    }
    handle.l1_strip = [False, False, False]
    handle.jobs = []
    handle.synthetic_results = {
        "l1": [
            # voter[0] failed (empty content)
            {"custom_id": f"{PMCID}__C0__l1__0", "content": "",
             "input_tokens": 0, "output_tokens": 0, "error": "empty"},
            # voter[1] and voter[2] succeeded with identical valid output
            {"custom_id": f"{PMCID}__C0__l1__1", "content": audit_json,
             "input_tokens": 0, "output_tokens": 0, "error": None},
            {"custom_id": f"{PMCID}__C0__l1__2", "content": audit_json,
             "input_tokens": 0, "output_tokens": 0, "error": None},
        ],
        "l2": [],
        "l3": [],
    }

    handle_path = runner.handle_path(PMCID)
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle.save(handle_path)

    advanced = runner.advance(handle)

    # The two surviving voters agree (identical content → KEEP at L1), so C0
    # finalises with the survivors' findings — even though voter[0] was None.
    # Pre-fix, Pass 2 would crash on the None slot before reaching this.
    assert "C0" in advanced.finalized, (
        f"Expected C0 to finalise from the 2 surviving voters, got "
        f"finalized={list(advanced.finalized)}, phase={advanced.phase}"
    )
    assert advanced.finalized["C0"]["chunk_id"] == "C0"
