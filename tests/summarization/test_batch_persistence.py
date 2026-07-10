"""Tests for filesystem persistence in the batch runner.

Bypasses the actual provider batch APIs by synthesising a completed
``BatchHandle`` and calling ``finalize()`` directly. No LLM/API calls.

To watch progress live::

    python -m pytest tests/summarization/test_batch_persistence.py -x -s \
        --log-cli-level=INFO
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Disable scispaCy/UMLS load before any pipeline import — these tests synthesise
# a completed BatchHandle and never need real CUI lookups; loading the linker
# would add 5-30s per test process for nothing. The env vars are read at call
# time, so setting them here is sufficient.
os.environ.setdefault("NLP_HISTO_DISABLE_UMLS", "1")
os.environ.setdefault("NLP_HISTO_SKIP_UMLS_ENRICHMENT", "1")


# Module-level logging to surface progress even without ``-s``. INFO lines hit
# the captured-stdout buffer for failed tests; live ``--log-cli-level=INFO``
# streams them.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)


def _trace(msg: str) -> None:
    print(f"[test] {msg}", flush=True, file=sys.stderr)

from pipeline.stages.summarization.batch.models import (
    BatchHandle,
    BatchPhase,
    VoterBatchConfig,
)
from pipeline.stages.summarization.batch.runner import BatchSummarizationRunner
from pipeline.stages.summarization.config import SummarizationConfig
from pipeline.stages.summarization.models import (
    AuditableSummary,
    AuditMetadata,
    DirectionEnum,
    Finding,
    FindingScope,
    MAP_PROMPT_VERSION,
    MAP_SCHEMA_VERSION,
    RelationTypeEnum,
)


PMCID = "PMC9000099"

# Fake voter identifier — these tests bypass the provider batch APIs entirely
# (synthesised completed BatchHandle → finalize()), so the model string is just
# a label for VoterBatchConfig and never reaches an LLM.
FAKE_MODEL = "fake-model-batch-persistence-test"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _finding() -> Finding:
    return Finding(
        category="IHC",
        claim="CD30 predicts worse OS",
        evidence=[f"S1|{PMCID}|101"],
        confidence="high",
        verbatim_support="CD30 was associated with worse OS in DLBCL.",
        subject_entity="CD30",
        outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.negative,
        scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        grounding_score=0.8,
    )


def _audit_summary() -> AuditableSummary:
    return AuditableSummary(
        chunk_id="C1",
        findings=[_finding()],
        summary_text="One finding about CD30.",
        audit_metadata=AuditMetadata(
            sentences_analyzed=5,
            sentences_cited=[f"S1|{PMCID}|101"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )


def _voter(temperature: float = 0.0) -> VoterBatchConfig:
    return VoterBatchConfig(FAKE_MODEL, provider="claude", temperature=temperature)


def _build_runner(tmp_path: Path) -> BatchSummarizationRunner:
    """Construct a batch runner without any structured-output-capable LLM.

    Works because run_reduce=False keeps Reduce/Rule stages unbuilt and
    ContradictionDetector unbuilt — no real LLM is touched.
    """
    cfg = SummarizationConfig()
    # Disable contradiction detector defensively (also gated on run_reduce now).
    cfg.contradiction_similarity_threshold = None  # type: ignore[misc]

    fake_embed = lambda texts: [[0.0] * 8 for _ in texts]
    return BatchSummarizationRunner(
        l1_voters=[_voter(0.0), _voter(0.3)],
        l2_voters=[_voter(0.2)],
        l3_model=_voter(0.0),
        escalation_llm=None,                # never used when run_reduce=False
        config=cfg,
        output_dir=tmp_path / "out",
        embed_fn=fake_embed,
        cascade_profile="smoke_fake",
        artifact_root=tmp_path / "runs",
        artifact_run_id="batch_test_run",
        run_modern_pipeline=True,
        run_reduce=False,
    )


def _completed_handle(audit: AuditableSummary) -> BatchHandle:
    """Return a BatchHandle in COMPLETE state with one chunk's MAP output."""
    return BatchHandle(
        pmcid=PMCID,
        phase=BatchPhase.COMPLETE,
        sentences=[{"sentence_id": "S1", "pmcid": PMCID, "text_element_id": 101,
                    "sentence": "CD30 was associated with worse OS."}],
        chunk_map={"C1": [{"sentence_id": "S1", "pmcid": PMCID,
                            "text_element_id": 101,
                            "sentence": "CD30 was associated with worse OS."}]},
        finalized={audit.chunk_id: audit.model_dump()},
        cascade_profile="smoke_fake",
        cascade_signature="testsig",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_batch_finalize_writes_full_artifact_layout(tmp_path: Path):
    _trace("building runner...")
    t0 = time.perf_counter()
    runner = _build_runner(tmp_path)
    _trace(f"runner built in {time.perf_counter() - t0:.1f}s")

    handle = _completed_handle(_audit_summary())
    _trace("calling finalize() — modern chain runs NORMALIZE→...→RESOLVE "
           "(scispaCy/UMLS disabled at module top for speed)")
    t0 = time.perf_counter()
    result = runner.finalize(handle)
    _trace(f"finalize() returned in {time.perf_counter() - t0:.1f}s")

    assert result["status"] == "success"
    assert result["pmcid"] == PMCID
    assert result["run_id"] == "batch_test_run"

    run_dir = tmp_path / "runs" / "batch_test_run"
    expected = [
        run_dir / "manifest.json",
        run_dir / "map" / PMCID / "findings.jsonl",
        run_dir / "map" / PMCID / "chunks.jsonl",
        run_dir / "map" / PMCID / "rejected_findings.jsonl",
        run_dir / "normalize" / PMCID / "normal_findings.jsonl",
        run_dir / "normalize" / PMCID / "entity_links.jsonl",
        run_dir / "normalize" / PMCID / "dedup_trace.jsonl",
        run_dir / "group" / PMCID / "groups.jsonl",
        run_dir / "group" / PMCID / "non_groupable.jsonl",
        run_dir / "canonicalize" / PMCID / "canonical_rules.jsonl",
        run_dir / "relate" / PMCID / "relations.jsonl",
        run_dir / "relate" / PMCID / "raw_pairs.jsonl",
        run_dir / "relate" / PMCID / "skipped_pairs.jsonl",
        run_dir / "resolve" / PMCID / "final_rules.jsonl",
        run_dir / "resolve" / PMCID / "score_trace.jsonl",
    ]
    for p in expected:
        assert p.exists(), f"missing artifact: {p}"

    # Manifest reflects completion + provenance
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert PMCID in payload["papers"]
    assert set(payload["stages_completed"]) == {
        "map", "normalize", "group", "canonicalize", "relate", "resolve",
    }
    assert payload["schema_version"] == MAP_SCHEMA_VERSION
    assert payload["prompt_version"] == MAP_PROMPT_VERSION
    assert payload["cascade_signature"] == "testsig"
    # Cascade profile name set on the runner ends up in config snapshot
    assert payload["config"]["cascade_profile"] == "smoke_fake"


def test_batch_finalize_lineage_chain_is_intact(tmp_path: Path):
    runner = _build_runner(tmp_path)
    handle = _completed_handle(_audit_summary())
    runner.finalize(handle)

    run_dir = tmp_path / "runs" / "batch_test_run"
    findings = _read_jsonl(run_dir / "map" / PMCID / "findings.jsonl")
    normals  = _read_jsonl(run_dir / "normalize" / PMCID / "normal_findings.jsonl")
    groups   = _read_jsonl(run_dir / "group" / PMCID / "groups.jsonl")
    canons   = _read_jsonl(run_dir / "canonicalize" / PMCID / "canonical_rules.jsonl")
    finals   = _read_jsonl(run_dir / "resolve" / PMCID / "final_rules.jsonl")

    assert len(findings) == 1
    assert findings[0]["pmcid"] == PMCID
    assert findings[0]["chunk_id"] == "C1"
    assert findings[0]["position_in_chunk"] == 0

    assert len(normals) == 1
    nf_id = normals[0]["normal_id"]
    assert nf_id.startswith("NF_")

    assert len(groups) == 1
    group_id = groups[0]["group_id"]
    assert nf_id in groups[0]["member_ids"]

    assert len(canons) == 1
    canonical_id = canons[0]["canonical_id"]
    assert canons[0]["group_id"] == group_id
    assert nf_id in canons[0]["member_normal_ids"]

    assert len(finals) == 1
    fr = finals[0]
    assert fr["canonical_id"] == canonical_id
    assert fr["group_id"] == group_id
    assert nf_id in fr["member_normal_ids"]
    # No relations for a single rule → both files exist but are empty.
    assert _read_jsonl(run_dir / "relate" / PMCID / "raw_pairs.jsonl") == []
    assert _read_jsonl(run_dir / "relate" / PMCID / "relations.jsonl") == []


def test_batch_finalize_disabled_when_artifact_root_none(tmp_path: Path):
    runner = BatchSummarizationRunner(
        l1_voters=[_voter()],
        l2_voters=[_voter()],
        l3_model=_voter(),
        escalation_llm=None,
        config=SummarizationConfig(),
        output_dir=tmp_path / "out",
        embed_fn=lambda texts: [[0.0]] * len(texts),
        cascade_profile="smoke_fake",
        artifact_root=None,                     # disabled
        run_modern_pipeline=True,
        run_reduce=False,
    )
    handle = _completed_handle(_audit_summary())
    result = runner.finalize(handle)
    assert result["status"] == "success"
    # No `runs/` directory created by the writer.
    assert not (tmp_path / "runs").exists()


def test_batch_finalize_legacy_only_when_modern_disabled(tmp_path: Path):
    runner = BatchSummarizationRunner(
        l1_voters=[_voter()],
        l2_voters=[_voter()],
        l3_model=_voter(),
        escalation_llm=None,
        config=SummarizationConfig(),
        output_dir=tmp_path / "out",
        embed_fn=lambda texts: [[0.0]] * len(texts),
        cascade_profile="smoke_fake",
        artifact_root=tmp_path / "runs",
        artifact_run_id="legacy_run",
        run_modern_pipeline=False,
        run_reduce=False,
    )
    handle = _completed_handle(_audit_summary())
    result = runner.finalize(handle)

    run_dir = tmp_path / "runs" / "legacy_run"
    # MAP artifacts always written; no modern stage artifacts.
    assert (run_dir / "map" / PMCID / "findings.jsonl").exists()
    assert not (run_dir / "normalize").exists()
    assert not (run_dir / "canonicalize").exists()
    assert not (run_dir / "resolve").exists()
    assert result["canonical_rules"] == []
    assert result["final_rules"] == []
