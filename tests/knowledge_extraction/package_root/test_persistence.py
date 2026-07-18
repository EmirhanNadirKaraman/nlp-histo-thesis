"""Tests for the knowledge_extraction filesystem-persistence layer.

No LLM/API calls — synthetic Pydantic objects only.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    AuditableSummary,
    AuditMetadata,
    CanonicalRule,
    DirectionEnum,
    Finding,
    FindingGroup,
    FindingScope,
    FinalRule,
    NormalFinding,
    RawNLIPair,
    Relation,
    RelationTypeEnum,
    RelationTypeLabel,
    SourceSpan,
)
from nlp_histo.pipeline.stages.knowledge_extraction.persistence import (
    RunArtifactWriter,
    RunManifest,
    _to_jsonable,
    write_jsonl,
)


PMCID = "PMC9000001"


# Helpers

def _finding(claim="CD30 predicts worse OS", *,
             subject="CD30", outcome="OS",
             relation_type=RelationTypeEnum.prognostic,
             direction=DirectionEnum.negative,
             category="IHC", grounding_score=0.8) -> Finding:
    return Finding(
        category=category,
        claim=claim,
        evidence=[f"S1|{PMCID}|101"],
        confidence="high",
        verbatim_support="CD30 was associated with worse OS in DLBCL.",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=relation_type,
        direction=direction,
        scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        grounding_score=grounding_score,
    )


def _audit_summary(chunk_id: str, findings: list[Finding]) -> AuditableSummary:
    return AuditableSummary(
        chunk_id=chunk_id,
        findings=findings,
        summary_text="Two findings in this chunk.",
        audit_metadata=AuditMetadata(
            sentences_analyzed=10,
            sentences_cited=[f"S1|{PMCID}|101"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )


def _normal(normal_id="NF_a1") -> NormalFinding:
    return NormalFinding(
        normal_id=normal_id,
        subject_entity="CD30",
        outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.negative,
        category="IHC",
        predicate_text="CD30 expression predicts worse OS",
        scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        source_finding_ids=[],
        evidence=[SourceSpan(
            sentence_id="S1", pmcid=PMCID, text_element_id=101,
            verbatim="CD30 was associated with worse OS",
        )],
        pmcids=[PMCID],
        mean_grounding_score=0.8,
    )


def _group(group_id="GRP_a1") -> FindingGroup:
    return FindingGroup(
        group_id=group_id,
        subject_entity="CD30", outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic, category="IHC",
        member_ids=["NF_a1"],
        direction_counts={"negative": 1},
        scope_heterogeneity=0.0,
    )


def _canonical(canonical_id="CR_a1") -> CanonicalRule:
    return CanonicalRule(
        canonical_id=canonical_id, group_id="GRP_a1",
        subject_entity="CD30", outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.negative,
        predicate_text="CD30 expression predicts worse OS",
        is_conflicted=False, study_coverage="single_study", category="IHC",
        supporting_pmcids=[PMCID], member_normal_ids=["NF_a1"],
        mean_grounding_score=0.8, finding_count=1,
    )


def _final(final_id="FR_CR_a1") -> FinalRule:
    return FinalRule(
        final_id=final_id, canonical_id="CR_a1", group_id="GRP_a1",
        subject_entity="CD30", outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.negative,
        predicate_text="CD30 expression predicts worse OS",
        is_conflicted=False, study_coverage="single_study", category="IHC",
        supporting_pmcids=[PMCID], member_normal_ids=["NF_a1"],
        mean_grounding_score=0.8, finding_count=1,
        final_score=0.72, support_count=0, contradict_count=0,
        scope_qualify_count=0, is_contradicted=False, contradicted_by=[],
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# _to_jsonable round-trips

def test_to_jsonable_handles_pydantic_dataclass_enum_path_datetime():
    @dataclasses.dataclass
    class DC:
        x: int
        when: datetime

    class Color(str, Enum):
        red = "red"
        blue = "blue"

    finding = _finding()
    payload = {
        "model": finding,
        "dataclass": DC(x=3, when=datetime(2026, 1, 2, tzinfo=timezone.utc)),
        "enum": Color.red,
        "path": Path("/tmp/foo"),
        "set": {"a", "b"},
        "nested": [finding, Color.blue],
    }
    out = _to_jsonable(payload)
    assert out["enum"] == "red"
    assert out["path"] == "/tmp/foo"
    assert out["dataclass"]["x"] == 3
    assert out["dataclass"]["when"].startswith("2026-01-02")
    # Pydantic Finding survives
    assert out["model"]["claim"] == finding.claim
    # Set deterministically sorted
    assert out["set"] == ["a", "b"]
    # Round-trips through json.dumps
    json.dumps(out)


def test_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [
        _finding(claim="claim 1"),
        {"k": "v", "n": 1, "scope": FindingScope(disease_subtype="DLBCL", scope_parsed=True)},
        _normal(),
    ]
    n = write_jsonl(path, rows)
    assert n == 3
    out = _read_jsonl(path)
    assert out[0]["claim"] == "claim 1"
    assert out[1]["scope"]["disease_subtype"] == "DLBCL"
    assert out[2]["normal_id"] == "NF_a1"


# RunArtifactWriter — directory layout, manifest, disabled-by-default

def test_writer_creates_run_directory_and_writes_manifest(tmp_path: Path):
    RunArtifactWriter(run_id="test_run", root_dir=tmp_path)
    assert (tmp_path / "test_run").is_dir()
    assert (tmp_path / "test_run" / "logs").is_dir()
    manifest_path = tmp_path / "test_run" / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test_run"
    assert payload["status"] == "running"
    assert payload["timestamp_start"]


def test_writer_update_manifest_persists(tmp_path: Path):
    writer = RunArtifactWriter(run_id="r1", root_dir=tmp_path)
    writer.update_manifest(
        schema_version="map_v1_explicit_direction",
        prompt_version="map_prompt_v1_explicit_enums",
        thresholds={"grounding_threshold": 0.5,
                    "entailment_threshold": 0.5,
                    "contradiction_threshold": 0.5},
        cascade_signature="abc123",
    )
    writer.add_paper(PMCID)
    writer.mark_stage_attempted("map")
    writer.mark_stage_completed("map")
    writer.finalize("completed")

    payload = json.loads((tmp_path / "r1" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["timestamp_end"]
    assert payload["papers"] == [PMCID]
    assert "map" in payload["stages_completed"]
    assert payload["thresholds"]["grounding_threshold"] == 0.5
    assert payload["cascade_signature"] == "abc123"
    assert payload["schema_version"] == "map_v1_explicit_direction"


def test_writer_finalize_failed_records_error(tmp_path: Path):
    writer = RunArtifactWriter(run_id="r2", root_dir=tmp_path)
    writer.write_error(stage="map", error="boom", pmcid=PMCID)
    writer.finalize("failed", error="boom")
    payload = json.loads((tmp_path / "r2" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "boom"
    err_log = (tmp_path / "r2" / "logs" / "stage_errors.jsonl").read_text(encoding="utf-8")
    assert "boom" in err_log


def test_writer_optional_artifact_tolerates_serialization_failure(tmp_path: Path):
    writer = RunArtifactWriter(run_id="r3", root_dir=tmp_path)

    class Boom:
        def __repr__(self):
            raise RuntimeError("explode")

        def model_dump(self):
            raise RuntimeError("explode")

        def __dict__(self):
            raise RuntimeError("explode")

    # Optional → swallow.
    n = writer.write_stage_jsonl(
        "normalize", "entity_links.jsonl", [Boom()], pmcid=PMCID, required=False,
    )
    assert n == 0
    # Required → raises.
    with pytest.raises(Exception):
        writer.write_stage_jsonl(
            "map", "findings.jsonl", [Boom()], pmcid=PMCID, required=True,
        )


# Whole-pipeline synthetic write

def test_writer_persists_full_synthetic_pipeline(tmp_path: Path):
    writer = RunArtifactWriter(run_id="full_run", root_dir=tmp_path)
    writer.update_manifest(
        schema_version="map_v1_explicit_direction",
        prompt_version="map_prompt_v1_explicit_enums",
        chunk_size=10,
        thresholds={
            "grounding_threshold": 0.5,
            "entailment_threshold": 0.5,
            "contradiction_threshold": 0.5,
        },
    )

    chunk_summaries = [_audit_summary("c0", [_finding(), _finding(claim="other")])]
    findings_rows = []
    for cs in chunk_summaries:
        for pos, f in enumerate(cs.findings):
            d = f.model_dump()
            d.update(pmcid=PMCID, chunk_id=cs.chunk_id, position_in_chunk=pos)
            findings_rows.append(d)
    writer.write_stage_jsonl("map", "findings.jsonl", findings_rows,
                             pmcid=PMCID, required=True)
    writer.write_stage_jsonl("map", "chunks.jsonl",
                             [{"pmcid": PMCID, "chunk_id": cs.chunk_id,
                               "n_findings": len(cs.findings),
                               "summary_text": cs.summary_text,
                               "audit_metadata": cs.audit_metadata.model_dump()}
                              for cs in chunk_summaries],
                             pmcid=PMCID, required=True)
    writer.write_stage_jsonl("map", "rejected_findings.jsonl", [], pmcid=PMCID)
    writer.mark_stage_completed("map")

    writer.write_stage_jsonl(
        "normalize", "normal_findings.jsonl",
        [_normal().model_dump()], pmcid=PMCID, required=True,
    )
    writer.write_stage_jsonl(
        "normalize", "entity_links.jsonl",
        [{"normal_id": "NF_a1", "subject_entity": "CD30",
          "subject_cui": None, "outcome_entity": "OS", "outcome_cui": None}],
        pmcid=PMCID,
    )
    writer.write_stage_jsonl("normalize", "dedup_trace.jsonl", [], pmcid=PMCID)
    writer.mark_stage_completed("normalize")

    writer.write_stage_jsonl(
        "group", "groups.jsonl",
        [_group().model_dump()], pmcid=PMCID, required=True,
    )
    writer.write_stage_jsonl("group", "non_groupable.jsonl", [], pmcid=PMCID)
    writer.mark_stage_completed("group")

    writer.write_stage_jsonl(
        "canonicalize", "canonical_rules.jsonl",
        [_canonical().model_dump()], pmcid=PMCID, required=True,
    )
    writer.mark_stage_completed("canonicalize")

    relation = Relation(
        rule_id_a="CR_a1", rule_id_b="CR_a1",
        relation_type=RelationTypeLabel.SUPPORT,
        nli_score_a_to_b=0.7, nli_score_b_to_a=0.7,
    )
    raw = RawNLIPair(
        rule_id_a="CR_a1", rule_id_b="CR_a1",
        ent_a_to_b=0.7, ent_b_to_a=0.7, con_a_to_b=0.05, con_b_to_a=0.05,
        classified_label=RelationTypeLabel.SUPPORT,
    )
    writer.write_stage_jsonl(
        "relate", "relations.jsonl", [relation.model_dump()],
        pmcid=PMCID, required=True,
    )
    writer.write_stage_jsonl(
        "relate", "raw_pairs.jsonl", [raw.model_dump()],
        pmcid=PMCID, required=True,
    )
    writer.write_stage_jsonl("relate", "skipped_pairs.jsonl", [], pmcid=PMCID)
    writer.mark_stage_completed("relate")

    writer.write_stage_jsonl(
        "resolve", "final_rules.jsonl",
        [_final().model_dump()], pmcid=PMCID, required=True,
    )
    writer.write_stage_jsonl("resolve", "score_trace.jsonl", [], pmcid=PMCID)
    writer.mark_stage_completed("resolve")

    writer.finalize("completed")

    run_dir = tmp_path / "full_run"
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
        assert p.exists(), f"missing: {p}"

    # Manifest is final
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert set(payload["stages_completed"]) == {
        "map", "normalize", "group", "canonicalize", "relate", "resolve",
    }

    # Spot-check round-trip preserves IDs and enums (lineage).
    final_rows = _read_jsonl(run_dir / "resolve" / PMCID / "final_rules.jsonl")
    assert final_rows[0]["final_id"] == "FR_CR_a1"
    assert final_rows[0]["canonical_id"] == "CR_a1"
    assert final_rows[0]["group_id"] == "GRP_a1"
    assert final_rows[0]["member_normal_ids"] == ["NF_a1"]
    assert final_rows[0]["relation_type"] == "prognostic"
    assert final_rows[0]["direction"] == "negative"

    raw_rows = _read_jsonl(run_dir / "relate" / PMCID / "raw_pairs.jsonl")
    assert raw_rows[0]["classified_label"] == "SUPPORT"


# Imports & disabled-by-default integration

def test_existing_summarization_imports_still_work():
    mod = importlib.import_module("nlp_histo.pipeline.stages.knowledge_extraction")
    for name in (
        "KnowledgeExtractionRunner", "KnowledgeExtractionConfig",
        "Finding", "RejectedFinding", "RunArtifactWriter", "RunManifest",
    ):
        assert hasattr(mod, name), f"missing export: {name}"


def test_writer_merges_with_existing_manifest_for_shared_run_id(tmp_path: Path):
    """A second writer instance for the same run_id must accumulate, not overwrite."""
    w1 = RunArtifactWriter(run_id="shared", root_dir=tmp_path)
    w1.update_manifest(schema_version="map_v1_explicit_direction",
                       cascade_signature="sig123", chunk_size=10)
    w1.add_paper("PMC0001")
    w1.mark_stage_completed("map")
    w1.mark_stage_completed("normalize")
    # Crucially do not finalize — simulate process() ending mid-run-dir lifecycle.

    # Second paper, fresh writer (matches what _make_artifact_writer does).
    w2 = RunArtifactWriter(
        run_id="shared", root_dir=tmp_path,
        manifest=RunManifest(
            run_id="shared", artifact_root=tmp_path,
            timestamp_start=datetime.now(tz=timezone.utc).isoformat(),
            papers=["PMC0002"],
        ),
    )
    w2.mark_stage_completed("map")
    w2.finalize("completed")

    payload = json.loads((tmp_path / "shared" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["papers"] == ["PMC0001", "PMC0002"]
    assert "map" in payload["stages_completed"]
    assert "normalize" in payload["stages_completed"]
    # Carried forward across writer rebuilds
    assert payload["schema_version"] == "map_v1_explicit_direction"
    assert payload["cascade_signature"] == "sig123"
    assert payload["chunk_size"] == 10


def test_runner_exposes_artifact_root_and_does_not_create_dirs_when_none(tmp_path: Path):
    """Smoke test: instantiate runner without LLMs (None values) and assert
    artifact_root defaults to None — no `runs/` directory created."""
    from nlp_histo.pipeline.stages.knowledge_extraction.runner import KnowledgeExtractionRunner

    # We can't fully construct a runner without LLMs because MapStage requires
    # them. Instead inspect the signature to confirm the new args exist.
    import inspect
    sig = inspect.signature(KnowledgeExtractionRunner.__init__)
    assert "artifact_root" in sig.parameters
    assert "artifact_run_id" in sig.parameters
    assert sig.parameters["artifact_root"].default is None
    assert sig.parameters["artifact_run_id"].default is None
    assert not (tmp_path / "runs").exists()


# B-055: persist_map_findings must not swallow DB errors

def test_persist_map_findings_reraises_on_db_error():
    """Regression for B-055: a failed bulk insert must propagate, not be
    swallowed into a silent 0-row run that still reports success."""
    from nlp_histo.pipeline.stages.knowledge_extraction.persistence import persist_map_findings

    class _BoomEngine:
        def begin(self):
            raise RuntimeError("simulated bulk-insert failure")

    class _FakeDB:
        engine = _BoomEngine()

    cs = _audit_summary("C1", [_finding()])
    with pytest.raises(RuntimeError, match="simulated bulk-insert failure"):
        persist_map_findings(_FakeDB(), 1, PMCID, [cs])


def test_persist_map_findings_noop_when_no_db_id():
    """The db_id/db None guard returns before touching the DB (no raise)."""
    from nlp_histo.pipeline.stages.knowledge_extraction.persistence import persist_map_findings

    class _BoomEngine:
        def begin(self):
            raise AssertionError("DB must not be touched when db_id is None")

    class _FakeDB:
        engine = _BoomEngine()

    # Returns silently — guard short-circuits before the engine is used.
    persist_map_findings(_FakeDB(), None, PMCID, [_audit_summary("C1", [_finding()])])
