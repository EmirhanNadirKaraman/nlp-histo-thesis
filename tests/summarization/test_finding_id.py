"""Tests for MAP `finding_id` lineage propagation.

Covers:
- determinism / sensitivity of `compute_finding_id`
- `Finding._finding_id` PrivateAttr is not exposed in the LLM-bound JSON schema
- legacy cached payloads (no `finding_id`) parse without crashing
- `NormalizeStage._merge` / `_wrap_single` populate `NormalFinding.source_finding_ids`

No LLM / API calls anywhere.
"""
from __future__ import annotations

import pytest

from pipeline.stages.summarization.stages import normalize_stage as _ns
from pipeline.stages.summarization.stages.normalize_stage import (
    NormalizeStage,
)
from pipeline.stages.summarization.models import (
    AuditableSummary,
    AuditMetadata,
    DirectionEnum,
    Finding,
    FindingScope,
    RelationTypeEnum,
    compute_finding_id,
)


PMCID = "PMC9000123"


@pytest.fixture(autouse=True)
def _stub_umls(monkeypatch):
    """Skip heavyweight scispacy/UMLS load during the test session.

    `_resolve_entity` is the only call site that triggers spacy model loads;
    stub it to an identity mapping so NormalizeStage runs in milliseconds.
    """
    def _identity(name, _synonyms):
        return (name, None)
    monkeypatch.setattr(_ns, "_resolve_entity", _identity)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _finding(
    claim: str = "MYC overexpression predicts worse OS",
    *,
    subject: str | None = "MYC",
    outcome: str | None = "OS",
    relation_type: RelationTypeEnum = RelationTypeEnum.prognostic,
    direction: DirectionEnum = DirectionEnum.negative,
    category: str = "molecular_genetics",
    text_element_id: int = 101,
    grounding_score: float = 0.85,
) -> Finding:
    return Finding(
        category=category,
        claim=claim,
        evidence=[f"S1|{PMCID}|{text_element_id}"],
        confidence="high",
        verbatim_support="MYC overexpression was associated with worse OS.",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=relation_type,
        direction=direction,
        scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        grounding_score=grounding_score,
    )


# ── compute_finding_id determinism ───────────────────────────────────────────

def test_compute_finding_id_is_deterministic():
    a = compute_finding_id(PMCID, "C1", 0, "MYC predicts worse OS")
    b = compute_finding_id(PMCID, "C1", 0, "MYC predicts worse OS")
    assert a == b
    assert isinstance(a, str) and len(a) == 12


def test_compute_finding_id_normalizes_whitespace_and_case():
    a = compute_finding_id(PMCID, "C1", 0, "MYC predicts worse OS")
    b = compute_finding_id(PMCID, "C1", 0, "  myc  predicts   WORSE os  ")
    assert a == b


def test_compute_finding_id_changes_with_claim_chunk_position_pmcid():
    base = compute_finding_id(PMCID, "C1", 0, "MYC predicts worse OS")
    assert compute_finding_id(PMCID, "C1", 0, "BCL2 predicts worse OS") != base
    assert compute_finding_id(PMCID, "C2", 0, "MYC predicts worse OS") != base
    assert compute_finding_id(PMCID, "C1", 1, "MYC predicts worse OS") != base
    assert compute_finding_id("PMC9999999", "C1", 0, "MYC predicts worse OS") != base


# ── Finding PrivateAttr behaviour ────────────────────────────────────────────

def test_finding_id_defaults_to_none_and_is_not_in_model_dump():
    f = _finding()
    assert f.finding_id is None
    # Private attrs are not serialized — keep them out of the LLM/cache surface.
    assert "finding_id" not in f.model_dump()
    assert "_finding_id" not in f.model_dump()


def test_finding_id_not_in_openai_strict_schema():
    schema = Finding.model_json_schema()
    props = schema.get("properties", {})
    # The MAP LLM is bound via `with_structured_output(AuditableSummary, strict=True)`
    # which derives its schema from Finding's properties. The id must not leak in.
    assert "finding_id" not in props
    assert "_finding_id" not in props


def test_legacy_finding_payload_without_finding_id_parses_cleanly():
    raw = {
        "category": "IHC",
        "claim": "CD30 predicts worse OS",
        "evidence": [f"S1|{PMCID}|101"],
        "confidence": "high",
        "verbatim_support": "CD30 was associated with worse OS.",
        "subject_entity": "CD30",
        "outcome_entity": "OS",
        "relation_type": "prognostic",
        "direction": "negative",
        "scope": None,
        "grounding_score": 0.7,
    }
    f = Finding.model_validate(raw)
    assert f.finding_id is None  # not in cache, not crashed


def test_legacy_cached_payload_with_extra_finding_id_key_is_ignored():
    # Older callers may have added a finding_id key by hand; Pydantic ignores
    # unknown public keys unless the model is configured otherwise. The
    # PrivateAttr is only writable via set_finding_id, so the value is *not*
    # silently adopted from input.
    raw = {
        "category": "IHC",
        "claim": "CD30 predicts worse OS",
        "evidence": [f"S1|{PMCID}|101"],
        "confidence": "high",
        "verbatim_support": "CD30 was associated with worse OS.",
        "subject_entity": "CD30",
        "outcome_entity": "OS",
        "relation_type": "prognostic",
        "direction": "negative",
        "scope": None,
        "grounding_score": 0.7,
        "finding_id": "deadbeef0000",
    }
    f = Finding.model_validate(raw)
    assert f.finding_id is None


def test_finding_id_survives_model_copy_update():
    f = _finding()
    f.set_finding_id("abc123def456")
    g = f.model_copy(update={"subject_entity": "BCL2"})
    assert g.finding_id == "abc123def456"
    assert g.subject_entity == "BCL2"


def test_auditable_summary_construction_preserves_finding_id():
    f = _finding()
    f.set_finding_id(compute_finding_id(PMCID, "C1", 0, f.claim))
    cs = AuditableSummary(
        chunk_id="C1",
        findings=[f],
        summary_text="x",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=[f"S1|{PMCID}|101"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )
    # _drop_invalid_findings rebuilds Findings from dicts via model_validate.
    # We expect the id assigned pre-construction to survive only when the
    # caller passes a Finding instance (not a dict) — which is the in-pipeline
    # path. Defensive assertion below documents that contract.
    assert cs.findings[0].finding_id is not None


# ── NormalizeStage propagation ───────────────────────────────────────────────

def test_normalize_merge_populates_source_finding_ids():
    stage = NormalizeStage()
    f1 = _finding(claim="MYC predicts worse OS")
    f2 = _finding(claim="MYC is associated with worse OS")
    # Same te_id + same (subject, outcome, relation_type) → merged into one NF.
    f1.set_finding_id(compute_finding_id(PMCID, "C1", 0, f1.claim))
    f2.set_finding_id(compute_finding_id(PMCID, "C1", 1, f2.claim))

    nfs = stage.normalize([f1, f2], PMCID)
    assert len(nfs) == 1
    nf = nfs[0]
    assert set(nf.source_finding_ids) == {f1.finding_id, f2.finding_id}
    # Deterministic order: first-seen.
    assert nf.source_finding_ids == [f1.finding_id, f2.finding_id]


def test_normalize_wrap_single_populates_source_finding_ids():
    stage = NormalizeStage()
    # Ungroupable: relation_type=unclear forces _wrap_single.
    f = _finding(relation_type=RelationTypeEnum.unclear)
    f.set_finding_id(compute_finding_id(PMCID, "C1", 0, f.claim))
    nfs = stage.normalize([f], PMCID)
    assert len(nfs) == 1
    assert nfs[0].source_finding_ids == [f.finding_id]


def test_normalize_handles_findings_without_finding_id_gracefully():
    """Legacy callers that don't assign finding_id must not crash NormalizeStage."""
    stage = NormalizeStage()
    f = _finding(claim="CD30 predicts worse OS")
    assert f.finding_id is None
    nfs = stage.normalize([f], PMCID)
    assert len(nfs) == 1
    # None ids are skipped; the field is present but empty.
    assert nfs[0].source_finding_ids == []
