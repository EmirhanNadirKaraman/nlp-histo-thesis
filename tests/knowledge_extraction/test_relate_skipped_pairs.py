"""Tests for RELATE pre-NLI gate skipped-pair trace (PERSISTENCE_TODOS #3).

Covers:
- `RelateStage.relate()` now returns a third element `skipped_pairs`.
- Each pre-NLI gate rejection becomes one SkippedPair with the expected
  metadata fields.
- `persist_relate_artifacts` writes the records to
  `runs/{run_id}/relate/{pmcid}/skipped_pairs.jsonl`.
- Backward-compat: callers that omit `skipped_pairs` still get an empty
  trace file (preserves prior on-disk shape).

No NLI / API calls anywhere. Tests are constructed so the eligible-pairs
set is empty, which short-circuits before any model load.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.stages.knowledge_extraction.observability.artifact_models import SkippedPair
from pipeline.stages.knowledge_extraction.stages import relate_stage as _rs
from pipeline.stages.knowledge_extraction.stages.relate_stage import RelateStage
from pipeline.stages.knowledge_extraction.models import (
    CanonicalRule,
    DirectionEnum,
    RelationTypeEnum,
)
from pipeline.stages.knowledge_extraction.persistence import (
    RunArtifactWriter,
    persist_relate_artifacts,
)


@pytest.fixture(autouse=True)
def _no_nli_load(monkeypatch):
    """Stub the NLI pipe loader so tests never touch HuggingFace.

    `_get_nli_pipe` is called eagerly inside `RelateStage.relate()` before the
    gate loop. None of these tests have an eligible pair (everything is
    gated out), so the pipe is never invoked — but it would still be loaded
    without this stub.
    """
    monkeypatch.setattr(_rs, "_get_nli_pipe", lambda *a, **kw: None)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rule(
    canonical_id: str,
    *,
    subject: str = "CD30",
    outcome: str = "OS",
    relation_type: RelationTypeEnum = RelationTypeEnum.prognostic,
    category: str = "IHC",
) -> CanonicalRule:
    return CanonicalRule(
        canonical_id=canonical_id,
        group_id=f"GRP_{canonical_id}",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=relation_type,
        direction=DirectionEnum.negative,
        predicate_text=f"{subject} predicts {outcome}",
        is_conflicted=False,
        study_coverage="single_study",
        category=category,
        supporting_pmcids=["PMC1"],
        member_normal_ids=[f"NF_{canonical_id}"],
        mean_grounding_score=0.8,
        finding_count=1,
    )


def _stage() -> RelateStage:
    """RelateStage with NLI thresholds — fine even though NLI never runs here."""
    return RelateStage(entailment_threshold=0.5, contradiction_threshold=0.5)


# ── Signature returns the trace list ─────────────────────────────────────────

def test_relate_returns_three_tuple_even_with_one_rule():
    # Fewer than 2 rules: early-return path. Still 3-tuple, all empty.
    out = _stage().relate([_rule("CR_a")], pmcid="PMC1")
    assert isinstance(out, tuple) and len(out) == 3
    relations, raw_pairs, skipped = out
    assert relations == [] and raw_pairs == [] and skipped == []


def test_relate_returns_three_tuple_with_zero_rules():
    out = _stage().relate([], pmcid="PMC1")
    assert out == ([], [], [])


# ── Each rejection reason produces a SkippedPair ─────────────────────────────

def test_category_mismatch_is_recorded():
    a = _rule("CR_a", category="IHC")
    b = _rule("CR_b", category="morphology")
    relations, raw_pairs, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert relations == []
    assert raw_pairs == []  # not eligible → never reaches NLI
    assert len(skipped) == 1
    sp = skipped[0]
    assert isinstance(sp, SkippedPair)
    assert sp.rule_id_a == "CR_a"
    assert sp.rule_id_b == "CR_b"
    assert sp.reason == "category_mismatch"
    assert sp.stage == "gate_pre_nli"
    assert sp.pmcid == "PMC1"
    assert sp.category_a == "IHC"
    assert sp.category_b == "morphology"


def test_relation_type_mismatch_is_recorded():
    a = _rule("CR_a", relation_type=RelationTypeEnum.prognostic)
    b = _rule("CR_b", relation_type=RelationTypeEnum.treatment_response)
    _, _, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert len(skipped) == 1
    assert skipped[0].reason == "relation_type_mismatch"
    assert skipped[0].relation_type_a == RelationTypeEnum.prognostic.value
    assert skipped[0].relation_type_b == RelationTypeEnum.treatment_response.value


def test_subject_mismatch_is_recorded():
    a = _rule("CR_a", subject="CD30")
    b = _rule("CR_b", subject="BCL2")
    _, _, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert len(skipped) == 1
    assert skipped[0].reason == "subject_mismatch"
    assert skipped[0].subject_entity_a == "CD30"
    assert skipped[0].subject_entity_b == "BCL2"


def test_outcome_mismatch_is_recorded():
    a = _rule("CR_a", outcome="OS")
    b = _rule("CR_b", outcome="PFS")
    _, _, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert len(skipped) == 1
    assert skipped[0].reason == "outcome_incompatible"
    assert skipped[0].outcome_entity_a == "OS"
    assert skipped[0].outcome_entity_b == "PFS"


def test_multiple_skips_across_three_rules():
    # All pairs fail the gate (each on different reason), so eligible=[].
    a = _rule("CR_a", category="IHC")
    b = _rule("CR_b", category="morphology")
    c = _rule("CR_c", category="treatment")
    _, _, skipped = _stage().relate([a, b, c], pmcid="PMC1")
    assert {(s.rule_id_a, s.rule_id_b) for s in skipped} == {
        ("CR_a", "CR_b"), ("CR_a", "CR_c"), ("CR_b", "CR_c"),
    }
    assert all(s.reason == "category_mismatch" for s in skipped)


# ── End-to-end through persist_relate_artifacts ──────────────────────────────

def _writer(tmp_path: Path) -> RunArtifactWriter:
    return RunArtifactWriter(run_id="r1", root_dir=tmp_path)


def test_persist_writes_skipped_pairs_jsonl(tmp_path):
    a = _rule("CR_a", category="IHC")
    b = _rule("CR_b", category="morphology")
    rels, raw, skipped = _stage().relate([a, b], pmcid="PMC9000777")

    w = _writer(tmp_path)
    persist_relate_artifacts(
        w, pmcid="PMC9000777",
        relations=rels, raw_pairs=raw, skipped_pairs=skipped,
    )

    path = tmp_path / "r1" / "relate" / "PMC9000777" / "skipped_pairs.jsonl"
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    row = lines[0]
    assert row["rule_id_a"] == "CR_a"
    assert row["rule_id_b"] == "CR_b"
    assert row["reason"] == "category_mismatch"
    assert row["stage"] == "gate_pre_nli"
    assert row["pmcid"] == "PMC9000777"


def test_persist_back_compat_no_skipped_arg_writes_empty_file(tmp_path):
    """Legacy callers passing relations + raw_pairs only must still work."""
    w = _writer(tmp_path)
    persist_relate_artifacts(
        w, pmcid="PMC9000777",
        relations=[], raw_pairs=[],  # skipped_pairs intentionally omitted
    )
    path = tmp_path / "r1" / "relate" / "PMC9000777" / "skipped_pairs.jsonl"
    assert path.exists()
    assert path.read_text().strip() == ""


def test_persist_empty_skipped_list_writes_empty_file(tmp_path):
    """No gate rejections → empty jsonl, not missing file."""
    w = _writer(tmp_path)
    persist_relate_artifacts(
        w, pmcid="PMC9000777",
        relations=[], raw_pairs=[], skipped_pairs=[],
    )
    path = tmp_path / "r1" / "relate" / "PMC9000777" / "skipped_pairs.jsonl"
    assert path.exists()
    assert path.read_text().strip() == ""


# ── B-049: non-polarity direction skip ───────────────────────────────────────

def _rule_with_direction(
    canonical_id: str,
    direction,
    *,
    subject: str = "CD30",
    outcome: str = "OS",
) -> CanonicalRule:
    """Variant of _rule() that lets the test set the direction explicitly."""
    return CanonicalRule(
        canonical_id=canonical_id,
        group_id=f"GRP_{canonical_id}",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=RelationTypeEnum.prognostic,
        direction=direction,
        predicate_text=f"{subject} predicts {outcome}",
        is_conflicted=False,
        study_coverage="single_study",
        category="IHC",
        supporting_pmcids=["PMC1"],
        member_normal_ids=[f"NF_{canonical_id}"],
        mean_grounding_score=0.8,
        finding_count=1,
    )


def test_unclear_direction_skips_pair():
    a = _rule_with_direction("CR_a", DirectionEnum.unclear)
    b = _rule_with_direction("CR_b", DirectionEnum.negative)
    _, _, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert len(skipped) == 1
    assert skipped[0].reason == "non_polarity_direction"


def test_no_direction_skips_pair():
    a = _rule_with_direction("CR_a", DirectionEnum.positive)
    b = _rule_with_direction("CR_b", DirectionEnum.no_direction)
    _, _, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert len(skipped) == 1
    assert skipped[0].reason == "non_polarity_direction"


def test_direction_none_skips_pair():
    """B-049 S1 normalizer: direction=None must be treated as 'unclear'.
    Pre-fix a naive `direction in {DirectionEnum.unclear, ...}` would have
    returned False on None, letting the pair through to NLI."""
    a = _rule_with_direction("CR_a", None)
    b = _rule_with_direction("CR_b", DirectionEnum.positive)
    _, _, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert len(skipped) == 1
    assert skipped[0].reason == "non_polarity_direction"


def test_non_polarity_skip_runs_before_other_gates():
    """Even if the rules also differ on category, the non-polarity skip
    should fire first — preserves consistent telemetry."""
    a = _rule_with_direction("CR_a", DirectionEnum.unclear)
    b = CanonicalRule(
        canonical_id="CR_b",
        group_id="GRP_b",
        subject_entity="DIFFERENT_SUBJ",  # subject mismatch as well
        outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.negative,
        predicate_text="x",
        is_conflicted=False,
        study_coverage="single_study",
        category="morphology",  # category mismatch as well
        supporting_pmcids=["PMC1"],
        member_normal_ids=["NF_b"],
        mean_grounding_score=0.8,
        finding_count=1,
    )
    _, _, skipped = _stage().relate([a, b], pmcid="PMC1")
    assert len(skipped) == 1
    assert skipped[0].reason == "non_polarity_direction"
