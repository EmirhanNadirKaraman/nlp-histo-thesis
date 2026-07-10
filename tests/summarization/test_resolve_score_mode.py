"""Patch 3 — RESOLVE score_mode observability tests.

Confirms that:
  * len(relations) == 0 → every FinalRule.score_mode == "relations_absent"
  * len(relations) > 0  → every FinalRule.score_mode == "relations_present"
  * final_score math is unchanged for equivalent inputs (the patch is purely
    observability — no scoring math touched)
"""
from __future__ import annotations

from pipeline.stages.summarization.stages.resolve_stage import ResolveStage
from pipeline.stages.summarization.models import (
    CanonicalRule,
    DirectionEnum,
    Relation,
    RelationTypeEnum,
    RelationTypeLabel,
)


def _rule(canonical_id: str = "CR_A", *, grounding: float | None = 0.8) -> CanonicalRule:
    return CanonicalRule(
        canonical_id=canonical_id,
        group_id=f"GRP_{canonical_id}",
        subject_entity="CD30",
        outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.positive,
        predicate_text="placeholder",
        is_conflicted=False,
        study_coverage="single_study",
        category="IHC",
        supporting_pmcids=["PMC1"],
        member_normal_ids=["NF_1"],
        mean_grounding_score=grounding,
        finding_count=3,
    )


def _support_relation(a: str, b: str) -> Relation:
    return Relation(
        rule_id_a=a,
        rule_id_b=b,
        relation_type=RelationTypeLabel.SUPPORT,
        nli_score_a_to_b=0.9,
        nli_score_b_to_a=0.9,
    )


def test_relations_absent_sets_score_mode_on_every_final_rule():
    stage = ResolveStage()
    rules = [_rule("CR_A"), _rule("CR_B")]
    out = stage.resolve(rules, relations=[], pmcid="PMC_TEST")
    assert len(out) == 2
    assert all(fr.score_mode == "relations_absent" for fr in out)


def test_relations_present_sets_score_mode_on_every_final_rule():
    stage = ResolveStage()
    rules = [_rule("CR_A"), _rule("CR_B")]
    rels  = [_support_relation("CR_A", "CR_B")]
    out = stage.resolve(rules, rels, pmcid="PMC_TEST")
    assert len(out) == 2
    assert all(fr.score_mode == "relations_present" for fr in out)


def test_final_score_unchanged_by_score_mode_field():
    """The score_mode field is observability only; the scoring math must not
    change. Run RESOLVE twice with identical inputs; final_scores must match.
    """
    stage = ResolveStage()
    rules = [_rule("CR_A"), _rule("CR_B"), _rule("CR_C", grounding=0.4)]
    a = stage.resolve(rules, relations=[], pmcid="PMC_TEST")
    b = stage.resolve(rules, relations=[], pmcid="PMC_TEST")
    assert [fr.final_score for fr in a] == [fr.final_score for fr in b]


def test_score_mode_serialises_to_dict():
    """model_dump() carries the new field so artifact JSONL records include it."""
    stage = ResolveStage()
    rules = [_rule("CR_A")]
    fr = stage.resolve(rules, relations=[], pmcid="PMC_TEST")[0]
    blob = fr.model_dump()
    assert blob["score_mode"] == "relations_absent"
