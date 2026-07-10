"""Regression test for B-071: _finding_to_pipeline must unwrap Enum-valued fields.

AuditableSummary.model_dump() yields Enum OBJECTS for relation_type / direction /
confidence / category. A bare str(enum) gives 'RelationTypeEnum.demographic' (the
enum repr), which never matches the silver string 'demographic' — silently
strict-penalising every matched finding (loose-F1 preserved, strict-F1 halved). The
fix routes those fields through _ev() (enum → .value); raw dicts (already strings)
pass through unchanged.
"""
from __future__ import annotations

from eval.silver.map_theta_sweep import _ev, _finding_to_pipeline
from pipeline.stages.knowledge_extraction.models import DirectionEnum, RelationTypeEnum


def test_ev_unwraps_enum_passes_through_str_and_none():
    assert _ev(RelationTypeEnum.demographic) == "demographic"
    assert _ev(DirectionEnum.positive) == "positive"
    assert _ev("increases") == "increases"
    assert _ev(None) is None


def test_finding_to_pipeline_unwraps_enum_fields():
    # the bug condition: relation_type/direction as Enum objects (as model_dump yields)
    f = {"category": "demographic", "claim": "c", "subject_entity": "a", "outcome_entity": "b",
         "relation_type": RelationTypeEnum.demographic, "direction": DirectionEnum.positive,
         "confidence": "high", "verbatim_support": "v"}
    pf = _finding_to_pipeline(f, "PMC1", "C1", 0.9)
    assert pf.relation_type == "demographic"      # NOT "RelationTypeEnum.demographic"
    assert pf.direction == "positive"             # NOT "DirectionEnum.positive"
    assert pf.category == "demographic"


def test_finding_to_pipeline_raw_strings_pass_through():
    f = {"category": "morphology", "relation_type": "increases",
         "direction": "negative", "confidence": "medium"}
    pf = _finding_to_pipeline(f, "PMC1", "C1", 0.9)
    assert pf.relation_type == "increases"
    assert pf.direction == "negative"
    assert pf.category == "morphology"
