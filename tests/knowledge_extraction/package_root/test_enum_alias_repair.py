"""Regression test for B-018 + extended enum repair (this session).

Covers:
* relation_type alias map (B-018): `prognosis` → `prognostic`,
  `treatment` → `treatment_response`.
* category alias map: `demographics` → `demographic` (B-016).
* Case-insensitive repair across category, relation_type, direction, and
  confidence (the new validators added 2026-05-15).
* Confidence alias map: `moderate` → `medium`, `weak` → `low`,
  `strong` → `high`, `very high` → `high`, etc.
* Raw values are still captured on the `_raw_*` PrivateAttrs even after
  alias / case repair (B-015 contract).

No LLM or DB required.
"""
from __future__ import annotations

import pytest

from pipeline.stages.knowledge_extraction.models import (
    DirectionEnum,
    Finding,
    RelationTypeEnum,
)


def _make(**overrides) -> dict:
    """Build a minimal Finding payload — callers override the field under
    test, everything else stays valid."""
    payload = dict(
        category="IHC",
        claim="CD30 positive in tumour cells",
        evidence=["S1|PMC123|456"],
        confidence="high",
        verbatim_support="CD30 was expressed",
        subject_entity="CD30",
        outcome_entity="staining",
        relation_type="expression",
        direction="positive",
        scope=None,
        grounding_score=None,
    )
    payload.update(overrides)
    return payload


# ── relation_type alias repair ────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("prognosis",          RelationTypeEnum.prognostic),
    ("treatment",          RelationTypeEnum.treatment_response),
    ("morphology",         RelationTypeEnum.has_feature),
    ("ihc",                RelationTypeEnum.expression),
    ("IHC",                RelationTypeEnum.expression),
    ("molecular_genetics", RelationTypeEnum.expression),
])
def test_relation_type_alias_repair(raw, expected):
    f = Finding.model_validate(_make(relation_type=raw))
    assert f.relation_type is expected
    # Raw value preserved per B-015 contract.
    assert f.raw_relation_type == raw


def test_relation_type_staging_not_aliased_falls_to_unclear():
    """`staging` is intentionally NOT aliased — descriptive vs prognostic
    crossover needs claim context. Falls through to unclear with
    `cross_field_bleed` observability so loss stays measurable."""
    f = Finding.model_validate(_make(relation_type="staging"))
    assert f.relation_type is RelationTypeEnum.unclear
    assert f.raw_relation_type == "staging"


@pytest.mark.parametrize("raw, expected", [
    ("Has_Feature", RelationTypeEnum.has_feature),
    ("EXPRESSION",  RelationTypeEnum.expression),
    ("Prognostic",  RelationTypeEnum.prognostic),
])
def test_relation_type_case_repair(raw, expected):
    f = Finding.model_validate(_make(relation_type=raw))
    assert f.relation_type is expected
    assert f.raw_relation_type == raw


def test_relation_type_case_folded_alias():
    """Capitalized alias still routes through alias_repair (e.g.
    `Prognosis` → `prognostic`, not coerced to `unclear`)."""
    f = Finding.model_validate(_make(relation_type="Prognosis"))
    assert f.relation_type is RelationTypeEnum.prognostic
    assert f.raw_relation_type == "Prognosis"


def test_unknown_relation_type_coerced_to_unclear():
    """An invalid value with no alias and no case match coerces to unclear —
    documents the fallback path so future refactors don't change semantics."""
    f = Finding.model_validate(_make(relation_type="associates_with"))
    assert f.relation_type is RelationTypeEnum.unclear


# ── cross-field bleed observability ───────────────────────────────────────────

@pytest.mark.parametrize("raw, recovered_value", [
    # Recovered via alias: log records the valid relation_type as coerced_value.
    ("prognosis",          "prognostic"),
    ("treatment",          "treatment_response"),
    ("morphology",         "has_feature"),
    ("ihc",                "expression"),
    ("IHC",                "expression"),
    ("molecular_genetics", "expression"),
    # Unrecovered: still tagged as cross_field_bleed; coerced_value is "unclear".
    ("staging",            "unclear"),
])
def test_cross_field_bleed_logged(monkeypatch, raw, recovered_value):
    """Category-name bleed into relation_type is tagged `cross_field_bleed`
    in enum_observations.jsonl regardless of whether alias-repair recovered
    the finding. Lets us count exact loss per run."""
    captured: list[dict] = []
    import pipeline.stages.knowledge_extraction.models as models_mod

    def _capture(**kw):
        captured.append(kw)

    monkeypatch.setattr(models_mod, "log_enum_observation", _capture)
    Finding.model_validate(_make(relation_type=raw))
    bleed_records = [r for r in captured
                     if r.get("field_name") == "relation_type"
                     and r.get("reason") == "cross_field_bleed"]
    assert len(bleed_records) == 1, captured
    assert bleed_records[0]["raw_value"] == raw
    assert bleed_records[0]["coerced_value"] == recovered_value


def test_non_bleed_unknown_still_tagged_unknown_value(monkeypatch):
    """A relation_type that is genuinely unknown (not a category-name) keeps
    its `unknown_value` reason — bleed detection must not over-fire."""
    captured: list[dict] = []
    import pipeline.stages.knowledge_extraction.models as models_mod

    def _capture(**kw):
        captured.append(kw)

    monkeypatch.setattr(models_mod, "log_enum_observation", _capture)
    Finding.model_validate(_make(relation_type="associates_with"))
    rt_records = [r for r in captured if r.get("field_name") == "relation_type"]
    assert len(rt_records) == 1
    assert rt_records[0]["reason"] == "unknown_value"


# ── category alias + case repair ──────────────────────────────────────────────

def test_category_alias_demographics_to_demographic():
    f = Finding.model_validate(_make(category="demographics"))
    assert f.category == "demographic"
    assert f.raw_category == "demographics"


def test_category_case_repair_titlecase():
    f = Finding.model_validate(_make(category="Demographic"))
    assert f.category == "demographic"


def test_category_case_repair_lowercase_ihc_preserves_uppercase_canonical():
    """`IHC` is the canonical uppercase form; `ihc` from the LLM must be
    repaired to `IHC`, not collapsed to a lowercase variant."""
    f = Finding.model_validate(_make(category="ihc"))
    assert f.category == "IHC"


def test_category_alias_with_case_fold():
    """Capitalized legacy alias (`Demographics`) routes through case-folded
    alias-repair branch."""
    f = Finding.model_validate(_make(category="Demographics"))
    assert f.category == "demographic"


# ── direction repair ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("POSITIVE",   DirectionEnum.positive),
    ("Positive",   DirectionEnum.positive),
    ("NO_DIRECTION", DirectionEnum.no_direction),
])
def test_direction_case_repair(raw, expected):
    f = Finding.model_validate(_make(direction=raw))
    assert f.direction is expected
    assert f.raw_direction == raw


def test_direction_null_becomes_no_direction():
    """Legacy contract preserved: `null`/empty direction coerces to
    no_direction, not unclear."""
    f = Finding.model_validate(_make(direction="null"))
    assert f.direction is DirectionEnum.no_direction


# ── confidence alias + case repair (this session) ─────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("HIGH",     "high"),
    ("Low",      "low"),
    ("MEDIUM",   "medium"),
])
def test_confidence_case_repair(raw, expected):
    f = Finding.model_validate(_make(confidence=raw))
    assert f.confidence == expected


@pytest.mark.parametrize("raw, expected", [
    ("moderate",   "medium"),
    ("weak",       "low"),
    ("strong",     "high"),
    ("very high",  "high"),
    ("very_high",  "high"),
    ("Very_Low",   "low"),
    ("uncertain",  "low"),
])
def test_confidence_alias_repair(raw, expected):
    f = Finding.model_validate(_make(confidence=raw))
    assert f.confidence == expected


def test_confidence_unknown_value_still_drops_finding():
    """The alias map is intentionally narrow — values outside it must still
    fail Literal validation so we can mine them via `bad_findings.jsonl`."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Finding.model_validate(_make(confidence="definitely_high"))


# ── direction alias repair (Issue 8, MAP_PROMPT_AUDIT) ───────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("maybe",    DirectionEnum.unclear),
    ("possibly", DirectionEnum.unclear),
    ("perhaps",  DirectionEnum.unclear),
    ("likely",   DirectionEnum.unclear),
    ("unknown",  DirectionEnum.unclear),
    ("Maybe",    DirectionEnum.unclear),  # case-fold path
    ("POSSIBLY", DirectionEnum.unclear),
    ("none",     DirectionEnum.no_direction),
    ("None",     DirectionEnum.no_direction),
    ("n/a",      DirectionEnum.no_direction),
    ("NA",       DirectionEnum.no_direction),
])
def test_direction_alias_repair(raw, expected):
    f = Finding.model_validate(_make(direction=raw))
    assert f.direction is expected
    # Raw value still captured per B-015.
    assert f._raw_direction == raw


def test_direction_unknown_value_still_coerces_to_unclear():
    """Values outside both the enum and the alias map fall through to
    `unclear` with reason='unknown_value' — preserves measurement surface."""
    f = Finding.model_validate(_make(direction="sideways"))
    assert f.direction is DirectionEnum.unclear
    assert f._raw_direction == "sideways"
