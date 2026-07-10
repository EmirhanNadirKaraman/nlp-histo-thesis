"""Tests for 'demographic' category in main pipeline models and validator.

No LLM or database required.
"""
from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from pipeline.stages.knowledge_extraction.models import (
    AuditableSummary,
    AuditMetadata,
    Finding,
    FindingScope,
    NormalFinding,
    RelationTypeEnum,
    SourceSpan,
)
from pipeline.stages.knowledge_extraction.stages.group_stage import _group_id
from pipeline.stages.knowledge_extraction.routing.schema_validator import (
    SchemaValidator,
    _VALID_CATEGORIES,
)

PMCID = "PMC99999"


# ── Schema acceptance ─────────────────────────────────────────────────────────

def test_finding_demographic_valid():
    """category='demographic' passes Pydantic validation on Finding."""
    f = Finding(
        category="demographic",
        claim="Slight male predominance (37 men/29 women)",
        evidence=["S1|PMC123|456"],
        confidence="high",
        verbatim_support="slight male predominance (37 men/29 women)",
        relation_type=RelationTypeEnum.demographic,
    )
    assert f.category == "demographic"


def test_finding_unknown_category_rejected():
    """Unknown categories are still rejected by Pydantic."""
    with pytest.raises(ValidationError):
        Finding(
            category="bogus_category",
            claim="x",
            evidence=["S1|PMC123|456"],
            confidence="high",
            verbatim_support="x",
        )


def test_normal_finding_demographic_valid():
    """category='demographic' passes Pydantic validation on NormalFinding."""
    nf = NormalFinding(
        normal_id="NF_demo",
        subject_entity="CEAN patients",
        outcome_entity="sex distribution",
        relation_type=RelationTypeEnum.demographic,
        direction=None,
        category="demographic",
        predicate_text="Slight male predominance",
        scope=FindingScope(),
        source_finding_ids=[],
        evidence=[SourceSpan(sentence_id="S1", pmcid=PMCID, text_element_id=1, verbatim="x")],
        pmcids=[PMCID],
        mean_grounding_score=None,
    )
    assert nf.category == "demographic"


# ── SchemaValidator ───────────────────────────────────────────────────────────

def test_valid_categories_includes_demographic():
    """Regression guard: _VALID_CATEGORIES must contain 'demographic'."""
    assert "demographic" in _VALID_CATEGORIES


def test_schema_validator_accepts_demographic():
    """SchemaValidator passes a demographic finding without INVALID_CATEGORY."""
    from pipeline.stages.knowledge_extraction.routing.models import ReasonCode

    summary = AuditableSummary(
        chunk_id="PMC123|0",
        findings=[
            Finding(
                category="demographic",
                claim="Male predominance in CEAN cohort",
                evidence=["S1|PMC123|0"],
                confidence="high",
                verbatim_support="male predominance",
                relation_type=RelationTypeEnum.demographic,
            )
        ],
        summary_text="",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["S1|PMC123|0"],
            pmcids_referenced=["PMC123"],
            uncited_sentences=[],
        ),
    )
    results = SchemaValidator().validate(summary)
    assert len(results) == 1
    assert ReasonCode.INVALID_CATEGORY not in results[0].reason_codes


# ── Grouping ──────────────────────────────────────────────────────────────────

def test_two_demographic_findings_same_group_id():
    """Same subject/outcome/relation_type/category='demographic' → same group_id."""
    id1 = _group_id("patients", "sex", "demographic", "demographic", pmcid="PMC1")
    id2 = _group_id("patients", "sex", "demographic", "demographic", pmcid="PMC1")
    assert id1 == id2


def test_demographic_vs_morphology_different_group_id():
    """Same subject/outcome/relation_type but different category → different group_id."""
    id_demo = _group_id("patients", "sex", "demographic", "demographic", pmcid="PMC1")
    id_morph = _group_id("patients", "sex", "demographic", "morphology", pmcid="PMC1")
    assert id_demo != id_morph


def test_demographic_prognosis_different_group_id():
    """demographic and prognosis with the same entities produce distinct buckets."""
    id_demo = _group_id("patients", "sex distribution", "demographic", "demographic", pmcid="PMC1")
    id_prog = _group_id("patients", "sex distribution", "demographic", "prognosis", pmcid="PMC1")
    assert id_demo != id_prog


# ── Alias repair (field_validator) ────────────────────────────────────────────

def test_finding_alias_repaired_via_constructor(caplog):
    """Legacy 'demographics' alias is repaired to 'demographic' when constructing Finding directly."""
    with caplog.at_level(logging.WARNING, logger="pipeline.stages.knowledge_extraction.models"):
        f = Finding(
            category="demographics",
            claim="Male predominance",
            evidence=["S1|PMC123|0"],
            confidence="high",
            verbatim_support="male predominance",
        )
    assert f.category == "demographic"
    assert "demographics" in caplog.text


def test_finding_alias_repaired_via_model_validate(caplog):
    """Alias repair fires on model_validate() — the path used by batch/dispatch.py:100."""
    raw_json = json.dumps({
        "chunk_id": "PMC123|0",
        "findings": [
            {
                "category": "demographics",
                "claim": "Male predominance",
                "evidence": ["S1|PMC123|0"],
                "confidence": "high",
                "verbatim_support": "male predominance",
            }
        ],
        "summary_text": "",
        "audit_metadata": {
            "sentences_analyzed": 1,
            "sentences_cited": ["S1|PMC123|0"],
            "pmcids_referenced": ["PMC123"],
            "uncited_sentences": [],
        },
    })
    with caplog.at_level(logging.WARNING, logger="pipeline.stages.knowledge_extraction.models"):
        summary = AuditableSummary.model_validate(json.loads(raw_json))
    assert summary.findings[0].category == "demographic"
    assert "demographics" in caplog.text


def test_finding_category_demographic_with_relation_type_demographic():
    """category='demographic' and relation_type=RelationTypeEnum.demographic agree on spelling."""
    f = Finding(
        category="demographic",
        claim="Slight male predominance",
        evidence=["S1|PMC123|0"],
        confidence="high",
        verbatim_support="slight male predominance",
        relation_type=RelationTypeEnum.demographic,
    )
    assert f.category == "demographic"
    assert f.relation_type == RelationTypeEnum.demographic


def test_alias_repair_does_not_touch_unknown_categories():
    """Unknown categories are not coerced — they pass through and Pydantic rejects them."""
    with pytest.raises(ValidationError):
        Finding(
            category="bogus_category",
            claim="x",
            evidence=["S1|PMC123|0"],
            confidence="high",
            verbatim_support="x",
        )
