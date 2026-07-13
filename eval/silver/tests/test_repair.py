"""Tests for the finding-field repair layer and the singular 'demographic' canonical category."""
from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from eval.silver.generation.generator import _repair_finding_fields
from eval.silver.generation.prompts import EXTRACT_FINDINGS_TOOL
from eval.silver.data.schemas import SilverFinding


# ── _repair_finding_fields ─────────────────────────────────────────────────────

def test_repair_category_demographics_to_demographic(caplog):
    """LLM alias 'demographics' (trailing s) is repaired to canonical 'demographic'."""
    with caplog.at_level(logging.WARNING, logger="eval.silver.generation.generator"):
        repaired = _repair_finding_fields({"category": "demographics", "claim": "test"}, "case1")
    assert repaired["category"] == "demographic"
    assert "demographic" in caplog.text


def test_repair_category_demographic_passthrough():
    """Canonical value 'demographic' passes through unchanged."""
    repaired = _repair_finding_fields({"category": "demographic", "claim": "test"})
    assert repaired["category"] == "demographic"


def test_repair_does_not_touch_valid_category():
    """Valid categories like 'IHC' must not be modified."""
    repaired = _repair_finding_fields({"category": "IHC", "claim": "test"})
    assert repaired["category"] == "IHC"


def test_repair_does_not_touch_unknown_invalid_category():
    """Unknown invalid categories are left unchanged — Pydantic will reject them."""
    repaired = _repair_finding_fields({"category": "bogus_category", "claim": "test"})
    assert repaired["category"] == "bogus_category"


def test_repair_returns_shallow_copy():
    """Repair must not mutate the original dict."""
    original = {"category": "demographic", "claim": "x"}
    _repair_finding_fields(original)
    assert original["category"] == "demographic"


def test_repair_relation_type_demographic_unchanged():
    """relation_type='demographic' is valid (RelationTypeEnum in pipeline depends on it) — not repaired."""
    item = {"category": "demographics", "relation_type": "demographic", "claim": "x"}
    repaired = _repair_finding_fields(item)
    assert repaired["relation_type"] == "demographic"


# ── SilverFinding schema ───────────────────────────────────────────────────────

def test_silver_finding_demographic_valid():
    """category='demographic' passes Pydantic validation."""
    sf = SilverFinding(
        category="demographic",
        claim="Slight male predominance (37 men/29 women)",
        relation_type="demographic",
        # direction="positive" is closest existing value for "more frequent / male predominance"
        # (the prompt documents: positive = present/increased/more frequent)
        direction="positive",
        confidence="high",
        verbatim_support="slight, but probably non-significant, male predominance (37 men/29 women)",
    )
    assert sf.category == "demographic"
    assert sf.relation_type == "demographic"
    assert sf.direction == "positive"


def test_silver_finding_demographics_category_rejected():
    """'demographics' (trailing s) is NOT a valid category — repair layer must be applied first."""
    with pytest.raises(ValidationError):
        SilverFinding(category="demographics", claim="x", confidence="medium")


def test_silver_finding_unknown_category_rejected():
    """Genuinely invalid categories must still be rejected."""
    with pytest.raises(ValidationError):
        SilverFinding(category="bogus_category", claim="x", confidence="medium")


# ── Tool schema ────────────────────────────────────────────────────────────────

def test_tool_schema_includes_demographic():
    """Regression guard: the LLM tool schema must list 'demographic' in the category enum."""
    category_enum = (
        EXTRACT_FINDINGS_TOOL["input_schema"]["properties"]["findings"]
        ["items"]["properties"]["category"]["enum"]
    )
    assert "demographic" in category_enum


def test_tool_schema_excludes_demographics_plural():
    """The tool schema must NOT expose the plural alias 'demographics' as a valid category."""
    category_enum = (
        EXTRACT_FINDINGS_TOOL["input_schema"]["properties"]["findings"]
        ["items"]["properties"]["category"]["enum"]
    )
    assert "demographics" not in category_enum


# ── End-to-end: the bug-report example no longer gets dropped ─────────────────

def test_example_finding_survives_repair_and_validation():
    """The malformed example from the bug report parses correctly after repair.

    Opus occasionally emits the plural 'demographics'; the repair layer maps it to
    the canonical singular 'demographic' before Pydantic validation.
    """
    raw = {
        "category": "demographics",
        "claim": "Slight male predominance (37 men/29 women), probably non-significant",
        "subject_entity": "CEAN patients",
        "outcome_entity": "sex distribution",
        "relation_type": "demographic",
        "direction": "positive",
        "confidence": "high",
        "verbatim_support": "slight, but probably non-significant, male predominance (37 men/29 women)",
    }
    repaired = _repair_finding_fields(raw, "test_case")
    sf = SilverFinding(**{k: v for k, v in repaired.items() if k in SilverFinding.model_fields})
    assert sf.category == "demographic"
    assert sf.relation_type == "demographic"  # kept: pipeline RelationTypeEnum.demographic
    assert sf.direction == "positive"
    assert "male predominance" in sf.claim
