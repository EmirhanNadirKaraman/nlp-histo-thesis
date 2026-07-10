"""Patch 4 — NORMALIZE dedup includes category.

Confirms that two findings differing only on category produce two
NormalFindings rather than collapsing into one (with the discarded category
silently lost). Also confirms that the dedup still merges identical findings
within the same category — the patch narrows the dedup, doesn't break it.
"""
from __future__ import annotations

import pytest

from pipeline.stages.summarization.stages.normalize_stage import NormalizeStage
from pipeline.stages.summarization.models import (
    DirectionEnum,
    Finding,
    FindingScope,
    RelationTypeEnum,
)


PMCID = "PMC12345"


@pytest.fixture(autouse=True)
def _disable_umls(monkeypatch):
    from pipeline.stages.summarization import umls_resources
    from pipeline.stages.summarization.stages import normalize_stage

    monkeypatch.setenv("NLP_HISTO_DISABLE_UMLS", "1")
    umls_resources._reset_for_tests()
    normalize_stage._UMLS_CACHE.clear()
    yield
    umls_resources._reset_for_tests()
    normalize_stage._UMLS_CACHE.clear()


def _finding(*, category: str, claim: str = "CD30 predicts worse OS") -> Finding:
    return Finding(
        category=category,
        claim=claim,
        evidence=[f"S1|{PMCID}|101"],
        confidence="high",
        verbatim_support="CD30 was associated with worse OS",
        subject_entity="CD30",
        outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=DirectionEnum.negative,
        scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        grounding_score=0.8,
    )


def test_different_category_kept_separate():
    """Two findings with identical (te_id, subject, outcome, relation, direction)
    but different category must produce two NormalFindings.

    Pre-fix: _dedup_key omitted category → both collapse into one, with
    rep.category (best-grounded member) silently winning.
    """
    a = _finding(category="IHC")
    b = _finding(category="prognosis")
    out = NormalizeStage().normalize([a, b], PMCID)
    categories = {nf.category for nf in out}
    assert "IHC" in categories
    assert "prognosis" in categories
    assert len(out) == 2


def test_same_category_still_merges():
    """The fix narrows the dedup; same-category dupes still merge."""
    a = _finding(category="IHC")
    b = _finding(category="IHC")
    out = NormalizeStage().normalize([a, b], PMCID)
    assert len(out) == 1
    assert out[0].category == "IHC"
