"""Regression test for MAP_PROMPT_AUDIT Issue 5.

`FindingScope.scope_parsed` is no longer LLM-controlled; an after-validator
recomputes it from the other sub-fields. The LLM's emitted value is ignored.
"""
from __future__ import annotations

from nlp_histo.pipeline.stages.knowledge_extraction.models import FindingScope


def test_empty_scope_has_scope_parsed_false() -> None:
    s = FindingScope.empty()
    assert s.scope_parsed is False


def test_any_non_null_subfield_flips_scope_parsed_true() -> None:
    s = FindingScope(
        disease_subtype="DLBCL",
        cohort_n=None,
        assay_method=None,
        biomarker_cutoff=None,
        tissue_site=None,
        treatment_context=None,
        endpoint=None,
        study_design=None,
        scope_parsed=False,  # LLM's wrong answer — should be overridden
    )
    assert s.scope_parsed is True


def test_llm_emitted_true_on_all_null_overridden_to_false() -> None:
    s = FindingScope(
        disease_subtype=None,
        cohort_n=None,
        assay_method=None,
        biomarker_cutoff=None,
        tissue_site=None,
        treatment_context=None,
        endpoint=None,
        study_design=None,
        scope_parsed=True,  # LLM hallucinated — should be overridden
    )
    assert s.scope_parsed is False


def test_cohort_n_zero_counts_as_provided() -> None:
    # cohort_n=0 is non-null; "the LLM said n=0" still means a sub-field was parsed.
    s = FindingScope(
        disease_subtype=None,
        cohort_n=0,
        assay_method=None,
        biomarker_cutoff=None,
        tissue_site=None,
        treatment_context=None,
        endpoint=None,
        study_design=None,
        scope_parsed=False,
    )
    assert s.scope_parsed is True
