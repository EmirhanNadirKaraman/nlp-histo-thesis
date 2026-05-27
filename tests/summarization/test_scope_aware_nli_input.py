"""Tests for the scope-aware NLI input builder in the RELATE stage.

EXP B.2 / `docs/EXP_B2_RESULTS.md` highlighted that the pairwise NLI was
operating on impoverished input — `predicate_text` only, with scope
(disease_subtype, tissue_site, assay_method, …) stripped. That dropped
SCOPE_QUALIFY signal: a "TP53 absent in AciCCIS" vs "TP53 present in AciCC"
pair was getting CONTRADICT when the right label is SCOPE_QUALIFY (different
tumour subtypes).

Option 1 from the follow-up plan: prepend a scope-tag prefix to each rule's
NLI text. Default ON via `RelateConfig.scope_aware_nli=True`. These tests
lock the behaviour without touching the NLI model itself.
"""
from __future__ import annotations

import pytest

from pipeline.stages.summarization.current_stages.relate_stage import (
    _build_nli_text,
)
from pipeline.stages.summarization.models import (
    CanonicalRule,
    DirectionEnum,
    FindingScope,
    RelationTypeEnum,
)


def _rule(
    *,
    predicate: str,
    scope: FindingScope | None,
    subject: str = "TP53",
    outcome: str = "Mutation",
    direction: DirectionEnum | None = DirectionEnum.absent,
) -> CanonicalRule:
    """Build a minimal CanonicalRule for testing the NLI text builder."""
    return CanonicalRule(
        canonical_id="CR_test",
        group_id="G_test",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=RelationTypeEnum.expression,
        direction=direction,
        predicate_text=predicate,
        is_conflicted=False,
        study_coverage="single_study",
        category="molecular_genetics",
        supporting_pmcids=["PMC1"],
        member_normal_ids=["NF_1"],
        mean_grounding_score=0.9,
        finding_count=1,
        scope=scope,
    )


def _scope(**overrides) -> FindingScope:
    """Build a FindingScope with all fields nulled out unless overridden."""
    defaults = dict(
        disease_subtype=None,
        cohort_n=None,
        assay_method=None,
        biomarker_cutoff=None,
        tissue_site=None,
        treatment_context=None,
        endpoint=None,
        study_design=None,
        scope_parsed=True,
    )
    defaults.update(overrides)
    return FindingScope(**defaults)


# ── Legacy / opt-out path ────────────────────────────────────────────────────


def test_scope_aware_false_returns_bare_predicate():
    """`scope_aware=False` reverts to legacy: predicate_text only, no prefix."""
    rule = _rule(
        predicate="TP53 mutations absent in AciCCIS",
        scope=_scope(disease_subtype="AciCCIS", tissue_site="breast"),
    )
    text = _build_nli_text(rule, scope_aware=False)
    assert text == "TP53 mutations absent in AciCCIS"
    assert "[scope:" not in text


def test_none_scope_falls_back_to_bare_predicate():
    """Backward compat: rules from older artifacts (scope=None) skip the prefix."""
    rule = _rule(predicate="plain claim", scope=None)
    text = _build_nli_text(rule, scope_aware=True)
    assert text == "plain claim"


def test_empty_scope_fields_skip_prefix():
    """All-None scope → no prefix added (avoids empty `[scope: ]` noise)."""
    rule = _rule(predicate="plain claim", scope=_scope())  # all fields None
    text = _build_nli_text(rule, scope_aware=True)
    assert text == "plain claim"
    assert "[scope:" not in text


# ── Enriched path ────────────────────────────────────────────────────────────


def test_scope_aware_prepends_disease_subtype_tag():
    """The headline use case: disease_subtype lands in the NLI text."""
    rule = _rule(
        predicate="TP53 mutations absent",
        scope=_scope(disease_subtype="AciCCIS"),
    )
    text = _build_nli_text(rule, scope_aware=True)
    assert text.startswith("[scope: disease=AciCCIS]")
    assert text.endswith("TP53 mutations absent")


def test_scope_aware_includes_multiple_non_null_fields():
    """Multi-field scope produces a `|`-joined prefix in a stable order."""
    rule = _rule(
        predicate="TP53 mutations absent",
        scope=_scope(disease_subtype="AciCCIS", tissue_site="breast", assay_method="IHC"),
    )
    text = _build_nli_text(rule, scope_aware=True)
    # Pipe-joined, in the order the helper enumerates fields.
    assert text.startswith("[scope: disease=AciCCIS | tissue=breast | assay=IHC]")


def test_scope_aware_skips_null_fields_between_set_ones():
    """Null fields don't introduce trailing/empty separators in the prefix."""
    rule = _rule(
        predicate="claim",
        # Set disease + assay only — tissue stays None and must not appear.
        scope=_scope(disease_subtype="DLBCL", assay_method="flow_cytometry"),
    )
    text = _build_nli_text(rule, scope_aware=True)
    assert text == "[scope: disease=DLBCL | assay=flow_cytometry] claim"
    assert "tissue" not in text


def test_two_rules_differing_only_on_scope_produce_different_texts():
    """The B.2 motivating case: pairs that disagree only on disease subtype.

    Pre-fix, predicate_text would have been identical or near-identical and
    NLI saw effectively the same text on both sides → false CONTRADICT.
    With the scope prefix, the disease difference is now in the input.
    """
    rule_a = _rule(
        predicate="TP53 mutations not detected",
        scope=_scope(disease_subtype="AciCCIS"),
    )
    rule_b = _rule(
        predicate="TP53 mutations not detected",
        scope=_scope(disease_subtype="AciCC"),
    )
    text_a = _build_nli_text(rule_a, scope_aware=True)
    text_b = _build_nli_text(rule_b, scope_aware=True)
    assert text_a != text_b, (
        f"Scope-aware NLI must surface disease subtype divergence; "
        f"got identical texts:\n  a: {text_a!r}\n  b: {text_b!r}"
    )
    assert "disease=AciCCIS" in text_a
    assert "disease=AciCC" in text_b


def test_scope_aware_flag_toggles_off_at_runtime():
    """`RelateStage(scope_aware_nli=False)` must round-trip to `_build_nli_text`.

    Smoke check that the constructor stashes the flag for `relate()` to use.
    The real run plumbs `cfg.relate.scope_aware_nli` through here.
    """
    from pipeline.stages.summarization.current_stages.relate_stage import RelateStage
    s_on  = RelateStage(scope_aware_nli=True)
    s_off = RelateStage(scope_aware_nli=False)
    assert s_on._scope_aware_nli is True
    assert s_off._scope_aware_nli is False
