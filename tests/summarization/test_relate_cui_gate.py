"""Patch 1 — RELATE CUI/synonym-aware gate regression tests.

Confirms that _should_compare:
  * uses CUI equality on subject when both rules have subject_cui
  * uses CUI equality on outcome when both rules have outcome_cui
  * falls back to synonym-aware string normalization when CUIs are absent
  * still rejects truly different outcomes

Pre-fix behaviour (for reference): subject was synonym-aware, outcome was raw
strip().lower(), CUIs were ignored entirely. A pair with matching outcome_cui
but divergent surface strings was rejected at the gate with outcome_incompatible.

The autouse `_disable_umls` fixture prevents NORMALIZE's `normalize_entity`
import (used by _norm_outcome / _norm_subject) from loading the UMLS KB.
"""
from __future__ import annotations

import pytest

from pipeline.stages.summarization.current_stages.relate_stage import _should_compare
from pipeline.stages.summarization.models import (
    CanonicalRule,
    DirectionEnum,
    RelationTypeEnum,
)


@pytest.fixture(autouse=True)
def _disable_umls(monkeypatch):
    from pipeline.stages.summarization import umls_resources
    from pipeline.stages.summarization.current_stages import normalize_stage

    monkeypatch.setenv("NLP_HISTO_DISABLE_UMLS", "1")
    umls_resources._reset_for_tests()
    normalize_stage._UMLS_CACHE.clear()
    yield
    umls_resources._reset_for_tests()
    normalize_stage._UMLS_CACHE.clear()


def _rule(
    *,
    canonical_id: str,
    subject_entity: str = "CD30",
    outcome_entity: str = "OS",
    subject_cui: str | None = None,
    outcome_cui: str | None = None,
    category: str = "IHC",
    relation_type: RelationTypeEnum = RelationTypeEnum.prognostic,
    direction: DirectionEnum | None = DirectionEnum.positive,
) -> CanonicalRule:
    return CanonicalRule(
        canonical_id=canonical_id,
        group_id=f"GRP_{canonical_id}",
        subject_entity=subject_entity,
        outcome_entity=outcome_entity,
        relation_type=relation_type,
        direction=direction,
        predicate_text="placeholder predicate text",
        is_conflicted=False,
        study_coverage="single_study",
        category=category,
        supporting_pmcids=["PMC1"],
        member_normal_ids=["NF_1"],
        mean_grounding_score=0.8,
        finding_count=1,
        subject_cui=subject_cui,
        outcome_cui=outcome_cui,
    )


def test_same_outcome_cui_different_surface_strings_comparable():
    a = _rule(canonical_id="A", outcome_entity="MGA",                outcome_cui="C9999")
    b = _rule(canonical_id="B", outcome_entity="microglandular adenosis", outcome_cui="C9999")
    eligible, reason = _should_compare(a, b)
    assert eligible is True, f"expected eligible, got reason={reason!r}"


def test_different_outcome_cui_skipped():
    a = _rule(canonical_id="A", outcome_entity="OS", outcome_cui="C0001")
    b = _rule(canonical_id="B", outcome_entity="OS", outcome_cui="C0002")
    eligible, reason = _should_compare(a, b)
    assert eligible is False
    assert reason == "outcome_cui_incompatible"


def test_same_subject_cui_different_surface_strings_comparable():
    a = _rule(canonical_id="A", subject_entity="ALK-positive ALCL", subject_cui="C7777")
    b = _rule(canonical_id="B", subject_entity="ALK+ ALCL",          subject_cui="C7777")
    eligible, reason = _should_compare(a, b)
    assert eligible is True, f"expected eligible, got reason={reason!r}"


def test_different_subject_cui_skipped():
    a = _rule(canonical_id="A", subject_cui="C0001")
    b = _rule(canonical_id="B", subject_cui="C0002")
    eligible, reason = _should_compare(a, b)
    assert eligible is False
    assert reason == "subject_cui_mismatch"


def test_synonym_outcome_no_cuis_comparable():
    """No CUIs on either side — outcome normalizer must use the synonym dict.

    "Overall Survival" and "overall survival" both normalize to "OS" via
    synonyms.yaml; pre-fix _norm_outcome was raw strip().lower() and would
    have produced "overall survival" vs "os" — incompatible.
    """
    a = _rule(canonical_id="A", outcome_entity="Overall Survival")
    b = _rule(canonical_id="B", outcome_entity="overall survival")
    eligible, reason = _should_compare(a, b)
    assert eligible is True, f"expected eligible, got reason={reason!r}"


def test_truly_different_outcomes_skipped():
    a = _rule(canonical_id="A", outcome_entity="OS")
    b = _rule(canonical_id="B", outcome_entity="PFS")
    eligible, reason = _should_compare(a, b)
    assert eligible is False
    assert reason == "outcome_incompatible"


def test_corpus_relate_gate_unchanged():
    """Patch 1 must not alter helpers/corpus_relate._should_compare_cross_paper —
    the cross-paper gate was already CUI-aware. This smoke test imports it and
    checks the same two paired cases still resolve identically.
    """
    from pipeline.stages.summarization.helpers.corpus_relate import (
        _should_compare_cross_paper,
    )
    a_match = _rule(canonical_id="A", outcome_entity="MGA",                outcome_cui="C9999")
    b_match = _rule(canonical_id="B", outcome_entity="microglandular adenosis", outcome_cui="C9999")
    ok, _ = _should_compare_cross_paper(a_match, b_match)
    assert ok is True

    a_diff = _rule(canonical_id="A", outcome_cui="C0001")
    b_diff = _rule(canonical_id="B", outcome_cui="C0002")
    not_ok, reason = _should_compare_cross_paper(a_diff, b_diff)
    assert not_ok is False
    assert reason == "outcome_cui_incompatible"
