"""Regression test for B-049 cross-paper non-polarity gate.

`_should_compare_cross_paper` in `stages/corpus_relate.py` must skip any
pair where either side carries `direction in {unclear, no_direction}`.

Direction can arrive as a `DirectionEnum`, a raw string (from
metadata dicts loaded back from cached JSON), or `None` (legacy rows).
The `direction_value` normalizer (added in B-049 / S1) handles all three.
"""
from __future__ import annotations

import pytest

from pipeline.stages.knowledge_extraction.stages.corpus_relate import (
    _should_compare_cross_paper,
)
from pipeline.stages.knowledge_extraction.models import (
    CanonicalRule,
    DirectionEnum,
    RelationTypeEnum,
)


def _rule(canonical_id: str, direction) -> CanonicalRule:
    """Minimal CanonicalRule with explicit direction."""
    return CanonicalRule(
        canonical_id=canonical_id,
        group_id=f"GRP_{canonical_id}",
        subject_entity="CD30",
        outcome_entity="OS",
        relation_type=RelationTypeEnum.prognostic,
        direction=direction,
        predicate_text="CD30 predicts OS",
        is_conflicted=False,
        study_coverage="single_study",
        category="IHC",
        supporting_pmcids=["PMC1"],
        member_normal_ids=["NF_x"],
        mean_grounding_score=0.8,
        finding_count=1,
    )


@pytest.mark.parametrize("dir_a, dir_b", [
    (DirectionEnum.unclear,      DirectionEnum.positive),
    (DirectionEnum.no_direction, DirectionEnum.negative),
    (DirectionEnum.positive,     DirectionEnum.unclear),
    (DirectionEnum.unclear,      DirectionEnum.unclear),
])
def test_non_polarity_pair_skipped(dir_a, dir_b):
    a = _rule("CR_a", dir_a)
    b = _rule("CR_b", dir_b)
    ok, reason = _should_compare_cross_paper(a, b)
    assert ok is False
    assert reason == "non_polarity_direction"


def test_polarity_only_pair_passes_non_polarity_gate():
    """Sanity check: two polarity-bearing directions don't get caught by the
    non-polarity skip (they may still fail later gates — that's fine)."""
    a = _rule("CR_a", DirectionEnum.positive)
    b = _rule("CR_b", DirectionEnum.negative)
    ok, reason = _should_compare_cross_paper(a, b)
    # Should NOT be rejected with non_polarity_direction. If rejected at all,
    # it would be by category / subject / outcome — none of which differ here.
    assert reason != "non_polarity_direction"
    assert ok is True


def test_direction_none_treated_as_non_polarity():
    """Legacy rows can have direction=None after JSON roundtrip. The
    S1 normalizer maps None → 'unclear' so the pair is correctly skipped.
    Without the normalizer this test would fail (None in {DirectionEnum.unclear}
    is False, so the naive check would let the pair through)."""
    a = _rule("CR_a", None)
    b = _rule("CR_b", DirectionEnum.positive)
    ok, reason = _should_compare_cross_paper(a, b)
    assert ok is False
    assert reason == "non_polarity_direction"
