"""Regression test for B-021.

Both helpers in `canonicalize_stage.py` previously excluded the dead string
`"None"` from the polarity-bearing set instead of the real `DirectionEnum`
value `"no_direction"`. That made every `no_direction` finding:

* leak into its own CanonicalRule bin via `_split_by_direction`, and
* count toward `is_conflicted=True` in `_compute_scope_fields` whenever a
  real-polarity finding was also present.

The fix replaces `"None"` with `"no_direction"` in both call sites.

No LLM, NLI, UMLS, or DB dependencies — pure helper-function tests.
"""
from __future__ import annotations

from pipeline.stages.summarization.current_stages.canonicalize_stage import (
    _compute_scope_fields,
    _split_by_direction,
)
from pipeline.stages.summarization.models import (
    DirectionEnum,
    FindingGroup,
    FindingScope,
    NormalFinding,
    RelationTypeEnum,
    SourceSpan,
)


def _nf(normal_id: str, direction: DirectionEnum | None, pmcid: str = "PMC1") -> NormalFinding:
    return NormalFinding(
        normal_id=normal_id,
        subject_entity="CD30",
        outcome_entity="expression",
        relation_type=RelationTypeEnum.expression,
        direction=direction,
        category="IHC",
        predicate_text="CD30 expression",
        scope=FindingScope.empty(),
        source_finding_ids=[],
        evidence=[SourceSpan(sentence_id="S1", pmcid=pmcid, text_element_id=1, verbatim="CD30 stains")],
        pmcids=[pmcid],
        mean_grounding_score=0.9,
    )


def _group(direction_counts: dict[str, int]) -> FindingGroup:
    return FindingGroup(
        group_id="GRP_test",
        subject_entity="CD30",
        outcome_entity="expression",
        relation_type=RelationTypeEnum.expression,
        category="IHC",
        member_ids=[],
        direction_counts=direction_counts,
        scope_heterogeneity=0.0,
    )


# ── _compute_scope_fields ─────────────────────────────────────────────────────

def test_no_direction_alone_is_not_conflicted():
    """A bin of only no_direction findings must not trip is_conflicted."""
    members = [_nf("a", DirectionEnum.no_direction), _nf("b", DirectionEnum.no_direction)]
    is_conflicted, _ = _compute_scope_fields(members)
    assert is_conflicted is False


def test_no_direction_plus_positive_is_not_conflicted():
    """Pre-fix this returned True because `no_direction` was counted as a
    distinct polarity alongside `positive`. Post-fix only one polarity-bearing
    direction is present, so is_conflicted stays False."""
    members = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.no_direction)]
    is_conflicted, _ = _compute_scope_fields(members)
    assert is_conflicted is False


def test_unclear_plus_positive_is_not_conflicted():
    """Existing `unclear` handling preserved."""
    members = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.unclear)]
    is_conflicted, _ = _compute_scope_fields(members)
    assert is_conflicted is False


def test_positive_plus_negative_is_conflicted():
    """Real-polarity contradiction still flips is_conflicted=True."""
    members = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.negative)]
    is_conflicted, _ = _compute_scope_fields(members)
    assert is_conflicted is True


def test_study_coverage_multi_study():
    members = [_nf("a", DirectionEnum.positive, "PMC1"), _nf("b", DirectionEnum.positive, "PMC2")]
    _, coverage = _compute_scope_fields(members)
    assert coverage == "multi_study"


# ── _split_by_direction ───────────────────────────────────────────────────────

def test_split_no_direction_only_collapses_to_unclear_bin():
    """All-no_direction group must fall to the single 'unclear' bin path —
    not produce a CanonicalRule with direction='no_direction'."""
    group = _group({"no_direction": 2})
    members = [_nf("a", DirectionEnum.no_direction), _nf("b", DirectionEnum.no_direction)]
    bins = _split_by_direction(group, members)
    assert bins == [("unclear", members)]


def test_split_no_direction_joins_single_polarity_bin():
    """Pre-fix this produced two bins (positive + no_direction). Post-fix the
    no_direction finding joins the positive bin (member_nfs unchanged in
    single-direction collapse, see implementation)."""
    group = _group({"positive": 1, "no_direction": 1})
    nf_pos = _nf("pos", DirectionEnum.positive)
    nf_nd = _nf("nd", DirectionEnum.no_direction)
    bins = _split_by_direction(group, [nf_pos, nf_nd])
    assert len(bins) == 1
    direction, bin_members = bins[0]
    assert direction == "positive"
    assert {m.normal_id for m in bin_members} == {"pos", "nd"}


def test_split_no_direction_attaches_to_largest_polarity_in_mixed_group():
    """Mixed positive/negative + a no_direction finding: the no_direction
    must attach to the largest polarity bin, not create a third."""
    group = _group({"positive": 2, "negative": 1, "no_direction": 1})
    nf_pos1 = _nf("p1", DirectionEnum.positive)
    nf_pos2 = _nf("p2", DirectionEnum.positive)
    nf_neg = _nf("n", DirectionEnum.negative)
    nf_nd = _nf("nd", DirectionEnum.no_direction)
    bins = dict(_split_by_direction(group, [nf_pos1, nf_pos2, nf_neg, nf_nd]))
    assert set(bins.keys()) == {"positive", "negative"}
    assert {m.normal_id for m in bins["positive"]} == {"p1", "p2", "nd"}
    assert {m.normal_id for m in bins["negative"]} == {"n"}


def test_split_real_polarity_split_preserved():
    """Two real polarities still produce two bins — guardrail against an
    over-aggressive fix that would collapse genuine contradictions."""
    group = _group({"positive": 1, "negative": 1})
    nf_pos = _nf("p", DirectionEnum.positive)
    nf_neg = _nf("n", DirectionEnum.negative)
    bins = dict(_split_by_direction(group, [nf_pos, nf_neg]))
    assert set(bins.keys()) == {"positive", "negative"}
