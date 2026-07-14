"""Regression tests for B-049 — per-direction CanonicalRule binning.

The old B-021 folding tests are obsolete: B-049 supersedes the "fold unclear
into majority" policy. Every observed direction now gets its own bin.

Pure helper-function tests — no LLM, NLI, UMLS, or DB dependencies.

Tested invariants:
* `_split_by_direction` emits one bin per observed direction (sorted order).
* `direction is None` normalises to the `"unclear"` bin (S1 normalizer).
* `_study_coverage` returns the right label by PMCID count.
* Group-level `is_conflicted` (set on every rule from the group) is True iff
  the group emits ≥2 polarity-bearing direction bins.
* **S5 core invariant (B-049 honesty fix):** no polarity-bearing rule's
  `member_normal_ids` contains a NormalFinding whose source direction was
  `unclear` or `no_direction`.
* 2 positive + 2 negative + 1 unclear (the B-026 reproducibility case) ships
  deterministic content across re-runs.
"""
from __future__ import annotations

from pipeline.stages.knowledge_extraction.stages.canonicalize_stage import (
    CanonicalizeStage,
    _split_by_direction,
    _study_coverage,
)
from pipeline.stages.knowledge_extraction.models import (
    DirectionEnum,
    FindingGroup,
    FindingScope,
    NormalFinding,
    POLARITY_BEARING_DIRS,
    RelationTypeEnum,
    SourceSpan,
    direction_value,
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


def _group(member_ids: list[str], direction_counts: dict[str, int]) -> FindingGroup:
    return FindingGroup(
        group_id="GRP_test",
        subject_entity="CD30",
        outcome_entity="expression",
        relation_type=RelationTypeEnum.expression,
        category="IHC",
        member_ids=member_ids,
        direction_counts=direction_counts,
        scope_heterogeneity=0.0,
    )


def _canonicalize(nfs: list[NormalFinding]):
    group = _group(
        member_ids=[nf.normal_id for nf in nfs],
        direction_counts={
            d: sum(1 for nf in nfs if direction_value(nf.direction) == d)
            for d in {direction_value(nf.direction) for nf in nfs}
        },
    )
    return CanonicalizeStage().canonicalize(
        [group], {nf.normal_id: nf for nf in nfs}, pmcid="PMC1"
    )


# ── _split_by_direction ───────────────────────────────────────────────────────

def test_split_one_bin_per_observed_direction():
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.negative)]
    bins = dict(_split_by_direction(nfs))
    assert set(bins.keys()) == {"positive", "negative"}
    assert [m.normal_id for m in bins["positive"]] == ["a"]
    assert [m.normal_id for m in bins["negative"]] == ["b"]


def test_split_unclear_gets_own_bin_no_folding():
    """B-049 core change: unclear no longer folds into a polarity bin."""
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.unclear)]
    bins = dict(_split_by_direction(nfs))
    assert set(bins.keys()) == {"positive", "unclear"}
    assert [m.normal_id for m in bins["positive"]] == ["a"]
    assert [m.normal_id for m in bins["unclear"]] == ["b"]


def test_split_no_direction_gets_own_bin():
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.no_direction)]
    bins = dict(_split_by_direction(nfs))
    assert set(bins.keys()) == {"positive", "no_direction"}


def test_split_only_no_direction_yields_single_bin():
    nfs = [_nf("a", DirectionEnum.no_direction), _nf("b", DirectionEnum.no_direction)]
    bins = _split_by_direction(nfs)
    assert len(bins) == 1
    assert bins[0][0] == "no_direction"


def test_split_none_direction_normalises_to_unclear():
    """S1 normalizer: a NormalFinding with direction=None lands in 'unclear'."""
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", None)]
    bins = dict(_split_by_direction(nfs))
    assert set(bins.keys()) == {"positive", "unclear"}
    assert [m.normal_id for m in bins["unclear"]] == ["b"]


def test_split_returns_sorted_order():
    """Determinism (supersedes B-026 — no max() tiebreak needed since no folding)."""
    nfs = [
        _nf("a", DirectionEnum.positive),
        _nf("b", DirectionEnum.negative),
        _nf("c", DirectionEnum.unclear),
    ]
    bins = _split_by_direction(nfs)
    keys = [k for k, _ in bins]
    assert keys == sorted(keys)


# ── _study_coverage ───────────────────────────────────────────────────────────

def test_study_coverage_single():
    nfs = [_nf("a", DirectionEnum.positive, "PMC1"), _nf("b", DirectionEnum.positive, "PMC1")]
    assert _study_coverage(nfs) == "single_study"


def test_study_coverage_multi():
    nfs = [_nf("a", DirectionEnum.positive, "PMC1"), _nf("b", DirectionEnum.positive, "PMC2")]
    assert _study_coverage(nfs) == "multi_study"


# ── Group-level is_conflicted (B-049 repurposing) ─────────────────────────────

def test_is_conflicted_false_for_single_polarity_group():
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.positive)]
    rules = _canonicalize(nfs)
    assert all(r.is_conflicted is False for r in rules)


def test_is_conflicted_false_for_polarity_plus_unclear():
    """polarity_count=1 → group not conflicted, even though the group has two bins."""
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.unclear)]
    rules = _canonicalize(nfs)
    assert {direction_value(r.direction) for r in rules} == {"positive", "unclear"}
    assert all(r.is_conflicted is False for r in rules)


def test_is_conflicted_false_for_polarity_plus_no_direction():
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.no_direction)]
    rules = _canonicalize(nfs)
    assert {direction_value(r.direction) for r in rules} == {"positive", "no_direction"}
    assert all(r.is_conflicted is False for r in rules)


def test_is_conflicted_false_for_only_no_direction():
    nfs = [_nf("a", DirectionEnum.no_direction), _nf("b", DirectionEnum.no_direction)]
    rules = _canonicalize(nfs)
    assert len(rules) == 1
    assert direction_value(rules[0].direction) == "no_direction"
    assert rules[0].is_conflicted is False


def test_is_conflicted_true_for_two_polarity_bins():
    nfs = [_nf("a", DirectionEnum.positive), _nf("b", DirectionEnum.negative)]
    rules = _canonicalize(nfs)
    assert all(r.is_conflicted is True for r in rules)


def test_is_conflicted_true_applies_to_every_rule_from_a_conflicted_group():
    """Even the unclear rule from a conflicted group carries is_conflicted=True.
    The flag is a property of the parent group, not the individual bin."""
    nfs = [
        _nf("a", DirectionEnum.positive),
        _nf("b", DirectionEnum.negative),
        _nf("c", DirectionEnum.unclear),
    ]
    rules = _canonicalize(nfs)
    assert {direction_value(r.direction) for r in rules} == {"positive", "negative", "unclear"}
    assert all(r.is_conflicted is True for r in rules)


# ── S5 core invariant — protects the B-049 honesty fix ────────────────────────

def test_no_unclear_leakage_into_polarity_bins():
    """B-049 honesty invariant: no polarity-bearing CanonicalRule may contain
    member_normal_ids whose source NormalFinding direction was unclear or
    no_direction. This is the test that protects the actual behavioural change.

    Uses the B-026 reproducibility scenario: 2 positive + 2 negative + 1 unclear.
    """
    nfs = [
        _nf("p1", DirectionEnum.positive),
        _nf("p2", DirectionEnum.positive),
        _nf("n1", DirectionEnum.negative),
        _nf("n2", DirectionEnum.negative),
        _nf("u",  DirectionEnum.unclear),
    ]
    rules = _canonicalize(nfs)
    by_direction = {direction_value(r.direction): r for r in rules}

    assert set(by_direction) == {"positive", "negative", "unclear"}
    assert len(by_direction["positive"].member_normal_ids) == 2
    assert len(by_direction["negative"].member_normal_ids) == 2
    assert len(by_direction["unclear"].member_normal_ids) == 1

    nf_by_id = {nf.normal_id: nf for nf in nfs}
    for r in rules:
        rule_dir = direction_value(r.direction)
        if rule_dir in POLARITY_BEARING_DIRS:
            for mid in r.member_normal_ids:
                src_dir = direction_value(nf_by_id[mid].direction)
                assert src_dir == rule_dir, (
                    f"polarity rule {rule_dir!r} contains member {mid!r} "
                    f"sourced from {src_dir!r} — leakage forbidden by B-049"
                )


def test_b026_determinism_2pos_2neg_1unclear():
    """The B-026 case: 2 positive + 2 negative + 1 unclear. Pre-fix the
    `max(non_unclear, key=len)` tie-break picked a bin to attach the unclear
    to in member-arrival order. Post-fix unclear gets its own bin and rule
    composition is identical across re-runs.
    """
    nfs = [
        _nf("p1", DirectionEnum.positive),
        _nf("p2", DirectionEnum.positive),
        _nf("n1", DirectionEnum.negative),
        _nf("n2", DirectionEnum.negative),
        _nf("u",  DirectionEnum.unclear),
    ]
    rules_a = _canonicalize(nfs)
    rules_b = _canonicalize(list(reversed(nfs)))  # different arrival order

    def _key(rs):
        return sorted(
            (direction_value(r.direction), tuple(sorted(r.member_normal_ids)))
            for r in rs
        )
    assert _key(rules_a) == _key(rules_b)
