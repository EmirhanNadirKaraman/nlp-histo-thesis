"""
Phase 3 smoke tests — GroupStage and is_groupable predicate.
No LLM or NLI model required.
"""
import pytest

from pipeline.stages.knowledge_extraction.models import (
    DirectionEnum,
    FindingGroup,
    FindingScope,
    NormalFinding,
    RelationTypeEnum,
    SourceSpan,
)
from pipeline.stages.knowledge_extraction.stages.group_stage import (
    GroupStage,
    is_groupable,
    _group_id,
)

PMCID = "PMC12345"


def _nf(
    normal_id: str = "NF_1",
    subject: str | None = "CD30",
    outcome: str | None = "OS",
    relation_type: RelationTypeEnum = RelationTypeEnum.prognostic,
    direction: DirectionEnum | None = DirectionEnum.negative,
    category: str = "IHC",
    scope: FindingScope | None = None,
) -> NormalFinding:
    return NormalFinding(
        normal_id=normal_id,
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=relation_type,
        direction=direction,
        category=category,
        predicate_text="CD30 predicts worse OS",
        scope=scope or FindingScope(),
        source_finding_ids=[],
        evidence=[SourceSpan(sentence_id="S1", pmcid=PMCID, text_element_id=101, verbatim="x")],
        pmcids=[PMCID],
        mean_grounding_score=0.8,
    )


# ── is_groupable ──────────────────────────────────────────────────────────────

def test_groupable_all_fields_present():
    assert is_groupable(_nf()) is True


def test_not_groupable_none_subject():
    assert is_groupable(_nf(subject=None)) is False


def test_not_groupable_none_outcome():
    assert is_groupable(_nf(outcome=None)) is False


def test_not_groupable_unclear_relation_type():
    assert is_groupable(_nf(relation_type=RelationTypeEnum.unclear)) is False


def test_not_groupable_all_none():
    assert is_groupable(_nf(subject=None, outcome=None, relation_type=RelationTypeEnum.unclear)) is False


def test_groupable_with_none_direction():
    """direction is NOT part of the groupability invariant — None direction is fine."""
    assert is_groupable(_nf(direction=None)) is True


# ── GroupStage.group — basic ──────────────────────────────────────────────────

def test_empty_input_returns_empty():
    assert GroupStage().group([], PMCID) == []


def test_single_groupable_finding():
    groups = GroupStage().group([_nf()], PMCID)
    assert len(groups) == 1
    assert groups[0].member_ids == ["NF_1"]
    assert groups[0].subject_entity == "CD30"
    assert groups[0].outcome_entity == "OS"


def test_ungroupable_finding_raises():
    with pytest.raises(ValueError, match="Non-groupable"):
        GroupStage().group([_nf(subject=None)], PMCID)


def test_ungroupable_unclear_relation_type_raises():
    with pytest.raises(ValueError, match="Non-groupable"):
        GroupStage().group([_nf(relation_type=RelationTypeEnum.unclear)], PMCID)


# ── Grouping key ──────────────────────────────────────────────────────────────

def test_same_entity_pair_same_group():
    nfs = [_nf(normal_id="NF_1"), _nf(normal_id="NF_2")]
    groups = GroupStage().group(nfs, PMCID)
    assert len(groups) == 1
    assert set(groups[0].member_ids) == {"NF_1", "NF_2"}


def test_different_outcome_different_group():
    nfs = [
        _nf(normal_id="NF_1", outcome="OS"),
        _nf(normal_id="NF_2", outcome="R-CHOP response"),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert len(groups) == 2


def test_different_subject_different_group():
    nfs = [
        _nf(normal_id="NF_1", subject="CD30"),
        _nf(normal_id="NF_2", subject="BCL2"),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert len(groups) == 2


def test_different_category_different_group():
    """category IS a grouping key — same subject/outcome/relation_type but different category → 2 groups."""
    nfs = [
        _nf(normal_id="NF_1", category="IHC"),
        _nf(normal_id="NF_2", category="prognosis"),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert len(groups) == 2


def test_different_relation_type_different_group():
    nfs = [
        _nf(normal_id="NF_1", relation_type=RelationTypeEnum.prognostic),
        _nf(normal_id="NF_2", relation_type=RelationTypeEnum.expression),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert len(groups) == 2


# ── Direction is NOT a grouping key ───────────────────────────────────────────

def test_opposing_directions_same_group():
    nfs = [
        _nf(normal_id="NF_1", direction=DirectionEnum.negative),
        _nf(normal_id="NF_2", direction=DirectionEnum.absent),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert len(groups) == 1
    assert set(groups[0].member_ids) == {"NF_1", "NF_2"}


def test_direction_counts_correct():
    nfs = [
        _nf(normal_id="NF_1", direction=DirectionEnum.negative),
        _nf(normal_id="NF_2", direction=DirectionEnum.negative),
        _nf(normal_id="NF_3", direction=DirectionEnum.absent),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert groups[0].direction_counts == {"negative": 2, "absent": 1}


def test_direction_counts_keys_are_strings():
    """direction_counts must use str values for JSON roundtrip safety."""
    nfs = [_nf(normal_id="NF_1", direction=DirectionEnum.positive)]
    groups = GroupStage().group(nfs, PMCID)
    key = list(groups[0].direction_counts.keys())[0]
    assert isinstance(key, str)
    assert key == "positive"


# ── scope_heterogeneity ───────────────────────────────────────────────────────

def test_heterogeneity_zero_single_member():
    groups = GroupStage().group([_nf()], PMCID)
    assert groups[0].scope_heterogeneity == 0.0


def test_heterogeneity_zero_when_scope_agrees():
    scope = FindingScope(disease_subtype="DLBCL", scope_parsed=True)
    nfs = [_nf(normal_id="NF_1", scope=scope), _nf(normal_id="NF_2", scope=scope)]
    groups = GroupStage().group(nfs, PMCID)
    assert groups[0].scope_heterogeneity == 0.0


def test_heterogeneity_nonzero_when_subtypes_differ():
    nfs = [
        _nf(normal_id="NF_1", scope=FindingScope(disease_subtype="GCB-DLBCL")),
        _nf(normal_id="NF_2", scope=FindingScope(disease_subtype="non-GCB DLBCL")),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert groups[0].scope_heterogeneity > 0.0


def test_heterogeneity_none_fields_not_counted_as_varying():
    """A field that is None in one member and non-None in another does not vary."""
    nfs = [
        _nf(normal_id="NF_1", scope=FindingScope(disease_subtype="DLBCL")),
        _nf(normal_id="NF_2", scope=FindingScope(disease_subtype=None)),
    ]
    groups = GroupStage().group(nfs, PMCID)
    assert groups[0].scope_heterogeneity == 0.0


def test_direction_counts_none_direction_stored_as_unclear():
    """None direction is stored as 'unclear' key in direction_counts."""
    nfs = [_nf(normal_id="NF_1", direction=None)]
    groups = GroupStage().group(nfs, PMCID)
    assert groups[0].direction_counts == {"unclear": 1}


# ── group_id stability ────────────────────────────────────────────────────────

def test_group_id_stable():
    assert _group_id("CD30", "OS", "prognostic", pmcid=PMCID) == _group_id(
        "CD30", "OS", "prognostic", pmcid=PMCID
    )


def test_group_id_differs_across_inputs():
    assert _group_id("CD30", "OS", "prognostic", pmcid=PMCID) != _group_id(
        "BCL2", "OS", "prognostic", pmcid=PMCID
    )
    assert _group_id("CD30", "OS", "prognostic", pmcid=PMCID) != _group_id(
        "CD30", "PFS", "prognostic", pmcid=PMCID
    )
    assert _group_id("CD30", "OS", "prognostic", pmcid=PMCID) != _group_id(
        "CD30", "OS", "expression", pmcid=PMCID
    )


def test_group_id_differs_across_pmcids():
    """Same (subject, outcome, relation, category) in two papers must produce distinct group_ids."""
    assert _group_id("CD30", "OS", "prognostic", pmcid="PMC_A") != _group_id(
        "CD30", "OS", "prognostic", pmcid="PMC_B"
    )


# ── B-022 regression: CUI population must not split otherwise-identical buckets ─


def _nf_with_cui(normal_id: str, subject_cui: str | None) -> NormalFinding:
    nf = _nf(normal_id=normal_id)
    return nf.model_copy(update={"subject_cui": subject_cui})


def test_b022_partial_cui_population_still_same_bucket():
    """Two findings with identical normalized subject/outcome/relation/category
    must land in the same FindingGroup, regardless of whether subject_cui is
    populated on one and missing on the other. Pre-fix: `subject_cui if subject_cui
    else subject` mixed namespaces; the CUI-bearing finding and the CUI-missing
    finding produced different keys and split into two singleton groups.
    """
    a = _nf_with_cui("NF_a", subject_cui="C0XXXXXX")
    b = _nf_with_cui("NF_b", subject_cui=None)
    groups = GroupStage().group([a, b], PMCID)
    assert len(groups) == 1
    assert set(groups[0].member_ids) == {"NF_a", "NF_b"}


def test_b022_group_id_independent_of_cui():
    """Helper-level: the same _group_id is returned whether or not a caller
    passes CUI hints (the signature no longer accepts them).
    """
    base = _group_id("CD30", "OS", "prognostic", pmcid=PMCID)
    # Same inputs → same key, regardless of which finding had a CUI upstream.
    assert _group_id("CD30", "OS", "prognostic", pmcid=PMCID) == base


# ── FindingGroup model ────────────────────────────────────────────────────────

def test_finding_group_subject_outcome_are_str():
    """FindingGroup.subject_entity and outcome_entity must be str, not Optional."""
    groups = GroupStage().group([_nf()], PMCID)
    assert isinstance(groups[0].subject_entity, str)
    assert isinstance(groups[0].outcome_entity, str)


def test_finding_group_roundtrip():
    nfs = [
        _nf(normal_id="NF_1", direction=DirectionEnum.negative),
        _nf(normal_id="NF_2", direction=DirectionEnum.absent),
    ]
    g = GroupStage().group(nfs, PMCID)[0]
    g2 = FindingGroup(**g.model_dump())
    assert g2.group_id == g.group_id
    assert g2.direction_counts == {"negative": 1, "absent": 1}
    assert isinstance(g2.subject_entity, str)
