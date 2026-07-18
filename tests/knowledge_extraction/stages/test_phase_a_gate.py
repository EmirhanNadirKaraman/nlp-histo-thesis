"""
Phase A gate tests — direction inference, expression marker gate,
and entity normalization.
No LLM or NLI model required.
"""
import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    CanonicalRule,
    DirectionEnum,
    RelationTypeEnum,
)
from nlp_histo.pipeline.stages.knowledge_extraction.stages.normalize_stage import (
    NormalizeStage,
    infer_direction,
    normalize_entity,
)
from nlp_histo.pipeline.stages.knowledge_extraction.stages.relate_stage import (
    _classify_pair,
    _norm_outcome_expression,
    _should_compare,
)


# infer_direction

@pytest.mark.parametrize("claim,expected", [
    # negative triggers
    ("CD30 was not expressed",          DirectionEnum.negative),
    ("No cytologic atypia was observed", DirectionEnum.negative),
    ("Necrosis was absent",             DirectionEnum.negative),
    ("Mitoses were not seen",           DirectionEnum.negative),
    ("Staining was negative",           DirectionEnum.negative),
    # positive triggers
    ("CD30 was positive",               DirectionEnum.positive),
    ("Open vascular channels were present", DirectionEnum.positive),
    ("CD34 expression was detected",    DirectionEnum.positive),
    ("CD31 is expressed",               DirectionEnum.positive),
    # unclear
    ("Ki-67 index 80%",                 DirectionEnum.unclear),
    ("Female predominance",             DirectionEnum.unclear),
    # negative wins over positive when both keywords present
    ("not expressed but reactive staining", DirectionEnum.negative),
])
def test_infer_direction(claim, expected):
    assert infer_direction(claim) == expected


# _norm_outcome_expression

@pytest.mark.parametrize("entity,expected", [
    ("CD31 expression",        "cd31"),
    ("CD34 expression",        "cd34"),
    ("CD30 positivity",        "cd30"),
    ("Ki-67 staining",         "ki-67"),
    ("BCL2 immunoreactivity",  "bcl2"),
    ("CD30",                   "cd30"),   # no suffix — unchanged
])
def test_norm_outcome_expression(entity, expected):
    assert _norm_outcome_expression(entity) == expected


# _should_compare

def _make_rule(
    subject: str,
    outcome: str,
    relation_type: RelationTypeEnum = RelationTypeEnum.expression,
    category: str = "IHC",
    predicate_text: str = "entity was expressed",
) -> CanonicalRule:
    return CanonicalRule(
        canonical_id=f"CR_{subject}_{outcome}",
        group_id=f"GRP_{subject}_{outcome}",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=relation_type,
        direction=DirectionEnum.positive,
        predicate_text=predicate_text,
        is_conflicted=False,
        study_coverage="single_study",
        category=category,  # type: ignore[arg-type]
        supporting_pmcids=["PMC10047158"],
        member_normal_ids=[],
        mean_grounding_score=0.8,
        finding_count=1,
    )


def test_should_compare_rejects_different_expression_markers():
    """CD31 expression vs CD34 expression — different markers → outcome_incompatible."""
    a = _make_rule("tumour cells", "CD31 expression")
    b = _make_rule("tumour cells", "CD34 expression")
    ok, reason = _should_compare(a, b)
    assert not ok
    assert reason == "outcome_incompatible"


def test_should_compare_rejects_different_features():
    """Open vascular channels vs cytologic atypia — different outcomes → outcome_incompatible."""
    a = _make_rule("CEAN", "open vascular channels", relation_type=RelationTypeEnum.has_feature)
    b = _make_rule("CEAN", "no cytologic atypia",   relation_type=RelationTypeEnum.has_feature)
    ok, reason = _should_compare(a, b)
    assert not ok
    assert reason == "outcome_incompatible"


def test_should_compare_accepts_same_expression_marker():
    """CD30 expression vs CD30 expression — same marker → eligible."""
    a = _make_rule("tumour cells", "CD30 expression")
    b = _make_rule("tumour cells", "CD30 expression", predicate_text="CD30 was strongly positive")
    ok, reason = _should_compare(a, b)
    assert ok
    assert reason == ""


def test_should_compare_rejects_subject_mismatch():
    a = _make_rule("tumour cells", "CD30 expression")
    b = _make_rule("stromal cells", "CD30 expression")
    ok, reason = _should_compare(a, b)
    assert not ok
    assert reason == "subject_mismatch"


def test_should_compare_rejects_category_mismatch():
    a = _make_rule("CEAN", "CD30 expression", category="IHC")
    b = _make_rule("CEAN", "CD30 expression", category="morphology")
    ok, reason = _should_compare(a, b)
    assert not ok
    assert reason == "category_mismatch"


def test_should_compare_rejects_relation_type_mismatch():
    a = _make_rule("CEAN", "open vascular channels", relation_type=RelationTypeEnum.has_feature)
    b = _make_rule("CEAN", "open vascular channels", relation_type=RelationTypeEnum.expression)
    ok, reason = _should_compare(a, b)
    assert not ok
    assert reason == "relation_type_mismatch"


# _classify_pair: CONTRADICT prerequisite

def _high_scores() -> dict[str, float]:
    return {"entailment": 0.05, "contradiction": 0.95, "neutral": 0.0}


def test_classify_pair_no_contradict_both_positive():
    """Two positive rules must not emit CONTRADICT even with high NLI contradiction score."""
    a = _make_rule("tumour cells", "CD30 expression",
                   predicate_text="CD30 was positive")
    b = _make_rule("tumour cells", "CD30 expression",
                   predicate_text="CD30 was strongly expressed")
    label = _classify_pair(a, b, _high_scores(), _high_scores(),
                           entailment_threshold=0.55, contradiction_threshold=0.65)
    assert label is None or label.value != "CONTRADICT"


def test_classify_pair_no_contradict_both_uncertain():
    """Two uncertain rules must not emit CONTRADICT."""
    a = _make_rule("tumour cells", "CD30 expression",
                   predicate_text="CD30 index 80%")
    b = _make_rule("tumour cells", "CD30 expression",
                   predicate_text="CD30 index 40%")
    label = _classify_pair(a, b, _high_scores(), _high_scores(),
                           entailment_threshold=0.55, contradiction_threshold=0.65)
    assert label is None or label.value != "CONTRADICT"


def test_classify_pair_contradict_positive_vs_negative():
    """Positive rule vs negative rule with high contradiction score → CONTRADICT.

    The polarity guard in `_classify_pair` only allows CONTRADICT when the two
    rules' structured `direction` fields point opposite ways — so we must set
    the negative rule's direction explicitly here.
    """
    a = _make_rule("tumour cells", "CD30 expression",
                   predicate_text="CD30 was positive")
    b = _make_rule("tumour cells", "CD30 expression",
                   predicate_text="CD30 was negative")
    b = b.model_copy(update={"direction": DirectionEnum.negative})
    label = _classify_pair(a, b, _high_scores(), _high_scores(),
                           entailment_threshold=0.55, contradiction_threshold=0.65)
    from nlp_histo.pipeline.stages.knowledge_extraction.models import RelationTypeLabel
    assert label == RelationTypeLabel.CONTRADICT


# Entity normalization

@pytest.mark.parametrize("surface,expected", [
    ("Cutaneous epithelioid angiomatous nodule", "CEAN"),
    ("cutaneous epithelioid angiomatous nodule", "CEAN"),
    ("CEAN",   "CEAN"),
    ("caen",   "CEAN"),
    ("Tumour cells",   "tumour cells"),
    ("tumour cells",   "tumour cells"),
    ("Tumor cells",    "tumour cells"),
    ("neoplastic cells", "tumour cells"),
    ("CD31",  "CD31"),
    ("cd31",  "CD31"),
    ("CD34",  "CD34"),
    ("cd34",  "CD34"),
])
def test_normalize_entity(surface, expected):
    assert normalize_entity(surface) == expected


def test_normalize_stage_normalizes_entities():
    """NormalizeStage._norm() is exercised via normalize()."""
    from nlp_histo.pipeline.stages.knowledge_extraction.models import Finding
    stage = NormalizeStage()
    findings = [
        Finding(
            category="IHC",
            claim="Tumour cells were CD31 positive",
            evidence=["S1|PMC10047158|1"],
            confidence="high",
            verbatim_support="tumour cells were CD31 positive",
            subject_entity="Tumor cells",
            outcome_entity="CD31 expression",
            relation_type=RelationTypeEnum.expression,
            direction=DirectionEnum.positive,
        )
    ]
    results = stage.normalize(findings, "PMC10047158")
    assert len(results) == 1
    assert results[0].subject_entity == "tumour cells"
    # UMLS linker resolves "CD31 expression" to its canonical concept name.
    assert results[0].outcome_entity == "CD31 Antigens"
