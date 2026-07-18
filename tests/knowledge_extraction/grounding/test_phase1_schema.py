"""
Phase 1 smoke tests — schema roundtrip and grounding filter.
No LLM or database required.
"""

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.grounding.grounding_filter import (
    filter_atomic_findings,
    score_findings,
)
from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    AtomicFinding,
    AuditableSummary,
    AuditMetadata,
    DirectionEnum,
    Finding,
    FindingScope,
    RelationTypeEnum,
)


# Fixtures

def _make_finding(**kwargs) -> Finding:
    defaults = dict(
        category="IHC",
        claim="CD30 predicts worse OS",
        evidence=["S1|PMC123|456"],
        confidence="high",
        verbatim_support="CD30 positivity was associated with worse OS",
    )
    defaults.update(kwargs)
    return Finding(**defaults)


class _StubTokenizer:
    """Whitespace tokenizer that satisfies the surface ``grounding_filter`` uses:
    callable returning ``{"input_ids": [...]}``, ``.decode``, ``.model_max_length``
    and ``.num_special_tokens_to_add(pair=True)``.  Premises in these tests are
    short enough that ``_split_windows`` returns a single window per pair, so we
    never exercise the truncation path — but we implement it correctly anyway.
    """

    model_max_length = 512

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(ids)

    def num_special_tokens_to_add(self, pair: bool = True) -> int:
        return 3


class _FakeNliPipe:
    """Mock HF NLI pipeline.  Emits the supplied entailment scores in order,
    one per (premise, hypothesis) call.  Exposes the ``.tokenizer`` attribute
    that ``_score_pairs`` reaches into for windowed tokenization.
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.tokenizer = _StubTokenizer()

    def __call__(self, inputs, truncation: bool = True):
        results = []
        for _ in inputs:
            s = self._scores.pop(0)
            results.append([
                {"label": "entailment", "score": s},
                {"label": "contradiction", "score": 1 - s},
            ])
        return results


def _fake_pipe(scores: list[float]) -> _FakeNliPipe:
    """Return a mock NLI pipe that emits the given entailment scores in order."""
    return _FakeNliPipe(scores)


# Schema construction

def test_finding_new_fields_construct():
    f = _make_finding(
        subject_entity="CD30",
        outcome_entity="overall survival",
        direction=DirectionEnum.negative,
        scope=FindingScope(disease_subtype="GCB-DLBCL", cohort_n=73, scope_parsed=True),
    )
    assert f.direction == DirectionEnum.negative
    assert f.scope.disease_subtype == "GCB-DLBCL"
    assert f.scope.cohort_n == 73
    assert f.scope.scope_parsed is True


def test_finding_new_fields_default_none():
    """relation_type defaults to unclear; direction defaults to no_direction; other optional fields default to None."""
    f = _make_finding()
    assert f.relation_type == RelationTypeEnum.unclear
    assert f.direction == DirectionEnum.no_direction
    assert f.scope is None
    assert f.subject_entity is None
    assert f.outcome_entity is None
    assert f.grounding_score is None


# Roundtrip (simulates PipelineCache serialize/deserialize)

def test_finding_roundtrip():
    f = _make_finding(
        direction=DirectionEnum.negative,
        scope=FindingScope(disease_subtype="GCB-DLBCL", cohort_n=73, scope_parsed=True),
    )
    f2 = Finding(**f.model_dump())
    assert f2.direction == DirectionEnum.negative
    assert f2.scope.disease_subtype == "GCB-DLBCL"
    assert f2.scope.cohort_n == 73


def test_auditable_summary_roundtrip():
    f = _make_finding(
        direction=DirectionEnum.absent,
        scope=FindingScope(treatment_context="R-CHOP", scope_parsed=True),
    )
    s = AuditableSummary(
        chunk_id="C1",
        findings=[f],
        summary_text="x",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=[],
            pmcids_referenced=[],
            uncited_sentences=[],
        ),
    )
    s2 = AuditableSummary(**s.model_dump())
    assert s2.findings[0].direction == DirectionEnum.absent
    assert s2.findings[0].scope.treatment_context == "R-CHOP"
    assert s2.findings[0].scope.scope_parsed is True


def test_pre_phase1_cache_entry_loads():
    """A cache entry written before Phase 1 (no new fields) must load without error."""
    old_payload = {
        "chunk_id": "C1",
        "findings": [
            {
                "category": "morphology",
                "claim": "Large cells with vesicular nuclei",
                "evidence": ["S2|PMC999|1"],
                "confidence": "medium",
                "verbatim_support": "Large cells were observed",
            }
        ],
        "summary_text": "summary",
        "audit_metadata": {
            "sentences_analyzed": 1,
            "sentences_cited": [],
            "pmcids_referenced": [],
            "uncited_sentences": [],
        },
    }
    s = AuditableSummary(**old_payload)
    assert s.findings[0].relation_type == RelationTypeEnum.unclear  # default when absent from cache
    assert s.findings[0].direction == DirectionEnum.no_direction  # legacy-default coercion
    assert s.findings[0].scope is None


# DirectionEnum

@pytest.mark.parametrize("value", ["positive", "negative", "absent", "partial", "unclear"])
def test_direction_enum_values(value):
    f = _make_finding(direction=DirectionEnum(value))
    data = f.model_dump()
    assert data["direction"] == value  # serializes as string
    f2 = Finding(**data)
    assert f2.direction == DirectionEnum(value)


# AtomicFinding defined but not wired

def test_atomic_finding_importable():
    """AtomicFinding must be importable and constructable (Phase 2 forward-compat)."""
    af = AtomicFinding(
        finding_id="PMC123__C1__F0__1",
        category="IHC",
        claim="CD30 predicts worse OS",
        subject_entity="CD30",
        outcome_entity="overall survival",
        direction=DirectionEnum.negative,
        scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        confidence="high",
        evidence=["S1|PMC123|456"],
        verbatim_support="CD30 was associated with worse OS",
        grounding_score=None,
    )
    assert af.scope.disease_subtype == "DLBCL"


# score_findings

def test_score_findings_writes_scores_in_place():
    findings = [_make_finding(), _make_finding(verbatim_support="")]
    score_findings(findings, nli_pipe=_fake_pipe([0.85, 0.0]))
    assert findings[0].grounding_score == pytest.approx(0.85)
    assert findings[1].grounding_score == 0.0


def test_score_findings_does_not_remove_any():
    findings = [_make_finding(), _make_finding(verbatim_support="")]
    score_findings(findings, nli_pipe=_fake_pipe([0.9, 0.1]))
    assert len(findings) == 2


def test_score_findings_empty_list():
    score_findings([], nli_pipe=_fake_pipe([]))  # must not raise


def test_score_findings_empty_verbatim_scores_zero_without_calling_pipe():
    calls = []

    def recording_pipe(inputs):
        calls.append(inputs)
        return []

    f = _make_finding(verbatim_support="   ")
    score_findings([f], nli_pipe=recording_pipe)
    assert f.grounding_score == 0.0
    assert calls == []  # empty verbatim short-circuits before hitting pipe


# filter_atomic_findings

def test_filter_atomic_findings_keeps_above_threshold():
    findings = [_make_finding(), _make_finding(claim="BCL2 claim", verbatim_support="BCL2 x")]
    kept = filter_atomic_findings(findings, threshold=0.5, nli_pipe=_fake_pipe([0.9, 0.3]))
    assert len(kept) == 1
    assert kept[0].claim == "CD30 predicts worse OS"


def test_filter_atomic_findings_writes_score_on_dropped():
    findings = [_make_finding(), _make_finding(claim="BCL2 claim", verbatim_support="BCL2 x")]
    filter_atomic_findings(findings, threshold=0.5, nli_pipe=_fake_pipe([0.9, 0.3]))
    assert findings[1].grounding_score == pytest.approx(0.3)


def test_filter_atomic_findings_empty_verbatim_dropped():
    findings = [_make_finding(verbatim_support="")]
    kept = filter_atomic_findings(findings, threshold=0.5, nli_pipe=_fake_pipe([]))
    assert kept == []
    assert findings[0].grounding_score == 0.0


def test_score_and_filter_same_objects():
    """score_findings + manual filter must agree with filter_atomic_findings."""
    f1 = _make_finding()
    f2 = _make_finding(claim="BCL2 claim", verbatim_support="BCL2 x")
    score_findings([f1, f2], nli_pipe=_fake_pipe([0.9, 0.3]))
    manual = [f for f in [f1, f2] if f.grounding_score >= 0.5]

    f3 = _make_finding()
    f4 = _make_finding(claim="BCL2 claim", verbatim_support="BCL2 x")
    auto = filter_atomic_findings([f3, f4], threshold=0.5, nli_pipe=_fake_pipe([0.9, 0.3]))

    assert len(manual) == len(auto) == 1
    assert manual[0].claim == auto[0].claim
