"""B-051 — hard-fail polarity rules in the MAP agreement gate.

Pre-fix: opposite-polarity paraphrases with embedding similarity ≈ 1.0
produced score = 0.80 (max-bounded by the 20% soft penalty in
`EmbeddingScorer`), passing `theta=0.7`. Two voters that directly
contradict could be accepted as KEEP.

Post-fix: `AgreementChecker.compute` calls `detect_polarity_conflict`
after the scorer runs but before theta is applied. When two comparable
findings carry opposite hard polarities (positive vs negative), the
checker forces `ChunkDecision.ESCALATE` and stamps
`bundle.score_details["hard_fail_reason"] = "polarity_conflict"`.

The router (`MapOutputRouter._agreement_gate`) translates that into a
`RoutingDecision` with `reason_codes=[ReasonCode.POLARITY_CONFLICT]` —
never co-emitted with low-agreement reason codes.

These tests are deterministic — they construct synthetic Pydantic
findings and a fake scorer; no real embeddings, no LLM calls.
"""
from __future__ import annotations


import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.agreement.checker import AgreementChecker
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.polarity_conflict import (
    detect_polarity_conflict,
)
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import (
    ChunkDecision,
    ScoreBundle,
)
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import VoterContext
from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    AuditableSummary,
    AuditMetadata,
    DirectionEnum,
    Finding,
    FindingScope,
    RelationTypeEnum,
)
from nlp_histo.pipeline.stages.knowledge_extraction.routing.models import GateOrigin, ReasonCode


# Fixtures


def _finding(
    *,
    subject: str | None = "CD30",
    outcome: str | None = "OS",
    direction: DirectionEnum = DirectionEnum.positive,
    relation_type: RelationTypeEnum = RelationTypeEnum.prognostic,
    category: str = "IHC",
    claim: str = "CD30 predicts OS",
    grounding_score: float = 0.9,
) -> Finding:
    return Finding(
        category=category,
        claim=claim,
        evidence=["S1|PMC1|101"],
        confidence="high",
        verbatim_support="CD30 was associated with OS",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=relation_type,
        direction=direction,
        scope=FindingScope(),
        grounding_score=grounding_score,
    )


def _voter(*findings: Finding, chunk_id: str = "C1") -> AuditableSummary:
    return AuditableSummary(
        chunk_id=chunk_id,
        findings=list(findings),
        summary_text="(test fixture)",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["S1"],
            pmcids_referenced=["PMC1"],
            uncited_sentences=[],
        ),
    )


class _HighAgreementScorerStub:
    """Returns a high-confidence KEEP bundle so theta would normally KEEP."""

    def __init__(self, confidence: float = 0.95) -> None:
        self._confidence = confidence

    def compute(self, outputs, source_text=None, context=None):  # noqa: ARG002
        return ScoreBundle(
            confidence=self._confidence,
            embedding_agreement=self._confidence,
            best_index=0,
            score_details={"pairwise_upper": [], "avg_sim": [self._confidence]},
        )


# Test 1: comparable positive vs negative → ESCALATE


def test_b051_comparable_positive_vs_negative_escalates():
    """High mocked embedding score must not override the polarity hard-fail."""
    pos = _voter(_finding(direction=DirectionEnum.positive))
    neg = _voter(_finding(direction=DirectionEnum.negative))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([pos, neg])
    assert bundle.decision == ChunkDecision.ESCALATE
    assert bundle.score_details["hard_fail_reason"] == "polarity_conflict"
    assert bundle.score_details["polarity_conflict_details"]["count"] == 1
    # Scorer output preserved for trace inspection
    assert bundle.embedding_agreement == pytest.approx(0.95)


# Test 2: same polarity, high score → KEEP (no regression)


def test_b051_comparable_positive_vs_positive_keeps():
    a = _voter(_finding(direction=DirectionEnum.positive))
    b = _voter(_finding(direction=DirectionEnum.positive))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([a, b])
    assert bundle.decision == ChunkDecision.KEEP
    assert (bundle.score_details or {}).get("hard_fail_reason") is None


# Test 3: non-comparable opposite polarity → no hard-fail


def test_b051_non_comparable_opposite_polarity_no_hard_fail():
    """Different subject_entity → not the same claim → no hard-fail; theta KEEPS."""
    pos = _voter(_finding(subject="CD30", direction=DirectionEnum.positive))
    neg = _voter(_finding(subject="BCL2", direction=DirectionEnum.negative))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([pos, neg])
    assert bundle.decision == ChunkDecision.KEEP
    assert (bundle.score_details or {}).get("hard_fail_reason") is None


# Test 4: unclear ↔ negative → no hard-fail


def test_b051_unclear_vs_negative_no_hard_fail():
    a = _voter(_finding(direction=DirectionEnum.unclear))
    b = _voter(_finding(direction=DirectionEnum.negative))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([a, b])
    assert bundle.decision == ChunkDecision.KEEP
    assert (bundle.score_details or {}).get("hard_fail_reason") is None


# Test 5: no_direction ↔ positive → no hard-fail


def test_b051_no_direction_vs_positive_no_hard_fail():
    a = _voter(_finding(direction=DirectionEnum.no_direction))
    b = _voter(_finding(direction=DirectionEnum.positive))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([a, b])
    assert bundle.decision == ChunkDecision.KEEP
    assert (bundle.score_details or {}).get("hard_fail_reason") is None


# Test 6: absent ↔ positive (intentionally excluded from v1)


def test_b051_absent_vs_positive_no_hard_fail():
    """v1 policy: `absent` semantically overlaps with `negative` in
    biomarker-expression contexts. Excluded until B-025 calibration. This
    test pins the v1 behaviour so a future widening is deliberate.
    """
    a = _voter(_finding(direction=DirectionEnum.absent))
    b = _voter(_finding(direction=DirectionEnum.positive))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([a, b])
    assert bundle.decision == ChunkDecision.KEEP
    assert (bundle.score_details or {}).get("hard_fail_reason") is None


# Test 7: partial ↔ negative (intentionally excluded from v1)


def test_b051_partial_vs_negative_no_hard_fail():
    """v1 policy: `partial` is bundled with `positive` in
    `relate_stage._POSITIVE_DIRECTIONS`; MAP must not be stricter than RELATE.
    """
    a = _voter(_finding(direction=DirectionEnum.partial))
    b = _voter(_finding(direction=DirectionEnum.negative))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([a, b])
    assert bundle.decision == ChunkDecision.KEEP
    assert (bundle.score_details or {}).get("hard_fail_reason") is None


# Test 8: partial ↔ positive → same RELATE class → no hard-fail


def test_b051_partial_vs_positive_no_hard_fail():
    a = _voter(_finding(direction=DirectionEnum.partial))
    b = _voter(_finding(direction=DirectionEnum.positive))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([a, b])
    assert bundle.decision == ChunkDecision.KEEP
    assert (bundle.score_details or {}).get("hard_fail_reason") is None


# Test 9: None subject_entity → comparability disqualified


def test_b051_none_subject_no_hard_fail():
    a = _voter(_finding(subject=None, direction=DirectionEnum.positive))
    b = _voter(_finding(subject="CD30", direction=DirectionEnum.negative))
    # Helper-level: no conflict reported
    assert detect_polarity_conflict([a, b]) is None
    # End-to-end: theta path takes over
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([a, b])
    assert bundle.decision == ChunkDecision.KEEP


# Test 10: trace metadata structure


def test_b051_trace_metadata_carries_polarity_conflict():
    pos = _voter(_finding(direction=DirectionEnum.positive))
    neg = _voter(_finding(direction=DirectionEnum.negative))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    bundle = checker.compute([pos, neg])
    details = bundle.score_details["polarity_conflict_details"]
    assert details["count"] == 1
    assert details["policy_version"] == "polarity_hard_fail_v1"
    pair = details["voter_pairs"][0]
    assert (pair["voter_i"], pair["voter_j"]) == (0, 1)
    assert {pair["direction_i"], pair["direction_j"]} == {"positive", "negative"}
    assert pair["comparability_key"][0] == "cd30"  # casefolded
    assert pair["comparability_key"][1] == "os"
    # Pre-existing scorer details preserved so calibration can audit
    assert "pairwise_upper" in bundle.score_details


# Test 11: router path emits POLARITY_CONFLICT only


def test_b051_router_path_polarity_conflict_reason_code():
    """`MapOutputRouter._agreement_gate` must translate the AgreementChecker
    hard-fail signal into a RoutingDecision with only ReasonCode.POLARITY_CONFLICT.

    Co-emitting INSUFFICIENT_AGREEMENT or ESCALATED_DUE_TO_LOW_AGREEMENT would
    lie about the cause (the score was high; only the structural check failed).
    """
    from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext
    from nlp_histo.pipeline.stages.knowledge_extraction.routing.router import MapOutputRouter

    pos = _voter(_finding(direction=DirectionEnum.positive))
    neg = _voter(_finding(direction=DirectionEnum.negative))
    checker = AgreementChecker(_HighAgreementScorerStub(), theta=0.7, reject_theta=0.2)
    router = MapOutputRouter(agreement_checker=checker)

    # Exercise `_agreement_gate` directly with the inputs `route()` would build
    # if all voters passed schema + provenance. This isolates the
    # router→reason-code translation from the unrelated gating logic.
    context = AgreementContext(
        voter_contexts=[
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=10.0),
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=10.0),
        ]
    )
    decision = router._agreement_gate(
        valid_outputs=[pos, neg],
        source_text="x",
        valid_voter_indices=[0, 1],
        context=context,
    )
    assert decision.decision == ChunkDecision.ESCALATE
    assert decision.gate_origin == GateOrigin.AGREEMENT_GATE
    assert decision.reason_codes == [ReasonCode.POLARITY_CONFLICT]
    assert ReasonCode.INSUFFICIENT_AGREEMENT not in decision.reason_codes
    assert ReasonCode.ESCALATED_DUE_TO_LOW_AGREEMENT not in decision.reason_codes
    assert "overridden" in decision.explanation.lower()
    assert "low agreement" not in decision.explanation.lower()


# Tests 12–14: `force_escalate_on_polarity_conflict` flag (promoted 2026-05-26)
#
# The flag preserves B-051 behaviour at its default (True). Setting False keeps
# the conflict marker in score_details (so sweep harnesses can count it) but
# does not override the scorer's natural decision. The tests below use the
# same `_HighAgreementScorerStub` as the canonical B-051 tests so the scorer's
# baseline decision is KEEP at the default theta — that's the only way to
# observe the override having (or not having) an effect: if the scorer itself
# said ESCALATE, the difference between True and False would be invisible.


def test_polarity_flag_true_preserves_b051_escalation():
    """``force_escalate_on_polarity_conflict=True`` (default) reproduces the
    canonical B-051 escalation: comparable positive/negative pair → ESCALATE,
    overriding the scorer's high-confidence KEEP."""
    pos = _voter(_finding(direction=DirectionEnum.positive))
    neg = _voter(_finding(direction=DirectionEnum.negative))
    checker = AgreementChecker(
        _HighAgreementScorerStub(), theta=0.7, reject_theta=0.2,
        force_escalate_on_polarity_conflict=True,
    )
    bundle = checker.compute([pos, neg])
    assert bundle.decision == ChunkDecision.ESCALATE
    assert bundle.score_details["hard_fail_reason"] == "polarity_conflict"
    assert bundle.score_details["polarity_conflict_details"]["count"] >= 1


def test_polarity_flag_false_records_marker_but_does_not_override():
    """``force_escalate_on_polarity_conflict=False`` still records the conflict
    in score_details (sweep harnesses count via this marker) but does not
    override the scorer's KEEP decision.

    The assertion is **"decision is what the scorer / theta would produce"**,
    not "decision is not ESCALATE" — a scorer that independently returned
    ESCALATE would still produce ESCALATE; the test would then be testing the
    wrong thing. Here the stub returns confidence=0.95 with no decision set,
    so AgreementChecker's theta fallback (theta=0.7) lands on KEEP. With
    flag=False we expect that KEEP to survive."""
    pos = _voter(_finding(direction=DirectionEnum.positive))
    neg = _voter(_finding(direction=DirectionEnum.negative))
    checker = AgreementChecker(
        _HighAgreementScorerStub(confidence=0.95),
        theta=0.7, reject_theta=0.2,
        force_escalate_on_polarity_conflict=False,
    )
    bundle = checker.compute([pos, neg])
    # Marker recorded for observability — sweep counter relies on this.
    assert bundle.score_details["hard_fail_reason"] == "polarity_conflict"
    assert bundle.score_details["polarity_conflict_details"]["count"] >= 1
    # Decision not overridden — the scorer's high-confidence KEEP survives.
    assert bundle.decision == ChunkDecision.KEEP
    # Embedding agreement preserved verbatim (no scorer state was mutated).
    assert bundle.embedding_agreement == pytest.approx(0.95)


def test_polarity_flag_false_no_conflict_no_marker():
    """``force_escalate_on_polarity_conflict=False`` with non-conflicting
    findings behaves identically to the default — no marker, scorer's KEEP
    stands. Parity test against the canonical positive/positive case."""
    a = _voter(_finding(direction=DirectionEnum.positive))
    b = _voter(_finding(direction=DirectionEnum.positive))
    checker = AgreementChecker(
        _HighAgreementScorerStub(), theta=0.7, reject_theta=0.2,
        force_escalate_on_polarity_conflict=False,
    )
    bundle = checker.compute([a, b])
    assert (bundle.score_details or {}).get("hard_fail_reason") is None
    assert bundle.decision == ChunkDecision.KEEP
