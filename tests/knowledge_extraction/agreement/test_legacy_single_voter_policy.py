"""``legacy_single_voter_policy`` — N=1 handling on the non-router path.

Before this knob, ``AgreementChecker.compute`` hard-coded N=1 → KEEP with
``confidence=1.0`` (the lone surviving voter accepted without peer
comparison). The new ``single_voter_policy`` parameter lets that branch
ESCALATE instead, so a chunk whose other voters' API calls failed can be
routed up the cascade rather than silently accepted.

These tests pin the new branching:

  * N=1 + policy="keep"      → KEEP,     confidence=1.0  (legacy default)
  * N=1 + policy="escalate"  → ESCALATE, confidence=0.0  (new option)
  * N=0                       → ESCALATE always (policy ignored)
  * N≥2                       → scorer runs as usual (policy does not bypass)

Plus an independence test against ``MapOutputRouter.single_voter_policy``:
the two policies live on different objects and do not bleed into each
other. The router's N=1 case fires only when the router is active; the
checker's N=1 case fires only on the legacy path.
"""
from __future__ import annotations

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.agreement.checker import AgreementChecker
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import (
    ChunkDecision,
    ScoreBundle,
)
from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    AuditableSummary,
    AuditMetadata,
    DirectionEnum,
    Finding,
    FindingScope,
    RelationTypeEnum,
)


# Fixtures


def _finding(
    *,
    subject: str = "CD30",
    outcome: str = "OS",
    direction: DirectionEnum = DirectionEnum.positive,
    claim: str = "CD30 predicts OS",
) -> Finding:
    return Finding(
        category="IHC",
        claim=claim,
        evidence=["S1|PMC1|101"],
        confidence="high",
        verbatim_support="CD30 was associated with OS",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=RelationTypeEnum.prognostic,
        direction=direction,
        scope=FindingScope(),
        grounding_score=0.9,
    )


def _voter(*findings: Finding, chunk_id: str = "C1") -> AuditableSummary:
    return AuditableSummary(
        chunk_id=chunk_id,
        findings=list(findings) or [_finding()],
        summary_text="(test fixture)",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["S1"],
            pmcids_referenced=["PMC1"],
            uncited_sentences=[],
        ),
    )


class _KeepingScorerStub:
    """Returns a high-confidence KEEP bundle for N>=2 inputs.

    Used to verify that the normal agreement path still runs when policy is
    set — the policy must not intercept N>=2 calls.
    """

    def __init__(self, confidence: float = 0.95) -> None:
        self._confidence = confidence
        self.calls: list[int] = []

    def compute(self, outputs, source_text=None, context=None):  # noqa: ARG002
        self.calls.append(len(outputs))
        return ScoreBundle(
            confidence=self._confidence,
            embedding_agreement=self._confidence,
            best_index=0,
            score_details={"pairwise_upper": [], "avg_sim": [self._confidence]},
        )


class _RaisingScorerStub:
    """Fails loudly if the scorer is invoked.

    Used to prove the N<2 fast-paths short-circuit before reaching the scorer.
    """

    def compute(self, outputs, source_text=None, context=None):  # noqa: ARG002
        raise AssertionError(
            "Scorer must not be invoked on N<2 — single-voter policy should "
            "short-circuit before the scorer is called."
        )


# N=1


def test_n1_keep_policy_returns_keep_confidence_one():
    """Default ``policy="keep"`` preserves the prior implicit behaviour."""
    checker = AgreementChecker(scorer=_RaisingScorerStub(), single_voter_policy="keep")
    bundle = checker.compute([_voter()])
    assert bundle.decision == ChunkDecision.KEEP
    assert bundle.confidence == pytest.approx(1.0)


def test_n1_keep_is_the_default_policy():
    """Constructor default must remain ``"keep"`` for back-compat."""
    checker = AgreementChecker(scorer=_RaisingScorerStub())
    bundle = checker.compute([_voter()])
    assert bundle.decision == ChunkDecision.KEEP
    assert bundle.confidence == pytest.approx(1.0)


def test_n1_escalate_policy_returns_escalate_confidence_zero():
    """``policy="escalate"`` treats N=1 as low-evidence."""
    checker = AgreementChecker(scorer=_RaisingScorerStub(), single_voter_policy="escalate")
    bundle = checker.compute([_voter()])
    assert bundle.decision == ChunkDecision.ESCALATE
    assert bundle.confidence == pytest.approx(0.0)


# N=0


@pytest.mark.parametrize("policy", ["keep", "escalate"])
def test_n0_always_escalates_regardless_of_policy(policy):
    """Empty voter list → ESCALATE / confidence=0.0 for every policy value."""
    checker = AgreementChecker(scorer=_RaisingScorerStub(), single_voter_policy=policy)
    bundle = checker.compute([])
    assert bundle.decision == ChunkDecision.ESCALATE
    assert bundle.confidence == pytest.approx(0.0)


# N>=2 — policy must not intercept normal agreement logic


@pytest.mark.parametrize("policy", ["keep", "escalate"])
def test_n2_uses_normal_agreement_logic_regardless_of_policy(policy):
    """For N>=2 the scorer is invoked and its decision is honoured."""
    scorer = _KeepingScorerStub(confidence=0.95)
    checker = AgreementChecker(scorer=scorer, theta=0.7, single_voter_policy=policy)
    bundle = checker.compute([_voter(), _voter()])
    # Scorer was reached (policy did not short-circuit) and its KEEP survived
    # the post-scorer polarity-conflict + theta logic.
    assert scorer.calls == [2], f"scorer expected to be called once with N=2, got {scorer.calls}"
    assert bundle.decision == ChunkDecision.KEEP
    assert bundle.embedding_agreement == pytest.approx(0.95)


def test_n3_uses_normal_agreement_logic():
    """N=3 sanity — covers the common L1 case."""
    scorer = _KeepingScorerStub(confidence=0.92)
    checker = AgreementChecker(scorer=scorer, theta=0.7, single_voter_policy="escalate")
    bundle = checker.compute([_voter(), _voter(), _voter()])
    assert scorer.calls == [3]
    assert bundle.decision == ChunkDecision.KEEP


# Independence from MapOutputRouter.single_voter_policy


def test_legacy_and_router_policies_are_independent_fields():
    """Storing opposite policies on each object must not cross-contaminate."""
    from nlp_histo.pipeline.stages.knowledge_extraction.routing import MapOutputRouter

    checker_escalate = AgreementChecker(
        scorer=_RaisingScorerStub(), single_voter_policy="escalate",
    )
    router_keep = MapOutputRouter(
        agreement_checker=AgreementChecker(scorer=_RaisingScorerStub()),
        single_voter_policy="keep",
    )

    assert checker_escalate._single_voter_policy == "escalate"
    assert router_keep._single_voter_policy == "keep"

    # Reverse the pairing — fields stay decoupled.
    checker_keep = AgreementChecker(
        scorer=_RaisingScorerStub(), single_voter_policy="keep",
    )
    router_escalate = MapOutputRouter(
        agreement_checker=AgreementChecker(scorer=_RaisingScorerStub()),
        single_voter_policy="escalate",
    )
    assert checker_keep._single_voter_policy == "keep"
    assert router_escalate._single_voter_policy == "escalate"


def test_router_n_eligible_1_unaffected_by_agreement_checker_policy():
    """Router's N_eligible=1 branch uses its own ``single_voter_policy``.

    Drives ``_chunk_decision_from_classifications`` directly with a single
    eligible voter and confirms the outcome reflects the router's policy
    even when the wrapped AgreementChecker carries the opposite policy.
    """
    from nlp_histo.pipeline.stages.knowledge_extraction.routing import MapOutputRouter
    from nlp_histo.pipeline.stages.knowledge_extraction.routing.router import VoterClassification

    # Router policy=KEEP, AgreementChecker policy=ESCALATE — they must not mix.
    router = MapOutputRouter(
        agreement_checker=AgreementChecker(
            scorer=_RaisingScorerStub(), single_voter_policy="escalate",
        ),
        single_voter_policy="keep",
    )
    one_eligible = [
        VoterClassification(
            voter_index=0,
            status="eligible",
            schema_validations=[],
            grounding_validations=[],
            grounding_pass_fraction=1.0,
            mean_evidence_chain_length=1.0,
        )
    ]
    decision = router._chunk_decision_from_classifications(one_eligible)
    assert decision is not None
    assert decision.decision == ChunkDecision.KEEP
    assert decision.valid_voter_indices == [0]

    # And vice-versa: router=ESCALATE, checker=KEEP.
    router2 = MapOutputRouter(
        agreement_checker=AgreementChecker(
            scorer=_RaisingScorerStub(), single_voter_policy="keep",
        ),
        single_voter_policy="escalate",
    )
    decision2 = router2._chunk_decision_from_classifications(one_eligible)
    assert decision2 is not None
    assert decision2.decision == ChunkDecision.ESCALATE
