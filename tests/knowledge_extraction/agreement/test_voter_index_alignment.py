"""B-041 regression tests — preserve original voter-spec index through evaluate_chunk.

Pre-fix: `_run_voters` returned a filtered survivor list and downstream
consumers (producer_from_outcome, make_decision_record) indexed `voter_specs`
with `bundle.best_index` — which was a survivor-list index. Any voter API
failure misattributed the kept chunk to the wrong (provider, model).

Post-fix: `evaluate_chunk` takes a `voter_indices` argument mapping survivor
position back to original index. `ChunkOutcome.valid_voter_indices` is the
single source of truth for producer attribution.
"""
from __future__ import annotations

from nlp_histo.pipeline.stages.knowledge_extraction.agreement.decision import (
    evaluate_chunk,
    make_decision_record,
    producer_from_outcome,
)
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import (
    ChunkDecision,
    ScoreBundle,
)


class _AgreementStub:
    """Minimal stand-in: KEEP with best_index=0, bypasses real scoring."""

    def compute(self, voters, source_text=None):
        return ScoreBundle(
            decision=ChunkDecision.KEEP, best_index=0, confidence=0.9
        )

    def best(self, voters, bundle):
        return voters[bundle.best_index]


VOTER_SPECS = [("openai", "gpt-4o-mini"), ("google", "gemini-flash-lite"), ("mistral", "mistral-large")]


def test_b041_no_router_voter_0_failure_attributes_to_correct_survivor():
    """Voter 0 fails; survivors [v1, v2]. KEEP picks survivor[0]=v1.
    Pre-fix: voter_specs[0] returned (openai, gpt-4o-mini) — WRONG.
    Post-fix: voter_specs[1] returned (google, gemini-flash-lite) — CORRECT.
    """
    voters = ["v1", "v2"]            # voter 0 failed
    survivor_indices = [1, 2]
    outcome = evaluate_chunk(
        voters, chunk=[], pmcid="p", source_text="t",
        agreement=_AgreementStub(), router=None,
        voter_indices=survivor_indices,
    )
    assert outcome.keep is True
    assert outcome.best == "v1"
    assert outcome.valid_voter_indices == [1, 2]
    assert producer_from_outcome(outcome, VOTER_SPECS) == ("google", "gemini-flash-lite")


def test_b041_no_router_identity_when_voter_indices_none():
    """Backwards-compat: when voter_indices is omitted, identity mapping is used."""
    voters = ["v0", "v1", "v2"]
    outcome = evaluate_chunk(
        voters, chunk=[], pmcid="p", source_text="t",
        agreement=_AgreementStub(), router=None,
    )
    assert outcome.valid_voter_indices == [0, 1, 2]
    assert producer_from_outcome(outcome, VOTER_SPECS) == ("openai", "gpt-4o-mini")


def test_b041_no_router_voter_1_failure_attributes_correctly():
    """Voter 1 fails; survivors [v0, v2]. KEEP picks survivor[0]=v0 (orig 0)."""
    outcome = evaluate_chunk(
        ["v0", "v2"], chunk=[], pmcid="p", source_text="t",
        agreement=_AgreementStub(), router=None,
        voter_indices=[0, 2],
    )
    assert outcome.valid_voter_indices == [0, 2]
    assert producer_from_outcome(outcome, VOTER_SPECS) == ("openai", "gpt-4o-mini")


def test_b041_no_router_all_voters_failed():
    """Empty survivor list short-circuits to keep=False with no bundle."""
    outcome = evaluate_chunk(
        [], chunk=[], pmcid="p", source_text="t",
        agreement=_AgreementStub(), router=None,
        voter_indices=[],
    )
    assert outcome.keep is False
    assert outcome.best is None
    assert outcome.agreement_bundle is None
    assert producer_from_outcome(outcome, VOTER_SPECS) is None


def test_b041_make_decision_record_records_original_index():
    """best_voter_index in the CascadeDecisionRecord is the ORIGINAL spec index."""
    outcome = evaluate_chunk(
        ["v2"], chunk=[], pmcid="p", source_text="t",
        agreement=_AgreementStub(), router=None,
        voter_indices=[2],
    )
    rec = make_decision_record(
        outcome,
        run_id="r", pmcid="p", chunk_id="C1", level="l1",
        voter_count=1,
        cascade_signature=None, cascade_profile=None,
        voter_specs=VOTER_SPECS,
    )
    assert rec.best_voter_index == 2
    assert rec.selected_provider == "mistral"
    assert rec.selected_model == "mistral-large"


def test_b041_router_path_maps_back_to_original_indices():
    """Router survivor-indices get mapped back to original indices in ChunkOutcome.

    Survivor list = [v1, v2]; router strips survivor 0 as UNUSABLE, eligible=[1]
    (router's survivor index). Mapped through voter_indices=[1,2] → original [2].
    """
    from nlp_histo.pipeline.stages.knowledge_extraction.routing.models import GateOrigin, RoutingDecision

    class _FakeRouter:
        def route(self, voters, chunk, pmcid, source_text):
            # KEEP, but router strips survivor 0 as UNUSABLE; bundle.best_index=0
            # indexes into the post-router eligible subset = [voters[1]] = "v2".
            return RoutingDecision(
                decision=ChunkDecision.KEEP,
                gate_origin=GateOrigin.AGREEMENT_GATE,
                reason_codes=[],
                explanation="ok",
                agreement_details=ScoreBundle(
                    decision=ChunkDecision.KEEP, best_index=0, confidence=0.9
                ),
                valid_voter_indices=[1],
            )

    outcome = evaluate_chunk(
        ["v1", "v2"], chunk=[], pmcid="p", source_text="t",
        agreement=_AgreementStub(), router=_FakeRouter(),
        voter_indices=[1, 2],
    )
    assert outcome.valid_voter_indices == [2]
    assert outcome.best == "v2"
    assert producer_from_outcome(outcome, VOTER_SPECS) == ("mistral", "mistral-large")
