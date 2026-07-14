"""
Unit tests for the semantic agreement upgrade.

Covered cases
-------------
 1. LexicalSimilarityStrategy: identical → 1.0; disjoint → 0.0
 2. SemanticAgreementScorer N=1: best_index=0, confidence=1.0
 3. SemanticAgreementScorer N=2: score = pairwise; best_index respects quality
 4. SemanticAgreementScorer N=3: best_index = consensus winner, not most-findings
 5. Empty voter penalty: voter with 0 findings → avg_sim=0.0, never wins
 6. Tie-break: equal avg_sim resolved by grounding_pass_fraction first
 7. Router N_eligible=1 default → ESCALATE (not agreement gate)
 8. Router N_eligible=1 single_voter_policy="keep" → KEEP
 9. AgreementChecker.best() uses bundle.best_index when set
10. AgreementChecker.best() quality-key fallback (no bundle)
11. _cascade() REJECT path → escalation model called (not None)
12. EmbeddingSimilarityStrategy.compute_matrix(): single embed call for all voters
13. HybridStructuredSimilarity: weighted sum of four signals correct
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.agreement.category_jaccard import CategoryJaccardScorer
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.checker import AgreementChecker
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.embedding import EmbeddingScorer
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.embedding_similarity import (
    EmbeddingSimilarityStrategy,
)
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.hybrid_structured import (
    HybridStructuredSimilarity,
    _evidence_jaccard,
)
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.lexical_similarity import (
    LexicalSimilarityStrategy,
)
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.semantic_scorer import SemanticAgreementScorer
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import (
    AgreementContext,
    ChunkDecision,
    ScoreBundle,
    VoterContext,
)
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditMetadata, AuditableSummary, Finding
from nlp_histo.pipeline.stages.knowledge_extraction.routing.models import GateOrigin, ReasonCode
from nlp_histo.pipeline.stages.knowledge_extraction.routing.router import MapOutputRouter


# ── Shared helpers ──────────────────────────────────────────────────────────────

PMCID = "PMC10047158"


def _finding(
    claim: str = "CD31 -> Positive",
    evidence: list[str] | None = None,
    verbatim: str = "CD31 was positive in all cases.",
) -> Finding:
    return Finding(
        category="IHC",
        claim=claim,
        evidence=evidence if evidence is not None else [f"S1|{PMCID}|42"],
        confidence="high",
        verbatim_support=verbatim,
    )


def _summary(
    findings: list[Finding],
    chunk_id: str = "C1",
) -> AuditableSummary:
    return AuditableSummary(
        chunk_id=chunk_id,
        findings=findings,
        summary_text="Test.",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["1"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )


def _ctx(*summaries: AuditableSummary, pass_fracs=None, mean_evs=None) -> AgreementContext:
    """Build an AgreementContext aligned with the given summaries."""
    n = len(summaries)
    pf = pass_fracs or [1.0] * n
    me = mean_evs or [1.0] * n
    return AgreementContext(
        voter_contexts=[VoterContext(grounding_pass_fraction=pf[i], mean_evidence_length=me[i]) for i in range(n)]
    )


def _chunk(sentence: str = "CD31 was positive in all cases.") -> list[dict]:
    return [{"pmcid": PMCID, "text_element_id": 42, "sentence": sentence}]


# ── 1. LexicalSimilarityStrategy ────────────────────────────────────────────────

class TestLexicalSimilarityStrategy:
    def setup_method(self):
        self.strat = LexicalSimilarityStrategy()

    def test_identical_summaries_score_one(self):
        s = _summary([_finding(claim="CD31 positive")])
        assert self.strat.similarity(s, s) == pytest.approx(1.0)

    def test_disjoint_claims_score_zero(self):
        a = _summary([_finding(claim="CD31 positive")])
        b = _summary([_finding(claim="FOXP3 negative")])
        assert self.strat.similarity(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _summary([_finding(claim="CD31 positive staining")])
        b = _summary([_finding(claim="CD31 negative expression")])
        score = self.strat.similarity(a, b)
        assert 0.0 < score < 1.0

    def test_both_empty_score_one(self):
        a = _summary([])
        b = _summary([])
        assert self.strat.similarity(a, b) == pytest.approx(1.0)

    def test_one_empty_score_zero(self):
        a = _summary([_finding(claim="CD31 positive")])
        b = _summary([])
        assert self.strat.similarity(a, b) == pytest.approx(0.0)


# ── 2 & 5. SemanticAgreementScorer — N=1 and empty voter ────────────────────────

class TestSemanticAgreementScorerBaseCases:
    def setup_method(self):
        self.strat = LexicalSimilarityStrategy()
        self.scorer = SemanticAgreementScorer(self.strat)

    def test_n1_returns_keep_with_best_index_zero(self):
        s = _summary([_finding()])
        bundle = self.scorer.compute([s], context=_ctx(s))
        assert bundle.decision == ChunkDecision.KEEP
        assert bundle.best_index == 0
        assert bundle.confidence == pytest.approx(1.0)

    def test_empty_voter_causes_escalate_when_only_one_non_empty(self):
        # Only 1 non-empty voter after exclusion — scorer escalates (no peer to compare).
        s0 = _summary([_finding(claim="CD31 positive")])
        s1 = _summary([])
        bundle = self.scorer.compute([s0, s1], context=_ctx(s0, s1))
        assert bundle.decision == ChunkDecision.ESCALATE

    def test_empty_voter_never_wins_even_with_high_pass_fraction(self):
        # Same: 1 non-empty voter → ESCALATE regardless of quality signals.
        s0 = _summary([_finding(claim="CD31 positive")])
        s_empty = _summary([])
        ctx = AgreementContext(voter_contexts=[
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=1.0),
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=99.0),
        ])
        bundle = self.scorer.compute([s0, s_empty], context=ctx)
        assert bundle.decision == ChunkDecision.ESCALATE

    def test_all_empty_voters_escalate(self):
        # Zero non-empty voters → fewer than 2 candidates → ESCALATE.
        s0 = _summary([])
        s1 = _summary([])
        bundle = self.scorer.compute([s0, s1], context=_ctx(s0, s1))
        assert bundle.decision == ChunkDecision.ESCALATE

    def test_empty_voter_excluded_before_matrix_preserves_full_score(self):
        """
        Two identical non-empty voters + one empty voter.
        After excluding the empty voter, the two non-empty voters are compared
        directly and should score 1.0.  If the empty voter were included in the
        matrix it would dilute both non-empty voters' avg_sim to 0.5.
        """
        s0 = _summary([_finding(claim="alpha beta gamma")])
        s1 = _summary([_finding(claim="alpha beta gamma")])
        s2 = _summary([])
        bundle = self.scorer.compute([s0, s1, s2], context=_ctx(s0, s1, s2))
        assert bundle.confidence == pytest.approx(1.0)
        assert bundle.best_index in {0, 1}


# ── 3. SemanticAgreementScorer — N=2 ────────────────────────────────────────────

class TestSemanticAgreementScorerN2:
    def setup_method(self):
        self.strat = LexicalSimilarityStrategy()
        self.scorer = SemanticAgreementScorer(self.strat)

    def test_identical_voters_score_one(self):
        s = _summary([_finding(claim="CD31 positive staining")])
        bundle = self.scorer.compute([s, s], context=_ctx(s, s))
        assert bundle.confidence == pytest.approx(1.0)

    def test_disjoint_voters_score_zero(self):
        s_a = _summary([_finding(claim="CD31 positive")])
        s_b = _summary([_finding(claim="FOXP3 negative")])
        bundle = self.scorer.compute([s_a, s_b], context=_ctx(s_a, s_b))
        assert bundle.confidence == pytest.approx(0.0)

    def test_best_index_by_grounding_quality_not_finding_count(self):
        # Two voters; same avg_sim. Voter 1 has better grounding quality.
        # Voter 0: 3 findings, lower pass_frac.
        # Voter 1: 1 finding, higher pass_frac.
        s0 = _summary([_finding("A"), _finding("B"), _finding("C")])
        s1 = _summary([_finding("A")])
        ctx = AgreementContext(voter_contexts=[
            VoterContext(grounding_pass_fraction=0.5, mean_evidence_length=1.0),
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=1.0),
        ])
        bundle = self.scorer.compute([s0, s1], context=ctx)
        # Both voters agree perfectly; tie-break: s1 wins on pass_frac.
        assert bundle.best_index == 1


# ── 4. SemanticAgreementScorer — N=3, consensus winner ──────────────────────────

class TestSemanticAgreementScorerN3:
    def setup_method(self):
        self.strat = LexicalSimilarityStrategy()
        self.scorer = SemanticAgreementScorer(self.strat)

    def test_n3_best_is_consensus_winner_not_most_findings(self):
        """
        Voter 0: 1 finding "alpha beta gamma" (consensus with voter 1)
        Voter 1: 1 finding "alpha beta gamma" (consensus with voter 0)
        Voter 2: 3 findings (most findings, but disjoint vocabulary)
        Expected: best_index = 0 or 1 (consensus pair), not 2.
        """
        s0 = _summary([_finding(claim="alpha beta gamma")])
        s1 = _summary([_finding(claim="alpha beta gamma")])
        s2 = _summary([_finding("x"), _finding("y"), _finding("z")])
        bundle = self.scorer.compute([s0, s1, s2], context=_ctx(s0, s1, s2))
        # Voter 2 has the most findings but lowest avg_sim — must not win.
        assert bundle.best_index in {0, 1}

    def test_n3_confidence_is_max_avg_sim(self):
        """Deferral score = max of the per-voter average similarities."""
        s0 = _summary([_finding(claim="alpha beta")])
        s1 = _summary([_finding(claim="alpha beta")])
        s2 = _summary([_finding(claim="delta epsilon")])
        bundle = self.scorer.compute([s0, s1, s2], context=_ctx(s0, s1, s2))
        assert bundle.confidence is not None
        assert bundle.confidence > 0.0


# ── 6. Tie-break by grounding quality ───────────────────────────────────────────

class TestTieBreaking:
    def setup_method(self):
        # Use a strategy that returns the same score for every pair.
        # spec=["similarity"] ensures hasattr(strat, "compute_matrix") is False,
        # so SemanticAgreementScorer falls through to the pairwise path.
        strat = MagicMock(spec=["similarity"])
        strat.similarity.return_value = 0.5
        self.scorer = SemanticAgreementScorer(strat)

    def test_tie_broken_by_grounding_pass_fraction(self):
        s0 = _summary([_finding()])
        s1 = _summary([_finding()])
        ctx = AgreementContext(voter_contexts=[
            VoterContext(grounding_pass_fraction=0.5, mean_evidence_length=1.0),
            VoterContext(grounding_pass_fraction=0.9, mean_evidence_length=1.0),
        ])
        bundle = self.scorer.compute([s0, s1], context=ctx)
        assert bundle.best_index == 1  # s1 wins: higher pass_frac

    def test_tie_broken_by_mean_evidence_length(self):
        s0 = _summary([_finding()])
        s1 = _summary([_finding()])
        ctx = AgreementContext(voter_contexts=[
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=1.0),
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=3.0),
        ])
        bundle = self.scorer.compute([s0, s1], context=ctx)
        assert bundle.best_index == 1  # s1 wins: longer evidence chain

    def test_context_length_mismatch_raises(self):
        s0 = _summary([_finding()])
        s1 = _summary([_finding()])
        ctx = AgreementContext(voter_contexts=[
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=1.0),
            # only 1 context for 2 outputs — mismatch
        ])
        with pytest.raises(ValueError, match="must be the same length"):
            self.scorer.compute([s0, s1], context=ctx)


# ── 7 & 8. Router single eligible voter policy ──────────────────────────────────

class TestRouterSingleVoterPolicy:
    def _make_router(self, policy: str = "escalate") -> MapOutputRouter:
        mock_checker = MagicMock(spec=AgreementChecker)
        mock_checker.compute.return_value = ScoreBundle(
            confidence=0.9, decision=ChunkDecision.KEEP
        )
        mock_checker._scorer = MagicMock(spec=[])  # no compute_scored
        return MapOutputRouter(
            agreement_checker=mock_checker,
            single_voter_policy=policy,  # type: ignore[arg-type]
        )

    def test_single_eligible_voter_escalates_by_default(self):
        finding = _finding()
        outputs = [_summary([finding])]
        router = self._make_router(policy="escalate")
        decision = router.route(outputs, chunk=_chunk(), pmcid=PMCID, source_text="t")
        assert decision.decision == ChunkDecision.ESCALATE
        assert decision.gate_origin == GateOrigin.GROUNDING_GATE

    def test_single_eligible_voter_keep_when_policy_is_keep(self):
        finding = _finding()
        outputs = [_summary([finding])]
        router = self._make_router(policy="keep")
        decision = router.route(outputs, chunk=_chunk(), pmcid=PMCID, source_text="t")
        assert decision.decision == ChunkDecision.KEEP
        assert decision.gate_origin == GateOrigin.AGREEMENT_GATE
        assert ReasonCode.HIGH_AGREEMENT in decision.reason_codes
        assert decision.valid_voter_indices == [0]


# ── 9 & 10. AgreementChecker.best() ────────────────────────────────────────────

class TestAgreementCheckerBest:
    def setup_method(self):
        self.checker = AgreementChecker(scorer=MagicMock())

    def test_uses_bundle_best_index_when_set(self):
        s0 = _summary([_finding("A")])
        s1 = _summary([_finding("B"), _finding("C")])
        bundle = ScoreBundle(best_index=0)
        result = self.checker.best([s0, s1], bundle=bundle)
        # bundle says index 0, even though s1 has more findings
        assert result is s0

    def test_bundle_best_index_one(self):
        s0 = _summary([_finding("A")])
        s1 = _summary([_finding("B"), _finding("C")])
        bundle = ScoreBundle(best_index=1)
        result = self.checker.best([s0, s1], bundle=bundle)
        assert result is s1

    def test_no_bundle_uses_quality_key(self):
        # s0: 1 finding with 2-item evidence chain
        # s1: 3 findings with 1-item evidence chains each
        # Quality key = (mean_ev, n_findings): s0=(2.0, 1), s1=(1.0, 3)
        # s0 wins because mean evidence length is higher.
        s0 = _summary([_finding(evidence=[f"S1|{PMCID}|42", f"S1|{PMCID}|43"])])
        s1 = _summary([_finding("A"), _finding("B"), _finding("C")])
        result = self.checker.best([s0, s1])
        assert result is s0

    def test_no_bundle_falls_back_to_finding_count_when_ev_equal(self):
        s0 = _summary([_finding()])           # 1 finding, 1 evidence
        s1 = _summary([_finding(), _finding()])  # 2 findings, 1 evidence each
        result = self.checker.best([s0, s1])
        assert result is s1  # same mean_ev=1.0, s1 wins on finding count


# ── 11. _cascade() REJECT path → escalation model ───────────────────────────────

class TestCascadeRejectPath:
    def test_reject_calls_escalation_not_none(self):
        """Router REJECT must fall back to the escalation model, not drop chunk."""
        from nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage import MapStage

        mock_voter_llm = MagicMock()
        mock_escalation_llm = MagicMock()
        mock_router = MagicMock(spec=MapOutputRouter)

        escalation_result = _summary([_finding()])
        mock_escalation_llm_chain = MagicMock()
        mock_escalation_llm_chain.invoke.return_value = escalation_result

        # Patch build_map_chain so we can control voter and escalation outputs
        with patch(
            "nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage.build_map_chain"
        ) as mock_build:
            voter_chain = MagicMock()
            voter_chain.invoke.return_value = _summary([_finding()])
            # 1 L1 voter + 1 L2 voter + 1 L3 escalation = 3 build_map_chain calls
            mock_build.side_effect = [
                voter_chain,
                voter_chain,
                mock_escalation_llm_chain,
            ]

            stage = MapStage(
                voter_llms=[mock_voter_llm],
                level2_voter_llms=[mock_voter_llm],
                escalation_llm=mock_escalation_llm,
                router=mock_router,
            )

        # _cascade() expects _last_escalation_counts to be seeded by process();
        # initialize it directly so we can test _cascade() in isolation.
        stage._last_escalation_counts = {
            "total": 0, "l1_kept": 0, "l2_escalated": 0, "l2_kept": 0,
            "l3_escalated": 0, "l3_kept": 0, "finalized": 0, "dropped": 0,
        }

        # Router says REJECT
        from nlp_histo.pipeline.stages.knowledge_extraction.routing.models import (
            GateOrigin,
            RoutingDecision,
        )

        mock_router.route.return_value = RoutingDecision(
            decision=ChunkDecision.REJECT,
            gate_origin=GateOrigin.GROUNDING_GATE,
            reason_codes=[ReasonCode.FABRICATED_VERBATIM_SUPPORT],
            explanation="All voters unusable.",
        )

        chunk = [{"pmcid": PMCID, "text_element_id": 42, "sentence": "Test."}]
        result = stage._cascade(chunk, PMCID, "C1")

        # Must NOT be None — escalation was called
        assert result is not None
        mock_escalation_llm_chain.invoke.assert_called_once()


# ── 12. EmbeddingSimilarityStrategy.compute_matrix() ────────────────────────────

class TestEmbeddingSimilarityStrategyMatrix:
    def test_single_api_call_for_all_voters(self):
        """All voter claims batched into one embedding call."""
        embed_fn = MagicMock(return_value=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        strat = EmbeddingSimilarityStrategy(embed_fn=embed_fn)
        s0 = _summary([_finding(claim="A")])
        s1 = _summary([_finding(claim="B")])
        s2 = _summary([_finding(claim="A")])

        strat.compute_matrix([s0, s1, s2])

        # embed_fn should be called exactly once with all 3 claims concatenated
        assert embed_fn.call_count == 1
        called_texts = embed_fn.call_args[0][0]
        assert len(called_texts) == 3

    def test_matrix_is_symmetric_with_unit_diagonal(self):
        embed_fn = MagicMock(
            return_value=[
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        strat = EmbeddingSimilarityStrategy(embed_fn=embed_fn)
        s0 = _summary([_finding(claim="A"), _finding(claim="B")])
        s1 = _summary([_finding(claim="C"), _finding(claim="D")])
        matrix = strat.compute_matrix([s0, s1])

        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[1, 1] == pytest.approx(1.0)
        assert matrix[0, 1] == pytest.approx(matrix[1, 0])

    def test_empty_outputs_returns_eye(self):
        embed_fn = MagicMock()
        strat = EmbeddingSimilarityStrategy(embed_fn=embed_fn)
        s0 = _summary([])
        s1 = _summary([])
        matrix = strat.compute_matrix([s0, s1])
        np.testing.assert_array_equal(matrix, np.eye(2))
        embed_fn.assert_not_called()


# ── 13. HybridStructuredSimilarity signal decomposition ─────────────────────────

class TestHybridStructuredSimilarity:
    def _strat(self, w_cat=0.0, w_emb=0.0, w_ent=0.0, w_ev=1.0, embed_fn=None):
        return HybridStructuredSimilarity(
            embed_fn=embed_fn or MagicMock(return_value=[[1.0, 0.0], [1.0, 0.0]]),
            w_category=w_cat,
            w_embedding=w_emb,
            w_entity=w_ent,
            w_evidence=w_ev,
        )

    def test_evidence_overlap_signal_isolated(self):
        # Same te_id in both voters → evidence_overlap = 1.0
        s0 = _summary([_finding(evidence=[f"S1|{PMCID}|42"])])
        s1 = _summary([_finding(evidence=[f"S1|{PMCID}|42"])])
        strat = self._strat(w_ev=1.0)
        score = strat.similarity(s0, s1)
        assert score == pytest.approx(1.0)

    def test_evidence_overlap_disjoint(self):
        s0 = _summary([_finding(evidence=[f"S1|{PMCID}|42"])])
        s1 = _summary([_finding(evidence=[f"S1|{PMCID}|99"])])
        strat = self._strat(w_ev=1.0)
        score = strat.similarity(s0, s1)
        assert score == pytest.approx(0.0)

    def test_category_overlap_signal_isolated(self):
        s0 = _summary([_finding(claim="A")])  # category=IHC
        s1 = _summary([_finding(claim="A")])  # category=IHC
        # Use category-only weight, mock embedding to avoid API
        embed_fn = MagicMock(return_value=[[1.0, 0.0], [1.0, 0.0]])
        strat = HybridStructuredSimilarity(
            embed_fn=embed_fn, w_category=1.0, w_embedding=0.0,
            w_entity=0.0, w_evidence=0.0,
        )
        score = strat.similarity(s0, s1)
        assert score == pytest.approx(1.0)

    def test_weighted_sum_adds_up(self):
        # Use weights that don't hit embedding API and easy to verify.
        # Two identical summaries → each signal = 1.0 → weighted sum = 1.0.
        s = _summary([_finding(evidence=[f"S1|{PMCID}|42"])])
        embed_fn = MagicMock(return_value=[[1.0, 0.0], [1.0, 0.0]])
        strat = HybridStructuredSimilarity(
            embed_fn=embed_fn,
            w_category=0.25, w_embedding=0.40, w_entity=0.0, w_evidence=0.35,
        )
        score = strat.similarity(s, s)
        # category=1.0, embedding=1.0, evidence=1.0 → 0.25 + 0.40 + 0.35 = 1.0
        # (entity weight is 0.0 so scispaCy is not called)
        assert score == pytest.approx(1.0)

    def test_evidence_jaccard_helper(self):
        s0 = _summary([_finding(evidence=[f"S1|{PMCID}|10", f"S1|{PMCID}|20"])])
        s1 = _summary([_finding(evidence=[f"S1|{PMCID}|10", f"S1|{PMCID}|30"])])
        # intersection={10}, union={10,20,30} → 1/3
        assert _evidence_jaccard(s0, s1) == pytest.approx(1 / 3)


# ── 14. Legacy scorer Protocol compliance (context param) ───────────────────────

class TestLegacyScorerProtocolCompliance:
    """
    Regression guard for Gap 1 fix: legacy scorers must accept context= without
    crashing.  A regression would reintroduce the TypeError that broke production
    when AgreementChecker.compute() forwarded context as a third positional arg.
    """

    def _ctx(self, *summaries: AuditableSummary) -> AgreementContext:
        return AgreementContext(voter_contexts=[
            VoterContext(grounding_pass_fraction=1.0, mean_evidence_length=1.0)
            for _ in summaries
        ])

    def test_embedding_scorer_accepts_context_no_crash(self):
        embed_fn = MagicMock(return_value=[[1.0, 0.0], [1.0, 0.0]])
        scorer = EmbeddingScorer(embed_fn=embed_fn)
        checker = AgreementChecker(scorer=scorer, theta=0.5)
        s0 = _summary([_finding(claim="A")])
        s1 = _summary([_finding(claim="A")])
        # Must not raise TypeError
        bundle = checker.compute([s0, s1], context=self._ctx(s0, s1))
        assert bundle.decision is not None

    def test_category_jaccard_scorer_accepts_context_no_crash(self):
        scorer = CategoryJaccardScorer()
        checker = AgreementChecker(scorer=scorer, theta=0.5)
        s0 = _summary([_finding(claim="A")])
        s1 = _summary([_finding(claim="A")])
        bundle = checker.compute([s0, s1], context=self._ctx(s0, s1))
        assert bundle.decision is not None


# ── 15. SemanticAgreementScorer theta param ──────────────────────────────────────

class TestSemanticScorerTheta:
    def setup_method(self):
        self.strat = LexicalSimilarityStrategy()

    def test_theta_keep_when_score_above_threshold(self):
        # Identical voters → score = 1.0; theta=0.5 → KEEP
        scorer = SemanticAgreementScorer(self.strat, theta=0.5)
        s = _summary([_finding(claim="alpha beta gamma")])
        bundle = scorer.compute([s, s], context=_ctx(s, s))
        assert bundle.decision == ChunkDecision.KEEP

    def test_theta_escalate_when_score_below_threshold(self):
        # Disjoint voters → score = 0.0; theta=0.5 → ESCALATE
        scorer = SemanticAgreementScorer(self.strat, theta=0.5)
        s0 = _summary([_finding(claim="alpha beta")])
        s1 = _summary([_finding(claim="delta epsilon")])
        bundle = scorer.compute([s0, s1], context=_ctx(s0, s1))
        assert bundle.decision == ChunkDecision.ESCALATE

    def test_theta_none_does_not_set_decision(self):
        # theta=None (default) — scorer must NOT set bundle.decision
        scorer = SemanticAgreementScorer(self.strat, theta=None)
        s = _summary([_finding(claim="alpha beta")])
        bundle = scorer.compute([s, s], context=_ctx(s, s))
        assert bundle.decision is None


# ── 16. AgreementChecker.best() bounds check ─────────────────────────────────────

class TestCheckerBestBoundsCheck:
    def setup_method(self):
        self.checker = AgreementChecker(scorer=MagicMock())

    def test_best_index_out_of_bounds_falls_back_to_quality_key(self):
        """
        best_index=5 is out of bounds for a 2-item list.
        Should fall back to _quality_key rather than raising IndexError.
        """
        s0 = _summary([_finding(evidence=[f"S1|{PMCID}|42", f"S1|{PMCID}|43"])])
        s1 = _summary([_finding("A"), _finding("B"), _finding("C")])
        bundle = ScoreBundle(best_index=5)  # out of bounds
        result = self.checker.best([s0, s1], bundle=bundle)
        # Falls back to quality key: s0 has higher mean evidence length
        assert result is s0
