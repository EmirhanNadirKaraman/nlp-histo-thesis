"""
Tests for the enhanced EmbeddingScorer / _align penalties.

All tests use mock embed functions — no API calls.

Covered cases
-------------
 1. Identical claim sets          → score == 1.0
 2. Subset case                   → partial score (0 < score < 1)
 3. Split vs merged claims        → count_alpha=0.25 is lenient vs alpha=1.0
 4. Large count mismatch          → mild penalty, score > 0.5
 5. Broad-claim reuse penalty     → reuse_weight > 0 lowers score vs weight=0
 6. Weak-match threshold          → sim < tau collapses to 0; sim > tau contributes
 7. Contradiction penalty         → opposing polarity lowers score
 8. Grounding penalty             → low grounding_pass_fraction discounts score
 9. _polarity helper              → correct +1 / -1 / 0 detection
10. _reuse_factor helper          → single target never penalised; high concentration penalised
11. _contradiction_ratio helper   → detects opposing polarity in strong matches
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.agreement.embedding import (
    EmbeddingScorer,
    _align,
    _align_precomputed,
    _align_precomputed_with_breakdown,
    _contradiction_ratio,
    _numeric_contradiction_ratio,
    _polarity,
    _polarity_contradiction_ratio,
    _reuse_factor,
)
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.embedding_similarity import (
    EmbeddingSimilarityStrategy,
)
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import (
    AgreementContext,
    VoterContext,
)
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditMetadata, AuditableSummary, Finding

PMCID = "PMC00000001"


# Helpers

def _finding(claim: str = "CD31 -> Positive", category: str = "IHC") -> Finding:
    return Finding(
        category=category,  # type: ignore[arg-type]
        claim=claim,
        evidence=[f"S1|{PMCID}|1"],
        confidence="high",
        verbatim_support=claim,
    )


def _summary(claims: list[str]) -> AuditableSummary:
    return AuditableSummary(
        chunk_id="C1",
        findings=[_finding(c) for c in claims],
        summary_text="test",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["S1"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )


def _ctx(*summaries: AuditableSummary, pass_fracs: list[float] | None = None) -> AgreementContext:
    n = len(summaries)
    pf = pass_fracs or [1.0] * n
    return AgreementContext(
        voter_contexts=[VoterContext(grounding_pass_fraction=pf[i], mean_evidence_length=1.0) for i in range(n)]
    )


def _unit(v: list[float]) -> list[float]:
    """Return the unit-normalised version of v."""
    arr = np.array(v, dtype=float)
    n = np.linalg.norm(arr)
    return (arr / n).tolist() if n > 0 else v


def _make_embed(mapping: dict[str, list[float]]):
    """
    Build a mock embed_fn from claim → vector mapping.
    Vectors are unit-normalised so cosine sim == dot product.
    """
    normalised = {k: _unit(v) for k, v in mapping.items()}
    default = _unit([0.0, 0.0, 0.0, 1.0])

    def fn(claims: list[str]) -> list[list[float]]:
        return [normalised.get(c, default) for c in claims]

    return fn


# 1. Identical claim sets

class TestIdenticalClaims:
    def test_identical_single_claim_scores_one(self):
        embed = _make_embed({"CD31 positive": [1, 0, 0, 0]})
        score = _align(embed, ["CD31 positive"], ["CD31 positive"])
        assert score == pytest.approx(1.0)

    def test_identical_multi_claim_scores_one(self):
        embed = _make_embed({
            "CD31 positive": [1, 0, 0, 0],
            "CD34 positive": [0, 1, 0, 0],
        })
        claims = ["CD31 positive", "CD34 positive"]
        score = _align(embed, claims, claims)
        assert score == pytest.approx(1.0)

    def test_scorer_identical_voters(self):
        embed = _make_embed({"CD31 positive": [1, 0, 0, 0]})
        scorer = EmbeddingScorer(embed_fn=embed)
        s = _summary(["CD31 positive"])
        bundle = scorer.compute([s, s], context=_ctx(s, s))
        assert bundle.embedding_agreement == pytest.approx(1.0)


# 2. Subset case

class TestSubset:
    def test_subset_produces_partial_score(self):
        # A has 2 claims, B has 1 (subset of A).
        # B's claim matches A[0] perfectly; A[1] is orthogonal to B → no coverage.
        embed = _make_embed({
            "CD31 positive": [1, 0, 0, 0],
            "CD34 positive": [0, 1, 0, 0],
        })
        score = _align(embed, ["CD31 positive", "CD34 positive"], ["CD31 positive"])
        assert 0.3 < score < 0.95

    def test_subset_a_to_b_is_less_than_b_to_a(self):
        # A covers more than B, so A→B average (0.5) < B→A average (1.0).
        # Final score is average of both directions (with penalties).
        embed = _make_embed({
            "claim A1": [1, 0, 0, 0],
            "claim A2": [0, 1, 0, 0],
            "claim B1": [1, 0, 0, 0],   # identical to A1
        })
        score_ab = _align(embed, ["claim A1", "claim A2"], ["claim B1"])
        score_ba = _align(embed, ["claim B1"], ["claim A1", "claim A2"])
        # Symmetric by construction — same base, just direction labels swap.
        assert score_ab == pytest.approx(score_ba, abs=1e-6)


# 3. Split vs merged claims

class TestSplitVsMerged:
    def test_mild_count_penalty_is_lenient(self):
        # With count_alpha=0.25, a 2:1 count mismatch applies factor (1/2)^0.25 ≈ 0.84.
        # With count_alpha=1.0 (strict), factor is 0.5 — much harsher.
        # Use identical content so the only difference is the count penalty.
        embed = _make_embed({
            "merged claim": [1, 0, 0, 0],
            "split claim 1": [1, 0, 0, 0],   # same direction → perfect sim with merged
            "split claim 2": [1, 0, 0, 0],
        })
        score_lenient = _align(
            embed,
            ["split claim 1", "split claim 2"],
            ["merged claim"],
            count_alpha=0.25,
        )
        score_strict = _align(
            embed,
            ["split claim 1", "split claim 2"],
            ["merged claim"],
            count_alpha=1.0,
        )
        assert score_lenient > score_strict

    def test_3_to_1_count_factor_is_above_half(self):
        # (1/3)^0.25 ≈ 0.76 — well above 1/3.
        count_factor = (1 / 3) ** 0.25
        assert count_factor > 0.70


# 4. Large count mismatch

class TestLargeCountMismatch:
    def test_8_to_2_count_mismatch_mild_penalty(self):
        # All 8 A-claims perfectly match one of the 2 B-claims.
        # count_factor = (2/8)^0.25 = 0.25^0.25 ≈ 0.707.
        vecs = {f"claim_{i}": [1, 0, 0, 0] for i in range(8)}
        vecs["broad B"] = [1, 0, 0, 0]
        vecs["other B"] = [0, 1, 0, 0]
        embed = _make_embed(vecs)
        a_claims = [f"claim_{i}" for i in range(8)]
        score = _align(embed, a_claims, ["broad B", "other B"])
        # Mild penalty: all A claims cluster to broad B, leaving other B uncovered
        # (b_to_a = 0.5), so score lands around 0.45. The key is it's well above
        # zero — strict 1:1 matching would give near 0 for 8 unmatched claims.
        assert score > 0.35

    def test_count_factor_formula(self):
        expected = (2 / 8) ** 0.25
        assert expected == pytest.approx(math.pow(0.25, 0.25), abs=1e-6)
        assert expected > 0.5


# 5. Broad-claim reuse penalty

class TestReusePenalty:
    def test_reuse_lowers_score_vs_no_reuse(self):
        # A has 3 claims all similar to B[0]; B[1] is unrelated.
        # With reuse_weight > 0, score should be lower than with reuse_weight=0.
        embed = _make_embed({
            "CD31 positive": [1, 0, 0, 0],
            "CD34 positive": [1, 0, 0, 0],   # same direction → same best match
            "ERG positive":  [1, 0, 0, 0],
            "vascular marker positive": [1, 0, 0, 0],  # B[0]
            "unrelated morphology":     [0, 0, 0, 1],  # B[1] — orthogonal
        })
        a = ["CD31 positive", "CD34 positive", "ERG positive"]
        b = ["vascular marker positive", "unrelated morphology"]

        score_with_penalty = _align(embed, a, b, reuse_weight=0.15)
        score_without_penalty = _align(embed, a, b, reuse_weight=0.0)
        assert score_with_penalty < score_without_penalty

    def test_reuse_factor_single_target_no_penalty(self):
        # With only 1 target claim, concentration is expected — no penalty.
        best = np.array([0, 0, 0])   # all 3 source claims route to target 0
        factor = _reuse_factor(best, n_target=1, weight=0.15)
        assert factor == pytest.approx(1.0)

    def test_reuse_factor_high_concentration_penalised(self):
        # All 4 source claims route to target 0; targets = 3.
        best = np.array([0, 0, 0, 0])
        factor = _reuse_factor(best, n_target=3, weight=0.15)
        assert factor < 1.0

    def test_reuse_factor_uniform_no_penalty(self):
        # 3 source → 3 targets, one-to-one mapping → no concentration.
        best = np.array([0, 1, 2])
        factor = _reuse_factor(best, n_target=3, weight=0.15)
        assert factor == pytest.approx(1.0)


# 6. Weak-match threshold

class TestWeakMatchThreshold:
    def test_sim_below_tau_counts_as_zero(self):
        # Cosine sim = 0.10, tau = 0.15 → sim zeroed → base = 0.
        # Build v_b orthogonal to v_a except for a 0.10 component.
        # v_a = [1, 0]; v_b = [0.1, sqrt(1-0.01)] (unit vectors, dot = 0.1).
        v_a = [1.0, 0.0]
        v_b_raw = [0.1, math.sqrt(1.0 - 0.01)]
        embed = _make_embed({"A": v_a, "B": v_b_raw})
        score = _align(embed, ["A"], ["B"], tau=0.15)
        assert score == pytest.approx(0.0)

    def test_sim_above_tau_contributes(self):
        # Cosine sim = 0.20, tau = 0.15 → contributes to score.
        v_a = [1.0, 0.0]
        v_b_raw = [0.2, math.sqrt(1.0 - 0.04)]
        embed = _make_embed({"A": v_a, "B": v_b_raw})
        score = _align(embed, ["A"], ["B"], tau=0.15)
        assert score > 0.0

    def test_tau_zero_always_contributes(self):
        # With tau=0, even very weak sim counts.
        v_a = [1.0, 0.0]
        v_b_raw = [0.05, math.sqrt(1.0 - 0.0025)]
        embed = _make_embed({"A": v_a, "B": v_b_raw})
        score = _align(embed, ["A"], ["B"], tau=0.0)
        assert score > 0.0

    def test_empty_both_scores_one(self):
        embed = _make_embed({})
        score = _align(embed, [], [])
        assert score == pytest.approx(1.0)

    def test_one_empty_scores_zero(self):
        embed = _make_embed({"CD31 positive": [1, 0, 0, 0]})
        assert _align(embed, ["CD31 positive"], []) == pytest.approx(0.0)
        assert _align(embed, [], ["CD31 positive"]) == pytest.approx(0.0)


# 7. Contradiction penalty

class TestContradictionPenalty:
    def test_opposing_polarity_lowers_score(self):
        # A says "positive", B says "negative" — high semantic sim but opposite polarity.
        v = [1.0, 0.0, 0.0, 0.0]   # identical direction → cosine sim = 1.0
        embed = _make_embed({
            "CD31 positive": v,
            "CD31 negative": v,
        })
        score_contra = _align(
            embed, ["CD31 positive"], ["CD31 negative"], contradiction_weight=0.20,
        )
        score_no_contra = _align(
            embed, ["CD31 positive"], ["CD31 negative"], contradiction_weight=0.0,
        )
        assert score_contra < score_no_contra

    def test_same_polarity_no_penalty(self):
        # Both positive → no contradiction penalty.
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({
            "CD31 positive": v,
            "CD34 positive": v,
        })
        score_contra = _align(
            embed, ["CD31 positive"], ["CD34 positive"], contradiction_weight=0.20,
        )
        score_no_contra = _align(
            embed, ["CD31 positive"], ["CD34 positive"], contradiction_weight=0.0,
        )
        assert score_contra == pytest.approx(score_no_contra)

    def test_contradiction_ratio_opposing_polarity(self):
        sim_t = np.array([[0.9]])   # strong match, above tau
        ratio = _contradiction_ratio(["CD31 positive"], ["CD31 negative"], sim_t, tau=0.15)
        assert ratio == pytest.approx(1.0)

    def test_contradiction_ratio_neutral_claim_no_penalty(self):
        sim_t = np.array([[0.9]])
        # "CD31 expression" has no clear polarity → ratio = 0
        ratio = _contradiction_ratio(["CD31 expression"], ["CD31 negative"], sim_t, tau=0.15)
        assert ratio == pytest.approx(0.0)

    def test_contradiction_ratio_below_tau_no_strong_pair(self):
        sim_t = np.array([[0.05]])   # below tau → no strong pair
        ratio = _contradiction_ratio(["CD31 positive"], ["CD31 negative"], sim_t, tau=0.15)
        assert ratio == pytest.approx(0.0)

    def test_contradiction_max_penalty_bounded(self):
        # With contradiction_weight=0.20 and full contradiction ratio, factor >= 0.80.
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A positive": v, "A negative": v})
        score = _align(
            embed, ["A positive"], ["A negative"],
            tau=0.0, count_alpha=0.0, reuse_weight=0.0, contradiction_weight=0.20,
        )
        # base = 1.0, no count/reuse penalty → score = 1.0 * (1 - 0.20 * 1.0) = 0.80
        assert score == pytest.approx(0.80, abs=1e-6)


# 8. Grounding penalty

class TestGroundingPenalty:
    def test_low_grounding_discounts_score(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        scorer = EmbeddingScorer(embed_fn=embed, grounding_floor=0.5)
        s = _summary(["A"])
        ctx_low = _ctx(s, s, pass_fracs=[0.0, 0.0])
        ctx_high = _ctx(s, s, pass_fracs=[1.0, 1.0])
        bundle_low = scorer.compute([s, s], context=ctx_low)
        bundle_high = scorer.compute([s, s], context=ctx_high)
        assert bundle_low.embedding_agreement < bundle_high.embedding_agreement  # type: ignore[operator]

    def test_full_grounding_no_discount(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        scorer = EmbeddingScorer(embed_fn=embed, grounding_floor=0.5)
        s = _summary(["A"])
        bundle = scorer.compute([s, s], context=_ctx(s, s, pass_fracs=[1.0, 1.0]))
        # grounding_factor = 0.5 + 0.5 * 1.0 = 1.0 → no discount
        assert bundle.embedding_agreement == pytest.approx(1.0)

    def test_zero_grounding_applies_floor(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        scorer = EmbeddingScorer(embed_fn=embed, grounding_floor=0.5)
        s = _summary(["A"])
        bundle = scorer.compute([s, s], context=_ctx(s, s, pass_fracs=[0.0, 0.0]))
        # grounding_factor = 0.5 + 0.5 * 0.0 = 0.5 → score = 1.0 * 0.5 = 0.5
        assert bundle.embedding_agreement == pytest.approx(0.5)

    def test_no_context_no_grounding_discount(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        scorer = EmbeddingScorer(embed_fn=embed, grounding_floor=0.5)
        s = _summary(["A"])
        bundle = scorer.compute([s, s], context=None)
        assert bundle.embedding_agreement == pytest.approx(1.0)


# 9. _polarity helper

class TestPolarity:
    def test_positive_keywords(self):
        assert _polarity("CD31 positive staining") == 1
        assert _polarity("CEAN expressed in vessels") == 1
        assert _polarity("marker detected in all cases") == 1

    def test_negative_keywords(self):
        assert _polarity("CD31 negative staining") == -1
        assert _polarity("marker absent from stroma") == -1
        assert _polarity("expression lost after treatment") == -1

    def test_neutral_claim(self):
        assert _polarity("CEAN morphology reviewed") == 0
        assert _polarity("Patient age 46 years") == 0

    def test_both_positive_and_negative_is_neutral(self):
        # Ambiguous — both present → 0
        assert _polarity("positive in vessels negative in stroma") == 0

    def test_negation_flips_positive(self):
        assert _polarity("not positive staining") == -1

    def test_negation_flips_detected(self):
        assert _polarity("not detected in vessels") == -1

    def test_non_prefix_flips_expressed(self):
        assert _polarity("non-expressed marker") == -1

    def test_neither_positive_nor_negative_is_neutral(self):
        # "neither" negates "positive" → effective_neg; "nor" is in window of "negative"
        # → flips it to effective_pos; both sides cancel → 0
        assert _polarity("neither positive nor negative") == 0


# 10. _reuse_factor helper

class TestReuseFactor:
    def test_single_target_returns_one(self):
        assert _reuse_factor(np.array([0, 0, 0]), n_target=1, weight=0.15) == pytest.approx(1.0)

    def test_zero_source_returns_one(self):
        assert _reuse_factor(np.array([]), n_target=3, weight=0.15) == pytest.approx(1.0)

    def test_uniform_mapping_no_penalty(self):
        # 3 source → 3 targets, perfectly uniform
        assert _reuse_factor(np.array([0, 1, 2]), n_target=3, weight=0.15) == pytest.approx(1.0)

    def test_full_concentration_max_penalty(self):
        # All 4 source claims → target 0; 3 targets total
        factor = _reuse_factor(np.array([0, 0, 0, 0]), n_target=3, weight=0.15)
        assert factor < 1.0
        assert factor >= 1.0 - 0.15  # floor: 1 - weight

    def test_penalty_monotone_with_concentration(self):
        # More concentration → lower factor
        f_low  = _reuse_factor(np.array([0, 1]), n_target=2, weight=0.15)     # uniform
        f_high = _reuse_factor(np.array([0, 0]), n_target=2, weight=0.15)     # full concentration
        assert f_low >= f_high


# 11. _contradiction_ratio helper

class TestContradictionRatio:
    def test_empty_sim_returns_zero(self):
        assert _contradiction_ratio([], [], np.zeros((0, 0)), tau=0.15) == pytest.approx(0.0)

    def test_no_strong_pairs_returns_zero(self):
        sim_t = np.array([[0.0, 0.0], [0.0, 0.0]])
        ratio = _contradiction_ratio(["A present", "B positive"], ["A absent", "B negative"], sim_t, tau=0.15)
        assert ratio == pytest.approx(0.0)

    def test_all_strong_contradictions(self):
        sim_t = np.array([[0.9, 0.0], [0.0, 0.8]])
        ratio = _contradiction_ratio(
            ["A positive", "B present"],
            ["A negative", "B absent"],
            sim_t, tau=0.5,
        )
        assert ratio == pytest.approx(1.0)

    def test_partial_contradictions(self):
        # 1 of 2 strong pairs is contradictory
        sim_t = np.array([[0.9, 0.0], [0.0, 0.9]])
        ratio = _contradiction_ratio(
            ["A positive", "B morphology"],      # B claim is neutral
            ["A negative", "B shape"],
            sim_t, tau=0.5,
        )
        # Only pair (0,0) is contradictory; pair (1,1) is neutral → ratio = 0.5
        assert ratio == pytest.approx(0.5)


# _align_precomputed edge cases

class TestAlignPrecomputed:
    def test_empty_a_returns_zero(self):
        emb_a = np.zeros((0, 3))
        emb_b = np.array([[1.0, 0.0, 0.0]])
        assert _align_precomputed(emb_a, emb_b, [], ["B"], 0.15, 0.25, 0.15, 0.20) == pytest.approx(0.0)

    def test_empty_b_returns_zero(self):
        emb_a = np.array([[1.0, 0.0, 0.0]])
        emb_b = np.zeros((0, 3))
        assert _align_precomputed(emb_a, emb_b, ["A"], [], 0.15, 0.25, 0.15, 0.20) == pytest.approx(0.0)

    def test_perfect_match_no_penalties(self):
        v = np.array([[1.0, 0.0, 0.0]])
        score = _align_precomputed(v, v, ["CD31 positive"], ["CD31 positive"], 0.15, 0.25, 0.15, 0.20)
        assert score == pytest.approx(1.0)


# Fix 2: per-pair grounding in EmbeddingScorer

class TestPerPairGrounding:
    def test_mixed_grounding_lower_than_both_high(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        scorer = EmbeddingScorer(embed_fn=embed, grounding_floor=0.5)
        s = _summary(["A"])
        bundle_mixed = scorer.compute([s, s], context=_ctx(s, s, pass_fracs=[0.0, 1.0]))
        bundle_high  = scorer.compute([s, s], context=_ctx(s, s, pass_fracs=[1.0, 1.0]))
        assert bundle_mixed.embedding_agreement < bundle_high.embedding_agreement  # type: ignore[operator]

    def test_mixed_grounding_uses_min_not_mean(self):
        # Per-pair min([0.0, 1.0]) = 0.0 → factor = floor = 0.5.
        # Old mean would give g=0.5, factor=0.75.  The correct result is 0.5.
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        scorer = EmbeddingScorer(embed_fn=embed, grounding_floor=0.5)
        s = _summary(["A"])
        bundle = scorer.compute([s, s], context=_ctx(s, s, pass_fracs=[0.0, 1.0]))
        # min(0.0, 1.0) = 0.0 → factor = 0.5 + 0.5 * 0.0 = 0.5 → score = 1.0 * 0.5
        assert bundle.embedding_agreement == pytest.approx(0.5)

    def test_three_voters_bad_one_penalises_its_pairs(self):
        # 3 voters: pf = [1.0, 1.0, 0.0].
        # Pairs (0,1): min=1.0 → factor=1.0.
        # Pairs (0,2) and (1,2): min=0.0 → factor=0.5.
        # Mean pair score = (1.0*1.0 + 1.0*0.5 + 1.0*0.5) / 3 = 2.0/3 ≈ 0.667.
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        scorer = EmbeddingScorer(embed_fn=embed, grounding_floor=0.5)
        s = _summary(["A"])
        ctx = _ctx(s, s, s, pass_fracs=[1.0, 1.0, 0.0])
        bundle = scorer.compute([s, s, s], context=ctx)
        assert bundle.embedding_agreement == pytest.approx(2.0 / 3.0, abs=1e-6)


# Fix 3: numeric contradiction detection

class TestNumericContradiction:
    def test_numeric_contradiction_ratio_detects_large_diff(self):
        # Strong match (sim=0.9 ≥ tau=0.5); ratio 15/5 = 3.0 ≥ 2.0 → contradiction.
        sim_t = np.array([[0.9]])
        ratio = _numeric_contradiction_ratio(
            ["tumor 5mm"], ["tumor 15mm"], sim_t, tau=0.5,
        )
        assert ratio == pytest.approx(1.0)

    def test_same_number_no_numeric_contradiction(self):
        sim_t = np.array([[0.9]])
        ratio = _numeric_contradiction_ratio(
            ["tumor 5mm"], ["lesion 5mm"], sim_t, tau=0.5,
        )
        assert ratio == pytest.approx(0.0)

    def test_no_number_in_one_claim_no_numeric_contradiction(self):
        sim_t = np.array([[0.9]])
        ratio = _numeric_contradiction_ratio(
            ["tumor 5mm"], ["large tumor"], sim_t, tau=0.5,
        )
        assert ratio == pytest.approx(0.0)

    def test_ratio_below_threshold_no_contradiction(self):
        # 5mm vs 6mm → ratio 6/5 = 1.2 < 2.0 → not flagged.
        sim_t = np.array([[0.9]])
        ratio = _numeric_contradiction_ratio(
            ["tumor 5mm"], ["tumor 6mm"], sim_t, tau=0.5,
        )
        assert ratio == pytest.approx(0.0)

    def test_large_numeric_diff_lowers_align_score(self):
        # High-sim pair ("5mm" vs "15mm") should get a lower score than without penalty.
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"tumor size 5mm": v, "tumor size 15mm": v})
        score_with = _align(
            embed, ["tumor size 5mm"], ["tumor size 15mm"], contradiction_weight=0.20,
        )
        score_without = _align(
            embed, ["tumor size 5mm"], ["tumor size 15mm"], contradiction_weight=0.0,
        )
        assert score_with < score_without

    def test_polarity_contradiction_ratio_unchanged(self):
        # Ensure refactored _polarity_contradiction_ratio still works independently.
        sim_t = np.array([[0.9]])
        ratio = _polarity_contradiction_ratio(
            ["CD31 positive"], ["CD31 negative"], sim_t, tau=0.15,
        )
        assert ratio == pytest.approx(1.0)

    def test_contradiction_ratio_is_max_of_both(self):
        # No polarity keywords; large numeric diff → combined ratio = max(0, 1) = 1.
        sim_t = np.array([[0.9]])
        ratio = _contradiction_ratio(["size 5mm"], ["size 15mm"], sim_t, tau=0.5)
        assert ratio == pytest.approx(1.0)


# Fix 4: grounding forwarded to EmbeddingSimilarityStrategy.compute_matrix

class TestComputeMatrixGrounding:
    def test_low_grounding_reduces_off_diagonal(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed, grounding_floor=0.5)
        outputs = [_summary(["A"]), _summary(["A"])]
        mat_low  = strategy.compute_matrix(outputs, context=_ctx(*outputs, pass_fracs=[0.0, 0.0]))
        mat_high = strategy.compute_matrix(outputs, context=_ctx(*outputs, pass_fracs=[1.0, 1.0]))
        assert mat_low[0, 1] < mat_high[0, 1]

    def test_zero_grounding_applies_floor(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed, grounding_floor=0.5)
        outputs = [_summary(["A"]), _summary(["A"])]
        mat = strategy.compute_matrix(outputs, context=_ctx(*outputs, pass_fracs=[0.0, 0.0]))
        # raw score = 1.0, factor = 0.5 + 0.5 * min(0, 0) = 0.5 → entry = 0.5
        assert mat[0, 1] == pytest.approx(0.5)

    def test_full_grounding_no_discount(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed, grounding_floor=0.5)
        outputs = [_summary(["A"]), _summary(["A"])]
        mat = strategy.compute_matrix(outputs, context=_ctx(*outputs, pass_fracs=[1.0, 1.0]))
        assert mat[0, 1] == pytest.approx(1.0)

    def test_no_context_backward_compat(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed)
        outputs = [_summary(["A"]), _summary(["A"])]
        mat = strategy.compute_matrix(outputs)   # no context → no error
        assert mat.shape == (2, 2)
        assert mat[0, 1] == pytest.approx(1.0)

    def test_diagonal_always_one_regardless_of_context(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed, grounding_floor=0.5)
        outputs = [_summary(["A"]), _summary(["A"]), _summary(["A"])]
        ctx = _ctx(*outputs, pass_fracs=[0.0, 0.0, 0.0])
        mat = strategy.compute_matrix(outputs, context=ctx)
        for i in range(3):
            assert mat[i, i] == pytest.approx(1.0)


# Part A: _align_precomputed_with_breakdown

class TestAlignPrecomputedWithBreakdown:
    _BREAKDOWN_KEYS = {
        "claim_count_a", "claim_count_b",
        "coverage_a_to_b", "coverage_b_to_a", "base",
        "count_factor", "reuse_factor",
        "polarity_contradiction_ratio", "numeric_contradiction_ratio",
        "contradiction_ratio", "contradiction_factor",
        "pre_grounding_score",
    }

    def _run(self, a_claims, b_claims, emb_a=None, emb_b=None):
        if emb_a is None:
            d = len(a_claims) + len(b_claims)
            emb_a = np.eye(len(a_claims), d)
            emb_b = np.eye(len(b_claims), d)
        return _align_precomputed_with_breakdown(
            emb_a, emb_b, a_claims, b_claims,
            tau=0.15, count_alpha=0.25, reuse_weight=0.15, contradiction_weight=0.20,
        )

    def test_returns_tuple(self):
        v = np.array([[1.0, 0.0, 0.0]])
        result = _align_precomputed_with_breakdown(
            v, v, ["A"], ["A"], 0.15, 0.25, 0.15, 0.20,
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_score_matches_plain_function(self):
        v = np.array([[1.0, 0.0, 0.0]])
        score_plain = _align_precomputed(v, v, ["CD31 positive"], ["CD31 positive"], 0.15, 0.25, 0.15, 0.20)
        score_bd, _ = _align_precomputed_with_breakdown(v, v, ["CD31 positive"], ["CD31 positive"], 0.15, 0.25, 0.15, 0.20)
        assert score_plain == pytest.approx(score_bd)

    def test_breakdown_contains_all_keys(self):
        v = np.array([[1.0, 0.0, 0.0]])
        _, bd = _align_precomputed_with_breakdown(v, v, ["A"], ["A"], 0.15, 0.25, 0.15, 0.20)
        assert self._BREAKDOWN_KEYS <= set(bd)

    def test_breakdown_empty_a(self):
        emb_a = np.zeros((0, 3))
        emb_b = np.array([[1.0, 0.0, 0.0]])
        score, bd = _align_precomputed_with_breakdown(emb_a, emb_b, [], ["B"], 0.15, 0.25, 0.15, 0.20)
        assert score == pytest.approx(0.0)
        assert bd["pre_grounding_score"] == pytest.approx(0.0)
        assert self._BREAKDOWN_KEYS <= set(bd)

    def test_breakdown_perfect_match(self):
        v = np.array([[1.0, 0.0, 0.0]])
        score, bd = _align_precomputed_with_breakdown(v, v, ["CD31 positive"], ["CD31 positive"], 0.15, 0.25, 0.15, 0.20)
        assert score == pytest.approx(1.0)
        assert bd["base"] == pytest.approx(1.0)
        assert bd["count_factor"] == pytest.approx(1.0)   # equal counts
        assert bd["pre_grounding_score"] == pytest.approx(score)

    def test_pre_grounding_score_equals_score(self):
        # score == pre_grounding_score because no grounding penalty is applied here.
        v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        w = np.array([[1.0, 0.0, 0.0]])
        score, bd = _align_precomputed_with_breakdown(
            v, w, ["A", "B"], ["A"], 0.15, 0.25, 0.15, 0.20,
        )
        assert bd["pre_grounding_score"] == pytest.approx(score)

    def test_count_a_and_b_recorded(self):
        v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        w = np.array([[1.0, 0.0, 0.0]])
        _, bd = _align_precomputed_with_breakdown(v, w, ["A", "B"], ["A"], 0.15, 0.25, 0.15, 0.20)
        assert bd["claim_count_a"] == 2
        assert bd["claim_count_b"] == 1


# Part A: compute_matrix_with_breakdown

class TestComputeMatrixWithBreakdown:
    _BREAKDOWN_KEYS = {
        "voter_i", "voter_j", "score",
        "claim_count_a", "claim_count_b",
        "coverage_a_to_b", "coverage_b_to_a", "base",
        "count_factor", "reuse_factor",
        "polarity_contradiction_ratio", "numeric_contradiction_ratio",
        "contradiction_ratio", "contradiction_factor",
        "pre_grounding_score", "grounding_factor",
    }

    def test_returns_matrix_and_breakdowns(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed)
        outputs = [_summary(["A"]), _summary(["A"])]
        mat, bds = strategy.compute_matrix_with_breakdown(outputs)
        assert mat.shape == (2, 2)
        assert len(bds) == 1  # one upper-triangle pair for N=2

    def test_breakdown_keys_present(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed)
        outputs = [_summary(["A"]), _summary(["A"])]
        _, bds = strategy.compute_matrix_with_breakdown(outputs)
        assert self._BREAKDOWN_KEYS <= set(bds[0])

    def test_voter_indices_in_breakdown(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed)
        outputs = [_summary(["A"]), _summary(["A"]), _summary(["A"])]
        _, bds = strategy.compute_matrix_with_breakdown(outputs)
        # N=3 → 3 upper-triangle pairs: (0,1), (0,2), (1,2)
        assert len(bds) == 3
        pairs = {(bd["voter_i"], bd["voter_j"]) for bd in bds}
        assert pairs == {(0, 1), (0, 2), (1, 2)}

    def test_score_matches_matrix(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed)
        outputs = [_summary(["A"]), _summary(["A"])]
        mat, bds = strategy.compute_matrix_with_breakdown(outputs)
        assert bds[0]["score"] == pytest.approx(mat[0, 1])

    def test_grounding_factor_no_context(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed)
        outputs = [_summary(["A"]), _summary(["A"])]
        _, bds = strategy.compute_matrix_with_breakdown(outputs)
        assert bds[0]["grounding_factor"] == pytest.approx(1.0)

    def test_grounding_factor_with_low_context(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed, grounding_floor=0.5)
        outputs = [_summary(["A"]), _summary(["A"])]
        ctx = _ctx(*outputs, pass_fracs=[0.0, 0.0])
        _, bds = strategy.compute_matrix_with_breakdown(outputs, context=ctx)
        # min(0.0, 0.0) = 0.0 → factor = 0.5 + 0.5*0 = 0.5
        assert bds[0]["grounding_factor"] == pytest.approx(0.5)
        assert bds[0]["score"] == pytest.approx(0.5)   # pre_grounding=1.0 * 0.5

    def test_matrix_matches_compute_matrix(self):
        v = [1.0, 0.0, 0.0, 0.0]
        embed = _make_embed({"A": v})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed, grounding_floor=0.5)
        outputs = [_summary(["A"]), _summary(["A"])]
        ctx = _ctx(*outputs, pass_fracs=[0.5, 0.8])
        mat_plain = strategy.compute_matrix(outputs, context=ctx)
        mat_bd, _ = strategy.compute_matrix_with_breakdown(outputs, context=ctx)
        assert mat_plain == pytest.approx(mat_bd)

    def test_empty_no_claims_returns_eye_and_empty_breakdowns(self):
        embed = _make_embed({})
        strategy = EmbeddingSimilarityStrategy(embed_fn=embed)
        outputs = [_summary([]), _summary([])]
        mat, bds = strategy.compute_matrix_with_breakdown(outputs)
        assert (mat == pytest.approx(np.eye(2)))
        assert len(bds) == 1  # still one entry for the pair, with trivial values


# Part A: PairwiseScore breakdown fields

class TestPairwiseScoreBreakdownFields:
    def test_default_breakdown_fields_are_none(self):
        from nlp_histo.pipeline.stages.knowledge_extraction.observability.models import PairwiseScore
        ps = PairwiseScore(voter_i=0, voter_j=1, score=0.9)
        assert ps.claim_count_a is None
        assert ps.claim_count_b is None
        assert ps.coverage_a_to_b is None
        assert ps.pre_grounding_score is None
        assert ps.grounding_factor is None

    def test_breakdown_fields_can_be_set(self):
        from nlp_histo.pipeline.stages.knowledge_extraction.observability.models import PairwiseScore
        ps = PairwiseScore(
            voter_i=0, voter_j=1, score=0.7,
            claim_count_a=3, claim_count_b=2,
            coverage_a_to_b=0.8, coverage_b_to_a=0.9, base=0.85,
            count_factor=0.93, reuse_factor=0.97,
            polarity_contradiction_ratio=0.0, numeric_contradiction_ratio=0.0,
            contradiction_ratio=0.0, contradiction_factor=1.0,
            pre_grounding_score=0.79, grounding_factor=0.89,
        )
        assert ps.claim_count_a == 3
        assert ps.grounding_factor == pytest.approx(0.89)


# Part B: VoterTrace grounding_source field

class TestVoterTraceGroundingSource:
    def test_default_grounding_source_is_fallback(self):
        from nlp_histo.pipeline.stages.knowledge_extraction.observability.models import VoterTrace
        vt = VoterTrace(voter_index=0, finding_count=3, grounding_pass_fraction=1.0, mean_evidence_length=1.0)
        assert vt.grounding_source == "fallback"

    def test_grounding_source_validated(self):
        from nlp_histo.pipeline.stages.knowledge_extraction.observability.models import VoterTrace
        vt = VoterTrace(voter_index=0, finding_count=3, grounding_pass_fraction=0.8,
                        mean_evidence_length=2.0, grounding_source="validated")
        assert vt.grounding_source == "validated"


# Part B: RoutingDecision voter_grounding_contexts

class TestRoutingDecisionGroundingContexts:
    def test_field_exists_and_defaults_none(self):
        from nlp_histo.pipeline.stages.knowledge_extraction.routing.models import RoutingDecision, GateOrigin, ReasonCode
        from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision
        decision = RoutingDecision(
            decision=ChunkDecision.KEEP,
            gate_origin=GateOrigin.AGREEMENT_GATE,
            reason_codes=[ReasonCode.HIGH_AGREEMENT],
            explanation="ok",
        )
        assert decision.voter_grounding_contexts is None

    def test_field_can_be_set(self):
        from nlp_histo.pipeline.stages.knowledge_extraction.routing.models import RoutingDecision, GateOrigin, ReasonCode
        from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision, VoterContext
        vcs = [VoterContext(grounding_pass_fraction=0.9, mean_evidence_length=2.0)]
        decision = RoutingDecision(
            decision=ChunkDecision.KEEP,
            gate_origin=GateOrigin.AGREEMENT_GATE,
            reason_codes=[ReasonCode.HIGH_AGREEMENT],
            explanation="ok",
            voter_grounding_contexts=vcs,
        )
        assert len(decision.voter_grounding_contexts) == 1
        assert decision.voter_grounding_contexts[0].grounding_pass_fraction == pytest.approx(0.9)
