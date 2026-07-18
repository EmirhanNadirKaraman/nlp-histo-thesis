"""Tests for ``AgreementConfig.alignment_strategy`` (soft_max / greedy / hungarian).

``soft_max`` is the unchanged default (bidirectional many-to-one coverage with the
full tau/count_alpha/reuse_weight/contradiction_weight penalty set). ``greedy`` and
``hungarian`` are one-to-one matching modes scored by a precision/recall F1 over the
matched similarities, so over/under-production is penalised. ``tau`` is reused as the
minimum valid-pair similarity; the other penalty weights do not apply to one-to-one.
"""
import types

import numpy as np
import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.agreement.embedding import _align, _align_one_to_one
from nlp_histo.pipeline.stages.knowledge_extraction.agreement.semantic_scorer import SemanticAgreementScorer
from nlp_histo.pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig


def _emb_from_sim(mat):
    """emb_b = identity ⇒ emb_a @ emb_b.T = emb_a = mat, so the sim matrix == mat."""
    mat = np.asarray(mat, dtype=float)
    return mat, np.eye(mat.shape[1])


# one-to-one reducer math

def test_one_to_one_exact_match_is_one():
    a = b = np.eye(3)
    assert _align_one_to_one(a, b, 0.15, "greedy") == pytest.approx(1.0)
    assert _align_one_to_one(a, b, 0.15, "hungarian") == pytest.approx(1.0)


def test_one_to_one_extra_findings_lower_score():
    # 7 vs 3: 3 perfect matches + 4 extra A → precision-penalised F1 ≈ 0.6.
    a7 = np.vstack([np.eye(3), np.eye(3)[[0, 1, 2, 0]]])
    b3 = np.eye(3)
    for strat in ("greedy", "hungarian"):
        assert _align_one_to_one(a7, b3, 0.15, strat) == pytest.approx(0.6, abs=1e-6)


def test_one_to_one_empty_conventions():
    assert _align_one_to_one(np.zeros((0, 3)), np.zeros((0, 3)), 0.15, "greedy") == 1.0
    assert _align_one_to_one(np.zeros((0, 3)), np.eye(3), 0.15, "greedy") == 0.0
    assert _align_one_to_one(np.eye(3), np.zeros((0, 3)), 0.15, "hungarian") == 0.0


def test_greedy_differs_from_hungarian():
    # greedy: (a1,b1)=0.9 then (a2,b2)=0.1 → matched 1.0 → F1 0.5
    # hungarian: (a1,b2)+(a2,b1)=1.65 → F1 0.825
    ea, eb = _emb_from_sim([[0.9, 0.8], [0.85, 0.1]])
    g = _align_one_to_one(ea, eb, 0.05, "greedy")
    h = _align_one_to_one(ea, eb, 0.05, "hungarian")
    assert g == pytest.approx(0.5, abs=1e-6)
    assert h == pytest.approx(0.825, abs=1e-6)
    assert h > g


def test_tau_drops_below_threshold_pairs():
    # tau=0.15 zeroes the 0.1/0.05 pairs; only the 0.9 pair survives → F1 0.45.
    ea, eb = _emb_from_sim([[0.9, 0.1], [0.1, 0.05]])
    assert _align_one_to_one(ea, eb, 0.15, "hungarian") == pytest.approx(0.45, abs=1e-6)


def test_bad_strategy_raises():
    with pytest.raises(ValueError):
        _align_one_to_one(np.eye(2), np.eye(2), 0.15, "nope")


# soft_max backward-compat + the config thread reaches the scorer

def _embed_factory():
    cache: dict[str, np.ndarray] = {}

    def embed(claims):
        out = []
        for c in claims:
            if c not in cache:
                rng = np.random.default_rng(abs(hash(c)) % (2 ** 32))
                v = rng.standard_normal(8)
                cache[c] = v / np.linalg.norm(v)
            out.append(cache[c])
        return out

    return embed


def _voter(claims):
    return types.SimpleNamespace(
        findings=[types.SimpleNamespace(claim=c, category="IHC") for c in claims]
    )


def test_soft_max_is_default_and_unchanged():
    embed = _embed_factory()
    a, b = ["c1", "c2", "c3"], ["c1", "c2"]
    assert _align(embed, a, b) == _align(embed, a, b, alignment_strategy="soft_max")
    assert AgreementConfig().alignment_strategy == "soft_max"


def _matrix_01(kind, strat, embed, voters, **hybrid_kwargs):
    cfg_kwargs = {"scorer_kind": kind, "alignment_strategy": strat}
    if hybrid_kwargs:
        cfg_kwargs["hybrid"] = HybridConfig(**hybrid_kwargs)
    sc = SemanticAgreementScorer.from_agreement_config(
        AgreementConfig(**cfg_kwargs), embed_fn=embed
    )
    return float(sc._strategy.compute_matrix(voters)[0, 1])


def test_alignment_thread_reaches_embedding_compute_matrix():
    embed = _embed_factory()
    voters = [_voter(["c1", "c2", "c3", "c4", "c5"]), _voter(["c1", "c2"])]
    soft = _matrix_01("embedding", "soft_max", embed, voters)
    hung = _matrix_01("embedding", "hungarian", embed, voters)
    assert abs(soft - hung) > 1e-6  # alignment_strategy actually flows to compute_matrix


def test_alignment_thread_reaches_hybrid_via_inner_strategy():
    embed = _embed_factory()
    voters = [_voter(["c1", "c2", "c3", "c4", "c5"]), _voter(["c1", "c2"])]
    # w_embedding=1 isolates the embedding sub-signal so alignment is the only mover.
    soft = _matrix_01("hybrid", "soft_max", embed, voters,
                      w_category=0.0, w_embedding=1.0, w_entity=0.0, w_evidence=0.0)
    hung = _matrix_01("hybrid", "hungarian", embed, voters,
                      w_category=0.0, w_embedding=1.0, w_entity=0.0, w_evidence=0.0)
    assert abs(soft - hung) > 1e-6
