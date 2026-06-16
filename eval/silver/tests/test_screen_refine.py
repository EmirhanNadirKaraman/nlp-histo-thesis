"""Tests for the screen→refine harness (run_new_summarization_sweeps).

These verify the guardrails that can't be shown by a live calibration run:
  * `_refine_specs` sweeps only APPLICABLE weights (tau always; count_alpha/
    reuse/contradiction only under soft_max; hybrid blend under any alignment),
    and skips the default-blend duplicate;
  * `_select_finalists` is additive + capped (never drops the top-K), honours
    keep-within / crowded-bump, and the diversity safety add;
  * `_deviation` (the "simpler config" tie-break) skips inert/None weights.

No API, no cache — pure functions over fabricated rows.
"""
from __future__ import annotations

import pytest

from eval.silver.run_new_summarization_sweeps import (
    _DEFAULT_BLEND,
    _DEFAULTS,
    _MAX_FINALISTS,
    _deviation,
    _refine_specs,
    _select_finalists,
    _stamp_row,
)


def _rest(name: str) -> str:
    """The variant suffix after `emb__scorer__alignment__`."""
    return name.split("__", 3)[3]


# ── _refine_specs: applicable weights only ───────────────────────────────────

@pytest.mark.parametrize("align", ["greedy", "hungarian"])
def test_refine_embedding_one_to_one_sweeps_tau_only(align):
    specs = _refine_specs("gemini", "embedding", align)
    rests = [_rest(s.name) for s in specs]
    # only base + tau variants; the three soft-align knobs are NOT swept
    assert all(r == "base" or r.startswith("tau_") for r in rests)
    assert any(r.startswith("tau_") for r in rests)
    # and they stay at defaults on every spec, with the one-to-one alignment set
    for s in specs:
        assert s.weights.count_alpha == _DEFAULTS["count_alpha"]
        assert s.weights.reuse_weight == _DEFAULTS["reuse_weight"]
        assert s.weights.contradiction_weight == _DEFAULTS["contradiction_weight"]
        assert s.weights.alignment_strategy == align


def test_refine_embedding_soft_max_sweeps_all_soft_align():
    rests = [_rest(s.name) for s in _refine_specs("gemini", "embedding", "soft_max")]
    assert any(r.startswith("tau_") for r in rests)
    assert any(r.startswith("count_alpha_") for r in rests)
    assert any(r.startswith("reuse_weight_") for r in rests)
    assert any(r.startswith("contradiction_weight_") for r in rests)


def test_refine_hybrid_one_to_one_keeps_blend_drops_soft_align():
    rests = [_rest(s.name) for s in _refine_specs("openai", "hybrid", "hungarian")]
    assert any(r.startswith("blend_") for r in rests)     # blend shapes pre-alignment sim → swept
    assert any(r.startswith("tau_") for r in rests)
    assert not any(r.startswith("count_alpha_") for r in rests)
    assert not any(r.startswith("reuse_weight_") for r in rests)
    assert not any(r.startswith("contradiction_weight_") for r in rests)


@pytest.mark.parametrize("align", ["soft_max", "hungarian"])
def test_refine_hybrid_skips_default_blend_duplicate(align):
    specs = _refine_specs("gemini", "hybrid", align)
    assert not any(s.name.endswith("__blend_default") for s in specs)  # base already covers it
    assert any(s.name.endswith("__blend_balanced") for s in specs)     # non-default blends present


def test_refine_specs_namespaced_by_embedder():
    g = {s.name for s in _refine_specs("gemini", "embedding", "soft_max")}
    o = {s.name for s in _refine_specs("openai", "embedding", "soft_max")}
    assert g.isdisjoint(o)
    assert all(n.startswith("gemini__") for n in g)


# ── _select_finalists: additive, capped, diversity ───────────────────────────

def _sb(*rows):
    """rows: (emb, scorer_kind, align, metric[, escalate[, f1_optimal]])."""
    out = {}
    for t in rows:
        emb, sk, al, m = t[0], t[1], t[2], t[3]
        esc = t[4] if len(t) > 4 else 0.0
        f1 = t[5] if len(t) > 5 else m
        out[(emb, sk, al)] = {"strict_f1_optimal": m, "escalate_rate": esc, "f1_optimal": f1}
    return out


def test_select_finalists_topk_only():
    sb = _sb(("gemini", "embedding", "soft_max", 0.80),
             ("gemini", "hybrid", "soft_max", 0.70),
             ("openai", "embedding", "greedy", 0.60),
             ("openai", "hybrid", "hungarian", 0.50))
    fin = _select_finalists(sb, "strict_f1_optimal", top_k=3, keep_within=0.0)
    assert [s for s, _ in fin] == [
        ("gemini", "embedding", "soft_max"),
        ("gemini", "hybrid", "soft_max"),
        ("openai", "embedding", "greedy"),
    ]


def test_select_finalists_crowded_bumps_to_four():
    sb = _sb(("gemini", "embedding", "soft_max", 0.800),
             ("gemini", "hybrid", "soft_max", 0.795),
             ("openai", "embedding", "greedy", 0.790),
             ("openai", "hybrid", "hungarian", 0.785),
             ("gemini", "embedding", "greedy", 0.600))
    fin = _select_finalists(sb, "strict_f1_optimal", top_k=3, keep_within=0.02)
    structs = {s for s, _ in fin}
    assert len(fin) == 4                                       # crowded → eff K = 4
    assert ("gemini", "embedding", "greedy") not in structs   # the 0.60 is out of band
    assert all(r in ("top-k", "keep-within") for _, r in fin)


def test_select_finalists_diversity_adds_other_embedder():
    sb = _sb(("gemini", "embedding", "soft_max", 0.800),
             ("gemini", "hybrid", "soft_max", 0.790),
             ("gemini", "embedding", "hungarian", 0.785),
             ("openai", "embedding", "soft_max", 0.770),   # gap 0.03 ≤ 2×keep_within; new embedder
             ("gemini", "hybrid", "greedy", 0.600))
    fin = dict(_select_finalists(sb, "strict_f1_optimal", top_k=3, keep_within=0.02))
    assert ("openai", "embedding", "soft_max") in fin
    assert fin[("openai", "embedding", "soft_max")] == "diversity:embedder"


def test_select_finalists_additive_caps_and_keeps_topk():
    keys = [("gemini", "embedding", "soft_max"), ("openai", "embedding", "soft_max"),
            ("gemini", "hybrid", "soft_max"), ("openai", "hybrid", "soft_max"),
            ("gemini", "embedding", "greedy"), ("openai", "embedding", "greedy"),
            ("gemini", "hybrid", "hungarian"), ("openai", "hybrid", "hungarian")]
    sb = {k: {"strict_f1_optimal": 0.80 - i * 0.001, "escalate_rate": 0.0, "f1_optimal": 0.80}
          for i, k in enumerate(keys)}
    fin = _select_finalists(sb, "strict_f1_optimal", top_k=3, keep_within=0.02)
    assert len(fin) <= _MAX_FINALISTS              # capped
    assert keys[0] in {s for s, _ in fin}          # genuine best never dropped


def test_select_finalists_empty():
    assert _select_finalists({}, "strict_f1_optimal", 3, 0.02) == []


# ── _stamp_row: scorer_kind/variant split + inert-weight blanking ────────────

def test_stamp_row_one_to_one_splits_names_and_blanks_inert():
    name = "gemini__hybrid__hungarian__base"
    r = {"scorer": name, "tau": 0.15, "count_alpha": 0.25,
         "reuse_weight": 0.15, "contradiction_weight": 0.20}
    _stamp_row(r, "gemini", {name: "hybrid"}, {name: "hungarian"})
    assert r["scorer_kind"] == "hybrid"            # family kind, separate from…
    assert r["variant"] == name                    # …the concrete variant name
    assert r["embedder"] == "gemini"
    assert r["alignment_strategy"] == "hungarian"
    assert r["count_alpha"] is None and r["reuse_weight"] is None
    assert r["contradiction_weight"] is None       # inert under one-to-one → blanked
    assert r["tau"] == 0.15                         # tau survives (it applies)


def test_stamp_row_soft_max_keeps_soft_align():
    name = "gemini__embedding__soft_max__base"
    r = {"scorer": name, "tau": 0.15, "count_alpha": 0.25,
         "reuse_weight": 0.15, "contradiction_weight": 0.20}
    _stamp_row(r, "gemini", {name: "embedding"}, {name: "soft_max"})
    assert r["scorer_kind"] == "embedding"
    assert r["count_alpha"] == 0.25 and r["reuse_weight"] == 0.15
    assert r["contradiction_weight"] == 0.20       # NOT blanked under soft_max


# ── _deviation: the "simpler config" tie-break ───────────────────────────────

def test_deviation_default_row_is_zero():
    row = {
        "embedder": "gemini", "scorer_kind": "embedding", "alignment_strategy": "soft_max",
        "tau": _DEFAULTS["tau"], "count_alpha": _DEFAULTS["count_alpha"],
        "reuse_weight": _DEFAULTS["reuse_weight"], "contradiction_weight": _DEFAULTS["contradiction_weight"],
        "w_category": _DEFAULT_BLEND[0], "w_embedding": _DEFAULT_BLEND[1],
        "w_entity": _DEFAULT_BLEND[2], "w_evidence": _DEFAULT_BLEND[3],
    }
    assert _deviation(row) == 0


def test_deviation_counts_nondefaults():
    # openai (+1), hungarian (+1), tau≠default (+1); missing weights skipped
    row = {"embedder": "openai", "scorer_kind": "embedding",
           "alignment_strategy": "hungarian", "tau": 0.30}
    assert _deviation(row) == 3


def test_deviation_skips_none_inert_weights():
    # one-to-one row with blanked soft-align weights → only the alignment deviates
    row = {"embedder": "gemini", "scorer_kind": "embedding", "alignment_strategy": "hungarian",
           "tau": _DEFAULTS["tau"], "count_alpha": None, "reuse_weight": None,
           "contradiction_weight": None}
    assert _deviation(row) == 1
