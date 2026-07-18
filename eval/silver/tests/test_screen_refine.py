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

from eval.silver.analysis.run_new_summarization_sweeps import (
    _DEFAULT_BLEND,
    _DEFAULTS,
    _deviation,
    _refine_specs,
    _select_finalists,
    _stamp_row,
)


def _rest(name: str) -> str:
    """The variant suffix after `emb__scorer__alignment__`."""
    return name.split("__", 3)[3]


# _refine_specs: applicable weights only

@pytest.mark.parametrize("align", ["greedy", "hungarian"])
def test_refine_embedding_one_to_one_sweeps_tau_only(align):
    specs = _refine_specs("gemini", "embedding", align)
    rests = [_rest(s.name) for s in specs]
    # only base + tau variants; the three soft-align knobs are not swept
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


# _select_finalists: CELL-LEVEL Pareto (quality + economy + knee)

def _cells(*specs):
    """specs: (emb, scorer_kind, align, theta, strict_f1, cost). Multi-θ cells —
    the cost axis is ``_cost_frac`` (price-weighted L2+L3 escalation, θ-driven), so the
    fixtures must have >1 cell/structure. ``cost`` is the intended ``_cost_frac`` in [0,1];
    we synthesise n_chunks/n_l2_invoked/n_l3_invoked so ``_cost_frac(row) == cost`` exactly
    (l2 == l3 == cost·n_chunks makes the price weights cancel). ``escalate_rate`` is kept as
    a reported column but is not the cost axis (see ``_cost_frac`` docstring)."""
    n = 100
    return [{"embedder": e, "scorer_kind": s, "alignment_strategy": a, "theta": t,
             "reject_theta": 0.1, "strict_f1_optimal": f1, "f1_optimal": f1,
             "escalate_rate": cost,
             "n_chunks": n, "n_l2_invoked": cost * n, "n_l3_invoked": cost * n}
            for e, s, a, t, f1, cost in specs]


# mirrors the real screen: hybrid wins on F1 at θ0.9; embedding owns the cheap frontier
SCREEN = _cells(
    ("gemini", "hybrid",    "soft_max", 0.90, 0.71, 0.97),   # quality corner
    ("gemini", "hybrid",    "soft_max", 0.70, 0.67, 0.90),
    ("gemini", "embedding", "soft_max", 0.90, 0.69, 0.92),   # max-F1 cell DOMINATED by hybrid θ0.9
    ("gemini", "embedding", "soft_max", 0.80, 0.58, 0.65),   # economy cell (f1 ≥ floor)
    ("gemini", "embedding", "soft_max", 0.70, 0.41, 0.19),   # cheapest globally (below floor)
    ("openai", "hybrid",    "greedy",   0.90, 0.60, 0.98),   # dominated at every cell
    ("openai", "hybrid",    "greedy",   0.70, 0.50, 0.95),
)


def _fin(cells=SCREEN, **kw):
    base = dict(metric="strict_f1_optimal", top_k=2, keep_within=0.02)
    base.update(kw)
    return {s: (reasons, ev) for s, reasons, ev in _select_finalists(cells, **base)}


def test_economy_structure_survives_even_though_its_maxf1_cell_is_dominated():
    # REGRESSION GUARD: gemini/embedding's max-F1 cell (θ0.9, 0.69/0.92) is dominated by
    # hybrid θ0.9 — collapsing to per-structure-best would drop it. Its cheap cells are on
    # the cell-level frontier, so it must stay a finalist. Fails on the collapsed design.
    fin = _fin()
    s = ("gemini", "embedding", "soft_max")
    assert s in fin
    reasons, _ = fin[s]
    assert "pareto" in reasons and "economy" in reasons


def test_dominated_structure_excluded():
    assert ("openai", "hybrid", "greedy") not in _fin()      # dominated at every cell


def test_economy_respects_quality_floor_and_shows_cheap_cell():
    _, ev = _fin()[("gemini", "embedding", "soft_max")]
    econ = ev["economy"]
    assert float(econ["strict_f1_optimal"]) >= 0.50          # not the 0.41 θ0.7 cell
    assert abs(float(econ["theta"]) - 0.80) < 1e-9           # the θ0.8 economy cell


def test_knee_picks_best_tradeoff_cell():
    reasons, _ = _fin()[("gemini", "hybrid", "soft_max")]
    assert "knee" in reasons                                 # max(f1 − 0.2·esc) at λ=0.2


def test_topk_is_quality_corner():
    reasons, _ = _fin()[("gemini", "hybrid", "soft_max")]
    assert "top-k" in reasons


def test_return_shape_is_struct_reasons_evidence():
    for struct, reasons, evidence in _select_finalists(SCREEN, "strict_f1_optimal", 2, 0.02):
        assert isinstance(struct, tuple) and isinstance(reasons, list) and isinstance(evidence, dict)


def test_selection_ignores_greedy_metric():
    poisoned = [{**c, "strict_f1_greedy": 0.99, "f1_greedy": 0.99} for c in SCREEN]
    assert set(_fin()) == set(_fin(cells=poisoned))          # greedy must not move selection


def test_select_finalists_empty():
    assert _select_finalists([], "strict_f1_optimal", 2, 0.02) == []


# _stamp_row: scorer_kind/variant split + inert-weight blanking

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
    assert r["contradiction_weight"] == 0.20       # Not blanked under soft_max


# _deviation: the "simpler config" tie-break

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
