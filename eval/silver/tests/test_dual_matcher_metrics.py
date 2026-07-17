"""Tests for dual-matcher (greedy + optimal/Hungarian) metric reporting and the
optimal-primary winner selection in the calibration harness.

Covers the metric-handling contract:
  * the two matchers genuinely produce different matched sets;
  * the sweep eval path reports BOTH metric sets (``*_optimal`` + ``*_greedy``);
  * legacy ``f1``/``strict_f1`` keys alias the greedy matcher (no silent mislabel);
  * the harness selects winners on ``strict_f1_optimal``, never on greedy.

No API calls — embeddings are mocked.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from nlp_histo.evaluation.matching.matcher import greedy_match, optimal_match
from eval.silver.analysis.pipeline_sweep import _evaluate_outputs
from nlp_histo.evaluation.schemas import (
    PipelineCaseOutput,
    PipelineFinding,
    SilverCaseResult,
    SilverFinding,
)


# ── helpers (mirror test_matcher.py) ─────────────────────────────────────────

def _silver(case_id="PMC1|1", claims=("s0", "s1")) -> SilverCaseResult:
    findings = [SilverFinding(category="IHC", claim=c, confidence="high") for c in claims]
    return SilverCaseResult(
        case_id=case_id, pmcid="PMC1", te_id=1,
        prompt_version="v1", model="claude-opus-4-5", findings=findings,
    )


def _pipeline(case_id="PMC1|1", claims=("p0", "p1")) -> PipelineCaseOutput:
    findings = [
        PipelineFinding(pipeline_run_id=1, run_id="run1", pmcid="PMC1",
                        chunk_id="C1", category="IHC", claim=c)
        for c in claims
    ]
    return PipelineCaseOutput(
        case_id=case_id, pmcid="PMC1", te_id=1,
        run_id="run1", pipeline_run_id=1, findings=findings,
    )


# ── matcher strategies produce different matched sets ────────────────────────

def test_greedy_and_optimal_matched_sets_differ():
    # greedy grabs the global max (0,0)=0.9 then is stuck (only 0.0 left < thr);
    # optimal recovers both by taking (0,1)+(1,0)=1.2 instead.
    sim = [[0.9, 0.6], [0.6, 0.0]]
    g = {p[:2] for p in greedy_match(sim, 0.5, 2, 2)}
    o = {p[:2] for p in optimal_match(sim, 0.5, 2, 2)}
    assert g == {(0, 0)}
    assert o == {(0, 1), (1, 0)}
    assert g != o


# ── _evaluate_outputs reports both matcher metric sets ───────────────────────

@patch("nlp_histo.evaluation.matching.matcher.get_embeddings")
def test_evaluate_outputs_reports_both_matchers(mock_emb):
    # Normalised cosines:  s0·p0=.80 s0·p1=.55 | s1·p0=.60 s1·p1=.10  (thr=.5)
    #   greedy: takes (s0,p0)=.80, then (s1,p1)=.10 < .5 → 1 match
    #   optimal: (s0,p1)+(s1,p0)=.55+.60=1.15 beats (s0,p0)+(s1,p1)=.90 → 2 matches
    vecs = [
        [0.80, 0.55, 0.2398],  # s0
        [0.60, 0.10, 0.7937],  # s1
        [1.0, 0.0, 0.0],       # p0
        [0.0, 1.0, 0.0],       # p1
    ]
    mock_emb.side_effect = lambda texts, embedder, cache: vecs
    silver = _silver()
    pipeline = _pipeline()

    out = _evaluate_outputs(
        [pipeline], {silver.case_id: silver},
        embedder=None, embed_cache=None, sim_threshold=0.5,
    )

    # both explicit metric sets are present
    for k in ("precision_optimal", "recall_optimal", "f1_optimal",
              "strict_f1_optimal", "n_matched_optimal",
              "precision_greedy", "recall_greedy", "f1_greedy",
              "strict_f1_greedy", "n_matched_greedy"):
        assert k in out, f"missing {k}"

    # the matchers genuinely diverge on this case
    assert out["n_matched_greedy"] == 1
    assert out["n_matched_optimal"] == 2
    assert out["f1_optimal"] > out["f1_greedy"]
    assert out["f1_optimal"] == pytest.approx(1.0)
    assert out["f1_greedy"] == pytest.approx(0.5)

    # legacy keys alias the GREEDY matcher (historical default — not mislabeled)
    assert out["f1"] == out["f1_greedy"]
    assert out["strict_f1"] == out["strict_f1_greedy"]
    assert out["precision"] == out["precision_greedy"]
    assert out["recall"] == out["recall_greedy"]
    assert out["n_matched"] == out["n_matched_greedy"]


# ── harness selects on strict_f1_optimal, never greedy ───────────────────────

def test_rank_selects_on_optimal_not_greedy():
    from eval.silver.analysis.run_new_summarization_sweeps import _rank
    # Row A wins on greedy but loses on optimal; selection must pick B.
    a = {"strict_f1_optimal": 0.60, "f1_optimal": 0.62,
         "strict_f1_greedy": 0.90, "f1_greedy": 0.91, "escalate_rate": 0.0}
    b = {"strict_f1_optimal": 0.70, "f1_optimal": 0.71,
         "strict_f1_greedy": 0.50, "f1_greedy": 0.51, "escalate_rate": 0.0}
    best = max([a, b], key=lambda r: _rank(r, "strict_f1_optimal"))
    assert best is b


def test_rank_tiebreaks_cost_then_f1_optimal():
    # Cost axis is _cost_frac (price-weighted L2+L3 escalation, from n_chunks/
    # n_l2_invoked/n_l3_invoked), NOT bare escalate_rate — see _cost_frac docstring.
    from eval.silver.analysis.run_new_summarization_sweeps import _rank
    # Primary ties → LOWER cost_frac wins (even with worse f1_optimal).
    lo_cost = {"strict_f1_optimal": 0.70, "f1_optimal": 0.60,
               "n_chunks": 100, "n_l2_invoked": 10, "n_l3_invoked": 10}   # cost_frac 0.1
    hi_cost = {"strict_f1_optimal": 0.70, "f1_optimal": 0.99,
               "n_chunks": 100, "n_l2_invoked": 40, "n_l3_invoked": 40}   # cost_frac 0.4
    assert _rank(lo_cost, "strict_f1_optimal") > _rank(hi_cost, "strict_f1_optimal")
    # Primary + cost tie → higher f1_optimal wins.
    x = {"strict_f1_optimal": 0.70, "f1_optimal": 0.80,
         "n_chunks": 100, "n_l2_invoked": 10, "n_l3_invoked": 10}
    y = {"strict_f1_optimal": 0.70, "f1_optimal": 0.60,
         "n_chunks": 100, "n_l2_invoked": 10, "n_l3_invoked": 10}
    assert _rank(x, "strict_f1_optimal") > _rank(y, "strict_f1_optimal")


def test_selection_metric_choices_exclude_greedy():
    """The harness --metric flag must not offer any greedy metric for selection."""
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "analysis" / "run_new_summarization_sweeps.py"
    tree = ast.parse(src.read_text())
    # find the --metric add_argument call and read its choices=[...]
    choices = None
    default = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--metric"):
            for kw in node.keywords:
                if kw.arg == "choices":
                    choices = [e.value for e in kw.value.elts]
                if kw.arg == "default":
                    default = kw.value.value
    assert choices is not None, "--metric add_argument not found"
    assert default == "strict_f1_optimal"
    assert all("greedy" not in c for c in choices)
    assert all(c.endswith("_optimal") for c in choices)
