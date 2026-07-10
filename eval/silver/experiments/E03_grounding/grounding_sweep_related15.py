#!/usr/bin/env python3
"""E03 — grounding-threshold sweep on related15, offline and FREE.

Reuses the MAP you already paid for instead of re-priming: replays the FROZEN-config
cascade (E05–E08 winner: gemini/hybrid/greedy, entity-heavy blend, θ=0.9/reject=0.1,
all 6 voters) over the related15 voter_cache, scores every MAP finding with the
pipeline's OWN grounding NLI (PubMedBERT-MNLI-MedNLI, premise=verbatim_support →
hyp=claim), then sweeps the grounding threshold and scores survivors vs silver.

No prime files and no API calls (cascade selection from cached voter outputs,
grounding is a local cross-encoder), but the DB must be reachable and populated:
step 1b swaps each finding's LLM-paraphrased verbatim_support for the real cited
paragraph from the DB (mirroring production runner.py) so grounding scores real
source text, not paraphrases. This collapses pipeline_sweep's
prime+grounding into one pass, reusing _replay + the SAME _apply_grounding_threshold
/ _evaluate_outputs / _write_csv that the stock sweep uses (no schema/filename risk).

Why not pipeline_sweep prime --source related15: that re-runs MAP (paid), and the
out/summaries cache is the wrong cascade profile (haiku_only) + already filtered.

Usage:
  python -m eval.silver.experiments.E03_grounding.grounding_sweep_related15
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.map_theta_sweep import (
    REPORTS_DIR,
    ScorerSpec,
    _build_scorer,
    _replay,
    _write_csv,
)
from eval.silver.matcher import SIMILARITY_THRESHOLD
from eval.silver.pipeline_sweep import _apply_grounding_threshold, _evaluate_outputs
from eval.silver.map_context import _load_map_context
from eval.silver.run_new_summarization_sweeps import (
    BEST_FORCE_ESCALATE_POLARITY, BEST_REJECT_THETA, BEST_SINGLE_VOTER_POLICY,
    BEST_THETA, BEST_VOTER_SUBSET, _filtered_voter_cache,
)
from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig
from pipeline.stages.knowledge_extraction.helpers.grounding_filter import GroundingFilter, _score_pairs

# Frozen MAP config = the shipped 5-voter / escalate pins (BEST_*).
THETA, REJECT = BEST_THETA, BEST_REJECT_THETA
# Grounding thresholds to sweep. None = filter off (keep everything).
GRID: list[float | None] = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# 5-voter MAP finding count at θ0.9/reject0.2 (threshold None), 2026-06-22. (6-voter was 2280.)
EXPECTED_N_PIPELINE = 2294

_CSV_FIELDS = [
    "grounding_threshold", "n_kept", "retention",
    "strict_f1_optimal", "f1_optimal", "precision_optimal", "recall_optimal",
    "strict_f1_greedy", "f1_greedy",
    "n_silver", "n_pipeline",
]


def _frozen_spec() -> ScorerSpec:
    cfg = AgreementConfig(
        scorer_kind="hybrid", alignment_strategy="greedy",
        tau=0.15, count_alpha=0.25, reuse_weight=0.15, contradiction_weight=0.20,
        hybrid=HybridConfig(w_category=0.15, w_embedding=0.30, w_entity=0.50, w_evidence=0.05),
    )
    return ScorerSpec("hybrid", "hybrid", cfg)


def _distribution(scores: list[float]) -> None:
    if not scores:
        print("  (no scores)")
        return
    s = sorted(scores)
    n = len(s)
    def pct(p): return s[min(n - 1, int(p * n))]
    def below(t): return sum(1 for x in s if x < t)
    print(f"  n={n}  min={s[0]:.3f}  p10={pct(.10):.3f}  p25={pct(.25):.3f}  "
          f"median={pct(.50):.3f}  p75={pct(.75):.3f}  max={s[-1]:.3f}")
    print(f"  below 0.3: {below(0.3)} ({below(0.3)/n:.1%})   "
          f"below 0.5: {below(0.5)} ({below(0.5)/n:.1%})   "
          f"below 0.7: {below(0.7)} ({below(0.7)/n:.1%})")


def main() -> None:
    print(f"E03 grounding sweep — frozen MAP θ={THETA}/reject={REJECT}, related15 (offline, no API)")
    ctx = _load_map_context("gemini", embed_cache_path=None)
    scorer = _build_scorer(_frozen_spec(), ctx.agreement_embed_fn)
    # SHIPPED config: 5-voter selection (drop_l2_2) over the 6-voter primer + the escalate gate.
    voter_cache = _filtered_voter_cache(ctx.voter_cache, BEST_VOTER_SUBSET)
    print(f"  voter subset = {BEST_VOTER_SUBSET}, single_voter_policy = {BEST_SINGLE_VOTER_POLICY}")

    # 1. Frozen-config cascade selection per case (cached voter outputs, no LLM).
    case_outputs, accept, _early, _npol, _invoked = _replay(
        voter_cache, scorer, THETA, REJECT,
        single_voter_policy=BEST_SINGLE_VOTER_POLICY,
        force_escalate_on_polarity_conflict=BEST_FORCE_ESCALATE_POLARITY,
    )
    case_outputs = [co for co in case_outputs if co.case_id in ctx.silver_by_case]

    # 1b. Reconcile with production: replace LLM-paraphrased verbatim_support with the
    #     real cited paragraph from the DB (mirrors runner.py _replace_verbatim_from_db),
    #     so grounding scores real source text rather than paraphrases.
    from database import get_db_connection
    from pipeline.stages.knowledge_extraction.persistence import replace_verbatim_from_db
    replace_verbatim_from_db(get_db_connection(), case_outputs)

    # 2. Ground every finding through the pipeline's own NLI path (local, free).
    pipe = GroundingFilter(threshold=0.0)._pipe
    all_findings = [f for co in case_outputs for f in co.findings]
    pairs = [(f.verbatim_support or "", f.claim or "") for f in all_findings]
    print(f"scoring {len(pairs)} (verbatim_support → claim) pairs with grounding NLI ...")
    scores = _score_pairs(pairs, pipe)
    for f, s in zip(all_findings, scores):
        f.grounding_score = s

    # Validation anchor: replay must reproduce the E07 finding count.
    n_total = len(all_findings)
    flag = "OK" if n_total == EXPECTED_N_PIPELINE else "⚠️ MISMATCH"
    print(f"\nvalidation: n_findings={n_total}  vs E07 n_pipeline={EXPECTED_N_PIPELINE}  [{flag}]")
    if n_total != EXPECTED_N_PIPELINE:
        print(f"  (n differs from the 6-voter anchor {EXPECTED_N_PIPELINE} — expected for the 5-voter config.)")

    print("\ngrounding-score distribution (frozen-config MAP findings):")
    _distribution(scores)

    # 3. Sweep thresholds, score survivors vs silver (same fns as the stock sweep).
    rows: list[dict] = []
    print(f"\n{'threshold':>10}{'n_kept':>8}{'retain':>8}{'strF1opt':>9}{'f1opt':>8}"
          f"{'Popt':>7}{'Ropt':>7}")
    for t in GRID:
        filtered = [_apply_grounding_threshold(co, t) for co in case_outputs]
        m = _evaluate_outputs(filtered, ctx.silver_by_case, ctx.embedder, ctx.embed_cache,
                              SIMILARITY_THRESHOLD)
        n_kept = sum(len(co.findings) for co in filtered)
        label = "None" if t is None else f"{t:.2f}"
        rows.append({"grounding_threshold": label, "n_kept": n_kept,
                     "retention": round(n_kept / n_total, 4) if n_total else 0.0, **m})
        print(f"{label:>10}{n_kept:>8}{n_kept/n_total if n_total else 0:>8.3f}"
              f"{m['strict_f1_optimal']:>9.4f}{m['f1_optimal']:>8.4f}"
              f"{m['precision_optimal']:>7.3f}{m['recall_optimal']:>7.3f}")

    best = max(rows, key=lambda r: r["strict_f1_optimal"])
    print(f"\nbest strict_f1_optimal: threshold={best['grounding_threshold']} "
          f"(strict_f1={best['strict_f1_optimal']:.4f}, retention={best['retention']:.3f})")
    pinned = next((r for r in rows if r["grounding_threshold"] == "0.50"), None)
    if pinned:
        print(f"pinned (run.yaml) threshold=0.50: strict_f1={pinned['strict_f1_optimal']:.4f}, "
              f"retention={pinned['retention']:.3f} — the groundedness cost vs best-F1.")

    out_dir = REPORTS_DIR / "E03_grounding"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"sweep_{ts}.csv"
    _write_csv(csv_path, rows, _CSV_FIELDS)
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
