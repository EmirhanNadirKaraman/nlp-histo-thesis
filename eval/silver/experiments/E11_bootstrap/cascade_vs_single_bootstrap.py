#!/usr/bin/env python3
"""E11 — cascade vs single-model paired bootstrap (RQ3), offline.

The significance partner to E10. E10 gives the point estimate ("does single-Sonnet
match the cascade?"); E11 gives a confidence interval + a bootstrap p-value on the
strict-F1 gap, so the cascade's value claim is defensible, not eyeballed.

PAPER-LEVEL (clustered) bootstrap: papers are the independent sampling unit; cases
within a paper are correlated, so resampling cases would be pseudo-replication and
give falsely-narrow CIs. We resample the 15 papers with replacement and recompute
the micro-averaged strict-F1 (so per-paper strict-TP / n_pipeline / n_silver are the
sufficient statistics). n=15 papers → CIs are honestly wide; that is the point.

Offline: cascade re-replayed in-run at the frozen config (θ0.9/r0.1), single models
read from the voter_cache. Reuses E10's _single_model_outputs / _frozen_spec.

Usage:
  python -m eval.silver.experiments.E11_bootstrap.cascade_vs_single_bootstrap
  python -m eval.silver.experiments.E11_bootstrap.cascade_vs_single_bootstrap --boot 20000 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import datetime, timezone

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.experiments.E10_baselines.single_model_baselines import (
    _frozen_spec,
    _single_model_outputs,
)
from eval.silver.analysis.map_context import _load_map_context
from eval.silver.analysis.map_theta_sweep import (
    REPORTS_DIR,
    _build_scorer,
    _make_voters,
    _replay,
)
from eval.silver.matching.matcher import SIMILARITY_THRESHOLD
from eval.silver.analysis.pipeline_sweep import _evaluate_outputs
from eval.silver.analysis.run_new_summarization_sweeps import (
    BEST_REJECT_THETA, BEST_THETA, BEST_VOTER_SUBSET, _filtered_voter_cache,
)

THETA, REJECT = BEST_THETA, BEST_REJECT_THETA   # calibrated 5-voter operating point (E07: θ0.9 / reject0.2)

_OUT_FIELDS = [
    "baseline", "cascade_strict_f1", "baseline_strict_f1", "observed_delta",
    "ci_lo", "ci_hi", "p_cascade_better", "significant_95",
]


def _per_paper_stats(case_outputs, ctx) -> dict:
    """pmcid → (strict_tp, n_pipeline, n_silver), recovered from micro strict-F1."""
    by_paper = defaultdict(list)
    for co in case_outputs:
        if co.case_id in ctx.silver_by_case:
            by_paper[co.pmcid].append(co)
    out = {}
    for pmcid, cos in by_paper.items():
        m = _evaluate_outputs(cos, ctx.silver_by_case, ctx.embedder, ctx.embed_cache, SIMILARITY_THRESHOLD)
        nsil, npipe, sf1 = m["n_silver"], m["n_pipeline"], m["strict_f1_optimal"]
        # strict_f1 = 2*strict_tp/(n_pipeline+n_silver)  ⇒  strict_tp = sf1*(npipe+nsil)/2
        out[pmcid] = (sf1 * (npipe + nsil) / 2.0, float(npipe), float(nsil))
    return out


def _micro_strict_f1(stats: dict, sample: list[str]) -> float:
    stp = sum(stats[p][0] for p in sample if p in stats)
    npipe = sum(stats[p][1] for p in sample if p in stats)
    nsil = sum(stats[p][2] for p in sample if p in stats)
    sp = stp / npipe if npipe else 0.0
    sr = stp / nsil if nsil else 0.0
    return 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="E11 cascade-vs-single paired bootstrap (offline).")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"E11 paper-level paired bootstrap — B={args.boot}, seed={args.seed}, frozen cascade θ{THETA}/r{REJECT}")
    ctx = _load_map_context("gemini", embed_cache_path=None)

    # cascade (frozen, in-run) + every single model
    scorer = _build_scorer(_frozen_spec(), ctx.agreement_embed_fn)
    # CASCADE arm = the calibrated 5-voter selection (BEST_VOTER_SUBSET=drop_l2_2); same filter
    # as the sweep. The single-model baselines below stay on the FULL 6-voter cache — Claude-Haiku
    # is still a legitimate single-model baseline to compare the cascade against.
    cascade_cache = _filtered_voter_cache(ctx.voter_cache, BEST_VOTER_SUBSET)
    print(f"  cascade voter subset = {BEST_VOTER_SUBSET}")
    cascade_outputs = _replay(cascade_cache, scorer, THETA, REJECT,
                              single_voter_policy="escalate",  # E08: escalate best for 5-voter
                              force_escalate_on_polarity_conflict=True)[0]
    L1, L2, L3 = _make_voters()
    L3_list = L3 if isinstance(L3, (list, tuple)) else [L3]
    voters = [("l1", i, v) for i, v in enumerate(L1)] + [("l2", i, v) for i, v in enumerate(L2)] \
        + [("l3", i, v) for i, v in enumerate(L3_list)]

    systems = {"CASCADE": _per_paper_stats(cascade_outputs, ctx)}
    labels = {}
    for level, idx, v in voters:
        key = f"{level}{idx}"
        labels[key] = f"{v.provider}/{v.model}"
        systems[key] = _per_paper_stats(
            _single_model_outputs(ctx.voter_cache, ctx.silver_by_case, level, idx), ctx)

    papers = sorted(systems["CASCADE"].keys())
    n = len(papers)
    full = {sys: _micro_strict_f1(st, papers) for sys, st in systems.items()}
    casc_full = full["CASCADE"]

    rng = random.Random(args.seed)
    deltas = {k: [] for k in systems if k != "CASCADE"}
    for _ in range(args.boot):
        sample = [rng.choice(papers) for _ in range(n)]
        casc = _micro_strict_f1(systems["CASCADE"], sample)
        for k in deltas:
            deltas[k].append(casc - _micro_strict_f1(systems[k], sample))

    def pct(xs, q):
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * len(s)))]

    rows = []
    for k in deltas:
        ds = deltas[k]
        obs = casc_full - full[k]
        lo, hi = pct(ds, 0.025), pct(ds, 0.975)
        p_better = sum(1 for d in ds if d > 0) / len(ds)
        rows.append({"baseline": f"single {k} {labels[k]}", "cascade_strict_f1": round(casc_full, 4),
                     "baseline_strict_f1": round(full[k], 4), "observed_delta": round(obs, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "p_cascade_better": round(p_better, 4),
                     "significant_95": "yes" if (lo > 0 or hi < 0) else "no"})

    print(f"\ncascade strict_f1 = {casc_full:.4f}  ({n} papers, paper-level bootstrap)")
    print(f"{'baseline':38}{'base_f1':>8}{'Δ(casc−base)':>13}{'95% CI':>20}{'P(casc>base)':>13}{'sig':>5}")
    for r in sorted(rows, key=lambda r: -r["observed_delta"]):
        ci = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
        print(f"{r['baseline'][:37]:38}{r['baseline_strict_f1']:>8.4f}{r['observed_delta']:>+13.4f}"
              f"{ci:>20}{r['p_cascade_better']:>13.3f}{r['significant_95']:>5}")

    son = next((r for r in rows if "sonnet" in r["baseline"].lower()), None)
    if son:
        verdict = ("cascade SIGNIFICANTLY beats Sonnet-only" if son["ci_lo"] > 0
                   else "Sonnet-only SIGNIFICANTLY beats cascade" if son["ci_hi"] < 0
                   else "NO significant difference — cascade does not beat Sonnet-only")
        print(f"\nHEADLINE — cascade vs single-Sonnet: Δ={son['observed_delta']:+.4f} "
              f"95% CI [{son['ci_lo']:+.4f}, {son['ci_hi']:+.4f}], P(casc>sonnet)={son['p_cascade_better']:.3f}")
        print(f"  → {verdict}")

    out_dir = REPORTS_DIR / "E11_bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"bootstrap_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["observed_delta"]):
            w.writerow(r)
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
