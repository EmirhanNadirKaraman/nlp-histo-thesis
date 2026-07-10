#!/usr/bin/env python3
"""E14 — held-out generalization test (heldout15), RQ3 / Phase 4.

Replays the FROZEN cascade config (θ0.9 / reject0.1, hybrid + entity-heavy scorer —
the E06–E08 winner) on the **heldout15** primer and scores MAP findings vs heldout15
silver → ``strict_f1_optimal``. The headline generalization number: does the config
calibrated on related15 (strict-F1 **0.7135**) hold on a *disjoint* 15-paper test set?

ZERO voter API — replays the cached heldout15 primer (only cheap, one-time agreement
embeddings, served from cache after the first run). It needs the heldout15 primer to
exist first (the one billable step — the cascade gather):

  python -m eval.silver.map_theta_sweep prime \
      --source eval/data/source_cases_heldout15.jsonl \
      --primer-dir eval/data/map_primer_heldout15 --split all
  python -m eval.silver.map_theta_sweep collect \
      --source eval/data/source_cases_heldout15.jsonl \
      --primer-dir eval/data/map_primer_heldout15 --split all

then (free):

  python -m eval.silver.experiments.E14_heldout.heldout_eval
  python -m eval.silver.experiments.E14_heldout.heldout_eval --theta-frontier

``--theta-frontier`` also replays a θ grid as a **descriptive** cost-quality readout
(does the tradeoff generalize?) — NOT selection: heldout15 is the test set, the
operating point is chosen on related15 (E09). See `BUGS.md`/THESIS for the
select-on-related15 / report-on-heldout15 rule.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.map_context import _load_map_context  # noqa: E402
from eval.silver.map_theta_sweep import (  # noqa: E402
    REPORTS_DIR,
    ScorerSpec,
    _build_scorer,
    _replay,
)
from eval.silver.matcher import SIMILARITY_THRESHOLD  # noqa: E402
from eval.silver.pipeline_sweep import _evaluate_outputs  # noqa: E402
from eval.silver.run_new_summarization_sweeps import (  # noqa: E402
    BEST_REJECT_THETA, BEST_THETA, BEST_VOTER_SUBSET, _filtered_voter_cache,
)
from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig  # noqa: E402

THETA, REJECT = BEST_THETA, BEST_REJECT_THETA   # calibrated 5-voter operating point (E07: θ0.9 / reject0.2)
# 5-voter (drop_l2_2 = Claude-Haiku dropped), θ0.9/reject0.2, single_voter_policy=escalate
# related15 calibration reference — E08 (escalate is the gate argmax for the 5-voter config).
RELATED15_STRICT_F1 = 0.7160
THETA_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_HELDOUT_PRIMER = _REPO_ROOT / "eval" / "data" / "map_primer_heldout15" / "voter_cache.json"
_HELDOUT_SILVER = _REPO_ROOT / "eval" / "data" / "silver_findings_heldout15.jsonl"


def _frozen_spec() -> ScorerSpec:
    """The E05–E08 winner scorer (= configs/run.yaml / E03 `_frozen_spec`)."""
    cfg = AgreementConfig(
        scorer_kind="hybrid", alignment_strategy="greedy",
        tau=0.15, count_alpha=0.25, reuse_weight=0.15, contradiction_weight=0.20,
        hybrid=HybridConfig(w_category=0.15, w_embedding=0.30, w_entity=0.50, w_evidence=0.05),
    )
    return ScorerSpec("hybrid", "hybrid", cfg)


def _score(voter_cache, scorer, silver_by_case, embedder, embed_cache, sim_thr, theta, reject):
    case_outputs, accept, _early, _npol, invoked = _replay(
        voter_cache, scorer, theta, reject,
        single_voter_policy="escalate", force_escalate_on_polarity_conflict=True,  # E08: escalate best for 5-voter
    )
    m = _evaluate_outputs(case_outputs, silver_by_case, embedder, embed_cache, sim_thr)
    n_chunks = sum(accept.values())
    escalate = (n_chunks - accept.get("l1", 0)) / n_chunks if n_chunks else 0.0
    return m, accept, invoked, escalate


def main() -> int:
    ap = argparse.ArgumentParser(description="E14 — held-out generalization (heldout15), frozen config.")
    ap.add_argument("--primer", default=str(_HELDOUT_PRIMER), help="heldout15 voter_cache.json")
    ap.add_argument("--silver", default=str(_HELDOUT_SILVER), help="heldout15 silver findings jsonl")
    ap.add_argument("--sim-threshold", type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--theta-frontier", action="store_true",
                    help="also replay a θ grid (descriptive cost-quality; not selection)")
    args = ap.parse_args()

    print(f"E14 held-out generalization — frozen cascade (θ{THETA}/reject{REJECT}, hybrid) on heldout15")
    print(f"  primer={Path(args.primer).relative_to(_REPO_ROOT)}  silver={Path(args.silver).name}")
    ctx = _load_map_context(
        "gemini", embed_cache_path=None,
        cache_path=Path(args.primer), silver_path=Path(args.silver),
    )
    scorer = _build_scorer(_frozen_spec(), ctx.agreement_embed_fn)
    # SHIPPED config = the calibrated voter selection (BEST_VOTER_SUBSET; drop_l2_2 = 5-voter,
    # Claude-Haiku dropped). Filter the 6-voter primer to the shipped set via the SAME
    # _filtered_voter_cache the sweep (E07/E08) uses, so every stage drops identically.
    # "all" → no-op (full 6 voters).
    voter_cache = _filtered_voter_cache(ctx.voter_cache, BEST_VOTER_SUBSET)
    print(f"  voter subset = {BEST_VOTER_SUBSET}")

    # ── PRIMARY: frozen θ0.9/reject0.1 → strict-F1 vs silver ──
    m, accept, invoked, escalate = _score(
        voter_cache, scorer, ctx.silver_by_case, ctx.embedder, ctx.embed_cache,
        args.sim_threshold, THETA, REJECT,
    )
    sf1 = m["strict_f1_optimal"]
    print(f"\n[PRIMARY] heldout15 @ frozen θ{THETA}/reject{REJECT}:")
    print(f"  strict_f1_optimal = {sf1:.4f}   loose_f1_optimal = {m['f1_optimal']:.4f}  "
          f"(P {m['precision_optimal']:.3f} / R {m['recall_optimal']:.3f})")
    print(f"  n_silver={m['n_silver']}  n_pipeline={m['n_pipeline']}  escalate_rate={escalate:.3f}")
    print(f"\n  related15 (calibration) reference: strict_f1 {RELATED15_STRICT_F1:.4f}")
    print(f"  GENERALIZATION GAP (heldout − related15) = {sf1 - RELATED15_STRICT_F1:+.4f}")
    print("  → ≈0 means the frozen config generalizes; large negative ⇒ overfit to related15.")

    rows = [{
        "regime": "frozen", "theta": THETA, "reject_theta": REJECT,
        "strict_f1_optimal": round(sf1, 4), "f1_optimal": round(m["f1_optimal"], 4),
        "precision_optimal": round(m["precision_optimal"], 4),
        "recall_optimal": round(m["recall_optimal"], 4),
        "n_silver": m["n_silver"], "n_pipeline": m["n_pipeline"],
        "escalate_rate": round(escalate, 4),
        "l1_invoked": invoked.get("l1", 0), "l2_invoked": invoked.get("l2", 0),
        "l3_invoked": invoked.get("l3", 0),
    }]

    # ── OPTIONAL: θ frontier (descriptive cost-quality; NOT selection) ──
    if args.theta_frontier:
        print("\n── θ frontier on heldout15 (DESCRIPTIVE — does the cost-quality tradeoff generalize?) ──")
        print(f"  {'theta':>6} {'strict_f1':>10} {'escalate':>9} {'l3_calls':>9}")
        for theta in THETA_GRID:
            mm, acc, inv, esc = _score(
                voter_cache, scorer, ctx.silver_by_case, ctx.embedder, ctx.embed_cache,
                args.sim_threshold, theta, REJECT,
            )
            print(f"  {theta:6.2f} {mm['strict_f1_optimal']:10.4f} {esc:9.3f} {inv.get('l3', 0):9d}")
            rows.append({
                "regime": "theta_frontier", "theta": theta, "reject_theta": REJECT,
                "strict_f1_optimal": round(mm["strict_f1_optimal"], 4),
                "f1_optimal": round(mm["f1_optimal"], 4),
                "precision_optimal": round(mm["precision_optimal"], 4),
                "recall_optimal": round(mm["recall_optimal"], 4),
                "n_silver": mm["n_silver"], "n_pipeline": mm["n_pipeline"],
                "escalate_rate": round(esc, 4),
                "l1_invoked": inv.get("l1", 0), "l2_invoked": inv.get("l2", 0),
                "l3_invoked": inv.get("l3", 0),
            })

    out_dir = REPORTS_DIR / "E14_heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"heldout_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
