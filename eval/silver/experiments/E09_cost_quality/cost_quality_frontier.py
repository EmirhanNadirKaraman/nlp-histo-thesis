#!/usr/bin/env python3
"""E09 — cost–quality frontier for the MAP cascade (RQ3), offline.

Synthesises the calibration: where do the QUALITY config (gemini/hybrid/greedy,
the E06/E07 winner) and the ECONOMY config (openai/hybrid/soft_max, the E06c
Pareto economy point) sit on the strict-F1-vs-cost frontier — using the per-tier
INVOCATION columns (n_l{1,2,3}_invoked) added this session, not just escalate_rate.

Cost model (PRIMARY axis): price-weighted per-tier cost per chunk
    cost ∝ Σ_tier  n_tier_invoked × Σ_{voter∈tier} price(voter)
priced from configs/model_prices.json (USD/1M tokens, input+output). Justification
for not needing per-call token counts: every tier summarises the SAME chunk into
the SAME AuditableSummary schema, so input tokens are identical across tiers and
outputs are comparable — the cost differential is dominated by price × call-count,
which is exactly the invocation index. (And these are offline replays over the
cached voter_cache, so per-config token logs don't exist anyway.) escalate_rate
(L3 Sonnet fraction) is reported as a secondary axis + a robustness check.

Inputs (existing CSVs, no API):
  quality θ-curve  → eval/reports/E07_map_theta/sweep_*.csv
  economy θ-curve  → eval/reports/E06c_voter_subset_refine/sweep_*.csv (voter_subset=all)

Usage:
  python -m eval.silver.experiments.E09_cost_quality.cost_quality_frontier
"""
from __future__ import annotations

import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.analysis.map_theta_sweep import REPORTS_DIR, _make_voters
from eval.silver.analysis.run_new_summarization_sweeps import (
    BEST_REJECT_THETA, BEST_THETA, BEST_VOTER_SUBSET, _subset_meta,
)

COST_LAMBDA = 0.20      # knee = argmax(strict_f1 − λ·cost_norm); matches structure_screen
MIN_ECONOMY_F1 = 0.50   # economy = cheapest operating point on the curve clearing this strict-F1 floor
# Verification anchor: the SHIPPED operating point (BEST_THETA/BEST_REJECT_THETA, escalate gate) must
# reproduce the E08 5-voter strict-F1 (0.7160). 2026-06-22: this experiment builds the frontier from the
# SHIPPED config's OWN θ-curve — the E08b stage (map_theta_shipped: θ×reject re-swept at the chosen gate,
# voter_subset=BEST_VOTER_SUBSET) — not a quality+economy two-structure join (voter sets differ now and
# E06c has no drop_l2_2 economy). REQUIRES E08b to have run. (θ0.9 is gate-invariant; E08b characterizes
# the shipped config, it does not re-select θ.)
_ANCHOR_SF1 = 0.7160

_OUT_FIELDS = [
    "structure", "theta", "reject_theta", "strict_f1_optimal", "f1_optimal",
    "escalate_rate", "n_chunks", "n_l1_invoked", "n_l2_invoked", "n_l3_invoked",
    "cost_per_chunk", "cost_norm", "on_frontier_cost", "on_frontier_escalate",
    "operating_point",
]


def _latest(pattern: str) -> Path:
    hits = [p for p in glob.glob(pattern) if "checkpoint" not in p]
    if not hits:
        raise SystemExit(f"no CSV matched {pattern} — run that stage first.")
    return Path(sorted(hits)[-1])


def _price_lookup() -> dict:
    from nlp_histo.pipeline.stages.knowledge_extraction.costing.pricing import (
        default_price_path,
    )

    raw = json.loads(default_price_path().read_text(encoding="utf-8"))["models"]
    def price(model: str) -> float:
        if model in raw:
            m = raw[model]
        else:  # date-suffix / prefix fallback (e.g. claude-sonnet-4-6-20251001)
            cand = sorted((k for k in raw if model.startswith(k) or k.startswith(model)),
                          key=len, reverse=True)
            if not cand:
                raise SystemExit(f"no price for voter model {model!r} in model_prices.json")
            m = raw[cand[0]]
        return float(m["input_per_1m"]) + float(m["output_per_1m"])
    return price


def _tier_factors() -> tuple[float, float, float]:
    """Σ voter prices per tier — the per-INVOCATION cost weight (all tier voters fire when the
    tier is invoked). EXCLUDES the shipped dropped voter (BEST_VOTER_SUBSET) so the cost matches
    the 5-voter cascade — a dropped L2 voter lowers the L2 price paid on every escalated chunk."""
    price = _price_lookup()
    l1, l2, l3 = _make_voters()
    l3_list = l3 if isinstance(l3, (list, tuple)) else [l3]
    dl, di, _ = _subset_meta(BEST_VOTER_SUBSET)
    def tp(voters, lvl):
        return sum(price(v.model) for i, v in enumerate(voters) if not (lvl == dl and i == di))
    return tp(l1, "l1"), tp(l2, "l2"), tp(l3_list, "l3")


def _f(r: dict, k: str) -> float:
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return 0.0


def _load(path: Path, structure: str, voter_subset: str | None) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        if voter_subset is not None and r.get("voter_subset", "all") != voter_subset:
            continue
        if not r.get("n_l3_invoked"):  # per-tier cost columns must be populated
            raise SystemExit(f"{path} row missing n_l*_invoked — CSV predates the cost columns.")
        out.append({"structure": structure, **r})
    return out


def _pareto(rows: list[dict], cost_key: str) -> set:
    """Indices of rows NOT dominated on (cost↓, strict_f1↑)."""
    pts = [(i, _f(r, cost_key), _f(r, "strict_f1_optimal")) for i, r in enumerate(rows)]
    keep = set()
    for i, c, q in pts:
        if not any(cj <= c and qj >= q and j != i and (cj < c or qj > q) for j, cj, qj in pts):
            keep.add(i)
    return keep


def main() -> None:
    q_csv = _latest(str(REPORTS_DIR / "E08b_map_theta_shipped" / "sweep_*.csv"))
    print(f"5-voter cost-quality frontier — the SHIPPED config's own θ×reject curve (E08b): {q_csv}")
    # The shipped config's θ-curve IS the cost-quality frontier: θ0.3 (cheap) → θ0.9 (max-F1), one
    # voter-consistent config. quality/economy/knee are operating points ON this curve. (The old
    # two-structure quality+economy join mixed a separate 6-voter economy from E06c — incoherent for
    # the 5-voter config, and E06c has no drop_l2_2 economy curve to combine.)
    rows = _load(q_csv, "shipped", None)
    subsets = sorted({r.get("voter_subset", "all") for r in rows})
    print(f"  voter_subset in E07: {subsets}  (shipped = {BEST_VOTER_SUBSET})")

    L1f, L2f, L3f = _tier_factors()
    print(f"per-invocation cost weights (Σ voter $/1M tok, dropped voter excluded): "
          f"L1={L1f:.2f}  L2={L2f:.2f}  L3={L3f:.2f}")
    for r in rows:
        nch = _f(r, "n_chunks") or 1.0
        r["cost_per_chunk"] = round(
            (_f(r, "n_l1_invoked") * L1f + _f(r, "n_l2_invoked") * L2f
             + _f(r, "n_l3_invoked") * L3f) / nch, 4)
    cmax = max(r["cost_per_chunk"] for r in rows) or 1.0
    for r in rows:
        r["cost_norm"] = round(r["cost_per_chunk"] / cmax, 4)

    # verification anchor: the calibrated operating point reproduces the E07 5-voter strict-F1
    qa = next((r for r in rows
               if abs(_f(r, "theta") - BEST_THETA) < 1e-9
               and abs(_f(r, "reject_theta") - BEST_REJECT_THETA) < 1e-9), None)
    ok = qa is not None and abs(_f(qa, "strict_f1_optimal") - _ANCHOR_SF1) < 5e-4
    print(f"\nverification: θ{BEST_THETA}/r{BEST_REJECT_THETA} strict_f1="
          f"{_f(qa, 'strict_f1_optimal') if qa else None} vs anchor {_ANCHOR_SF1}  "
          f"[{'OK' if ok else '⚠️ MISMATCH — check the E07 CSV / pins'}]")

    # frontiers under both cost models (robustness check)
    fr_cost = _pareto(rows, "cost_per_chunk")
    fr_esc = _pareto(rows, "escalate_rate")
    for i, r in enumerate(rows):
        r["on_frontier_cost"] = "true" if i in fr_cost else "false"
        r["on_frontier_escalate"] = "true" if i in fr_esc else "false"

    # tagged operating points (all on the shipped config's θ-curve)
    quality_pt = max(rows, key=lambda r: _f(r, "strict_f1_optimal"))
    eligible = [r for r in rows if _f(r, "strict_f1_optimal") >= MIN_ECONOMY_F1]
    economy_pt = min(eligible or rows, key=lambda r: r["cost_per_chunk"])
    knee_pt = max(rows, key=lambda r: _f(r, "strict_f1_optimal") - COST_LAMBDA * r["cost_norm"])
    for r in rows:
        r["operating_point"] = ""
    quality_pt["operating_point"] = (quality_pt["operating_point"] + "|quality").strip("|")
    economy_pt["operating_point"] = (economy_pt["operating_point"] + "|economy").strip("|")
    knee_pt["operating_point"] = (knee_pt["operating_point"] + "|knee").strip("|")

    # print frontier (cost-sorted)
    print(f"\n{'struct':8}{'θ':>5}{'rej':>5}{'strF1':>8}{'cost/ch':>9}{'cnorm':>7}"
          f"{'esc':>6}{'L2inv':>6}{'L3inv':>6}  frontier(cost/esc)  pt")
    for r in sorted(rows, key=lambda r: r["cost_per_chunk"]):
        fr = ("C" if r["on_frontier_cost"] == "true" else "·") + ("E" if r["on_frontier_escalate"] == "true" else "·")
        print(f"{r['structure']:8}{_f(r,'theta'):>5.2f}{_f(r,'reject_theta'):>5.2f}"
              f"{_f(r,'strict_f1_optimal'):>8.4f}{r['cost_per_chunk']:>9.2f}{r['cost_norm']:>7.3f}"
              f"{_f(r,'escalate_rate'):>6.3f}{int(_f(r,'n_l2_invoked')):>6}{int(_f(r,'n_l3_invoked')):>6}"
              f"   {fr:>3}              {r['operating_point']}")

    # robustness: do the two cost models pick the same frontier set?
    same = fr_cost == fr_esc
    print(f"\nrobustness: cost-frontier vs escalate-frontier identical? {same}"
          + ("" if same else f"  (cost-only:{sorted(fr_cost-fr_esc)}  esc-only:{sorted(fr_esc-fr_cost)})"))
    print("operating points:")
    for tag, r in (("quality", quality_pt), ("economy", economy_pt), ("knee", knee_pt)):
        print(f"  {tag:8} {r['structure']}/θ{_f(r,'theta'):.2f}/r{_f(r,'reject_theta'):.2f}  "
              f"strict_f1={_f(r,'strict_f1_optimal'):.4f}  cost/chunk={r['cost_per_chunk']:.2f}  "
              f"esc={_f(r,'escalate_rate'):.3f}")

    out_dir = REPORTS_DIR / "E09_cost_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"frontier_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["cost_per_chunk"]):
            w.writerow(r)
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
