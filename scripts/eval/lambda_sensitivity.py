#!/usr/bin/env python3
"""λ sensitivity for the `knee` operating point — how stable is COST_LAMBDA = 0.20?

The knee is picked by weighted-sum scalarization (thesis §3.5.7):

    knee = argmax( strict_f1_optimal − λ · cost )

with λ pinned at 0.20 in two places — `eval/silver/analysis/run_new_summarization_sweeps.py`
(`COST_LAMBDA`, cost axis = `_cost_frac`) and
`eval/silver/experiments/E09_cost_quality/cost_quality_frontier.py` (cost axis = `cost_norm`).
That value was never swept. This script answers "for which λ does the chosen cell stay the
same?" without re-running anything.

The score is linear in λ for each cell, so the argmax over λ is the upper envelope of a line
arrangement: the breakpoints are computed exactly (as chord slopes between cells), not sampled
on a grid, so a band edge cannot be missed between grid points.

Pure re-analysis of frozen CSVs — no network, no API calls, no cost.

    python3 scripts/eval/lambda_sensitivity.py                    # print both analyses
    python3 scripts/eval/lambda_sensitivity.py --csv <path>       # write the bands as CSV
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SHIPPED_LAMBDA = 0.20

E09_CSV = _REPO_ROOT / "eval/reports/E09_cost_quality/frontier_20260622T165135.csv"
E06_CSV = _REPO_ROOT / "eval/reports/E06_family_refine/sweep_20260622T013444.csv"

# run_new_summarization_sweeps.TIER_COST_BLENDED — 6-voter prices, matching the
# sweep that E06's structure knee was selected from.
TIER_COST_BLENDED = {"l2": 9.60, "l3": 18.00}


def _cost_frac(row: dict) -> float:
    """Mirror of run_new_summarization_sweeps._cost_frac."""
    n = float(row.get("n_chunks") or 0.0)
    if n <= 0:
        return 0.0
    w2, w3 = TIER_COST_BLENDED["l2"], TIER_COST_BLENDED["l3"]
    l2 = float(row.get("n_l2_invoked") or 0.0)
    l3 = float(row.get("n_l3_invoked") or 0.0)
    return (w2 * l2 + w3 * l3) / ((w2 + w3) * n)


def load_e09(path: Path) -> list[dict]:
    """E09 frontier — cost axis is the precomputed `cost_norm` column."""
    return [
        {
            "f1": float(r["strict_f1_optimal"]),
            "cost": float(r["cost_norm"]),
            "label": f"θ{r['theta']}/r{r['reject_theta']}",
            "extra": f"${float(r['cost_per_chunk']):.2f}/chunk, esc {float(r['escalate_rate']):.3f}",
        }
        for r in csv.DictReader(path.open())
    ]


def load_e06(path: Path) -> list[dict]:
    """E06 family_refine — cost axis recomputed with `_cost_frac`."""
    rows = []
    for r in csv.DictReader(path.open()):
        try:
            f1 = float(r["strict_f1_optimal"])
        except (KeyError, TypeError, ValueError):
            continue          # unscored cell
        rows.append({
            "f1": f1,
            "cost": _cost_frac(r),
            "label": f"{r['variant'].split('__', 2)[-1]} θ{r['theta']}/r{r['reject_theta']}",
            "extra": f"{r['embedder']}/{r['scorer_kind']}/{r['alignment_strategy']}",
        })
    return rows


def envelope(rows: list[dict], lam_max: float = 5.0) -> list[tuple]:
    """Exact λ intervals over which the argmax cell is constant.

    Adjacent intervals whose winning cell has the same label AND the same
    (f1, cost) to 4 dp are merged — near-duplicate cells (e.g. reject_theta
    variants that resolve identically) otherwise split a band cosmetically.
    """
    bands: list[tuple] = []
    lam = 0.0
    while lam < lam_max:
        cur = max(rows, key=lambda r: r["f1"] - lam * r["cost"])
        nxt = None
        for r in rows:
            if r is cur or r["cost"] >= cur["cost"]:
                continue      # not cheaper → can never overtake as λ grows
            cross = (cur["f1"] - r["f1"]) / (cur["cost"] - r["cost"])
            if cross > lam + 1e-9 and (nxt is None or cross < nxt - 1e-9):
                nxt = cross
        hi = nxt if nxt is not None else float("inf")
        bands.append((lam, hi, cur))
        if nxt is None or nxt >= lam_max:
            break
        lam = nxt

    merged: list[tuple] = []
    for lo, hi, r in bands:
        key = (r["label"], round(r["f1"], 4), round(r["cost"], 4))
        if merged:
            p_lo, _, p_r = merged[-1]
            if (p_r["label"], round(p_r["f1"], 4), round(p_r["cost"], 4)) == key:
                merged[-1] = (p_lo, hi, p_r)
                continue
        merged.append((lo, hi, r))
    return merged


def report(name: str, rows: list[dict]) -> list[dict]:
    print(f"\n=== {name} — {len(rows)} scored cells ===")
    bands = envelope(rows)
    out = []
    for lo, hi, r in bands:
        hits = lo <= SHIPPED_LAMBDA < hi
        hi_s = "∞" if hi == float("inf") else f"{hi:.4f}"
        print(f"  λ ∈ [{lo:7.4f}, {hi_s:>7})  {r['label']:<46} "
              f"sf1 {r['f1']:.4f}  cost {r['cost']:.4f}  {r['extra']}"
              f"{'   ← λ=0.20' if hits else ''}")
        out.append({
            "sweep": name, "lambda_lo": round(lo, 6),
            "lambda_hi": "inf" if hi == float("inf") else round(hi, 6),
            "cell": r["label"], "strict_f1_optimal": r["f1"],
            "cost": round(r["cost"], 6), "detail": r["extra"],
            "contains_shipped_lambda": hits,
        })
    band = next((b for b in bands if b[0] <= SHIPPED_LAMBDA < b[1]), None)
    if band:
        lo, hi, r = band
        hi_s = "∞" if hi == float("inf") else f"{hi:.4f}"
        print(f"  → at λ={SHIPPED_LAMBDA}: {r['label']} (sf1 {r['f1']:.4f}); "
              f"stable for λ ∈ [{lo:.4f}, {hi_s})")
        print(f"    headroom below {SHIPPED_LAMBDA - lo:+.4f} · above "
              f"{'unbounded' if hi == float('inf') else f'{hi - SHIPPED_LAMBDA:+.4f}'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Write the λ bands to this CSV path.")
    args = ap.parse_args()

    rows_out = []
    for name, path, loader in (
        ("E09 cost-quality frontier (cost_norm)", E09_CSV, load_e09),
        ("E06 family_refine structures (cost_frac)", E06_CSV, load_e06),
    ):
        if not path.exists():
            print(f"skip {name}: {path} not found", file=sys.stderr)
            continue
        rows_out += report(name, loader(path))

    if args.csv and rows_out:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {args.csv} ({len(rows_out)} bands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
