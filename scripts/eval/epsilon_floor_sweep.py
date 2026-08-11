#!/usr/bin/env python3
"""ε-constraint floor sweep for the E09 cost–quality frontier.

Companion to `lambda_sensitivity.py`. That script asks "which cell does the *knee* pick, and for
which λ?"; this one asks the same of the ε-constraint points (`economy`, `balanced`):

    point(ε) = argmin cost  subject to  strict_f1_optimal >= ε

Selection changes exactly when ε crosses the strict-F1 of a Pareto-optimal cell, so the bands are
solved exactly rather than sampled — no floor value can fall between two grid points and be missed.

The headline the sweep is built to check: ε-constraint reaches **every** Pareto-optimal cell as the
floor moves, whereas weighted-sum scalarization (the `knee`) can only ever return a vertex of the
upper convex hull. Both counts are printed.

Pure re-analysis of a frozen CSV — no network, no API calls, no cost.

    python3 scripts/eval/epsilon_floor_sweep.py
    python3 scripts/eval/epsilon_floor_sweep.py --csv <path>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The re-run that carries the `balanced` point; falls back to the frozen 3-point CSV.
FRONTIER_CANDIDATES = [
    _REPO_ROOT / "eval/reports/E09_cost_quality/frontier_20260809T133949.csv",
    _REPO_ROOT / "eval/reports/E09_cost_quality/frontier_20260622T165135.csv",
]

SHIPPED_FLOORS = {"economy": 0.50, "balanced": 0.60}   # MIN_ECONOMY_F1 / MIN_BALANCED_F1


def load(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(path.open()):
        rows.append({
            "label": f"θ{r['theta']}/r{r['reject_theta']}",
            "f1": float(r["strict_f1_optimal"]),
            "cpc": float(r["cost_per_chunk"]),
            "esc": float(r["escalate_rate"]),
            "tag": r.get("operating_point", ""),
        })
    return rows


def pareto(rows: list[dict]) -> list[dict]:
    """Cells not dominated on (cost ↓, strict_f1 ↑), cheapest first."""
    best: dict = {}
    for r in rows:                       # collapse exact cost ties to the best F1
        k = round(r["cpc"], 4)
        if k not in best or r["f1"] > best[k]["f1"]:
            best[k] = r
    out, top = [], float("-inf")
    for r in sorted(best.values(), key=lambda r: r["cpc"]):
        if r["f1"] > top:
            out.append(r)
            top = r["f1"]
    return out


def hull(par: list[dict]) -> list[dict]:
    """Upper convex hull of the Pareto set — the only cells a weighted sum can return."""
    h: list[dict] = []
    for r in par:
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (b["f1"] - a["f1"]) * (r["cpc"] - a["cpc"]) <= \
               (r["f1"] - a["f1"]) * (b["cpc"] - a["cpc"]):
                h.pop()
            else:
                break
        h.append(r)
    return h


def bands(par: list[dict]) -> list[tuple]:
    """Exact ε intervals over which the ε-constraint pick is constant.

    Ordered cheapest-first, the pick for ε in (f1[i-1], f1[i]] is cell i: any floor above the
    previous cell's F1 disqualifies it, and cell i is the next-cheapest that still clears.
    """
    out = []
    lo = 0.0
    for r in par:
        out.append((lo, r["f1"], r))
        lo = r["f1"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None, help="Write the ε bands to this CSV.")
    args = ap.parse_args()

    path = next((p for p in FRONTIER_CANDIDATES if p.exists()), None)
    if path is None:
        print("error: no E09 frontier CSV found", file=sys.stderr)
        return 1
    rows = load(path)
    par = pareto(rows)
    hl = hull(par)
    hset = {id(x) for x in hl}
    quality = max(par, key=lambda r: r["f1"])

    print(f"{path.name} — {len(rows)} cells · {len(par)} Pareto-optimal · "
          f"{len(hl)} on the upper convex hull")
    print(f"  ε-constraint can reach all {len(par)}; a weighted sum can reach only the "
          f"{len(hl)} hull vertices.\n")

    print("ε floor → selected cell (cheapest clearing the floor)")
    print(f"  {'ε floor range':>22}  {'cell':<12}{'strict_f1':>10}{'$/chunk':>9}{'esc':>7}"
          f"   {'vs quality':>22}  λ-reachable")
    out_rows = []
    for lo, hi, r in bands(par):
        save = 100 * (1 - r["cpc"] / quality["cpc"])
        drop = quality["f1"] - r["f1"]
        vs = "— (is quality)" if r is quality else f"{save:5.1f}% cheaper, −{drop:.4f} F1"
        hits = [n for n, f in SHIPPED_FLOORS.items() if lo < f <= hi]
        mark = ("   ← " + ", ".join(hits)) if hits else ""
        reach = "yes" if id(r) in hset else "NO"
        print(f"  ({lo:>7.4f}, {hi:>7.4f}]  {r['label']:<12}{r['f1']:>10.4f}{r['cpc']:>9.2f}"
              f"{r['esc']:>7.3f}   {vs:>22}  {reach:>11}{mark}")
        out_rows.append({
            "eps_lo_exclusive": round(lo, 6), "eps_hi_inclusive": round(hi, 6),
            "cell": r["label"], "strict_f1_optimal": r["f1"], "cost_per_chunk": r["cpc"],
            "escalate_rate": r["esc"],
            "pct_cheaper_than_quality": round(save, 2), "f1_below_quality": round(drop, 4),
            "reachable_by_weighted_sum": id(r) in hset,
            "shipped_floor": ",".join(hits),
        })

    print(f"\nAny floor above {quality['f1']:.4f} is infeasible (no cell clears it).")
    print("Shipped floors:")
    for name, f in sorted(SHIPPED_FLOORS.items(), key=lambda kv: kv[1]):
        pick = min((r for r in par if r["f1"] >= f), key=lambda r: r["cpc"], default=None)
        if pick is None:
            print(f"  {name:<9} ε={f}: infeasible")
            continue
        band = next(b for b in bands(par) if b[2] is pick)
        print(f"  {name:<9} ε={f} → {pick['label']} ({pick['f1']:.4f} @ {pick['cpc']:.2f});"
              f" unchanged for ε ∈ ({band[0]:.4f}, {band[1]:.4f}]"
              f"  headroom −{f - band[0]:.4f} / +{band[1] - f:.4f}")

    if args.csv and out_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.csv} ({len(out_rows)} bands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
