#!/usr/bin/env python3
"""E06b/E06c — voter-subset-aware per-chunk cost + domination analysis (RQ3), offline.

Augments a voter-subset sweep CSV (the E06b ``voter_subset_screen`` or E06c
``voter_subset_refine`` output of ``run_new_summarization_sweeps.py``) with the
CORRECT price-weighted per-chunk cost, writes the augmented CSV as a sidecar
NEXT TO the input (``<stem>_cost.csv``), and prints a per-config domination
table (each voter drop vs its ``voter_subset="all"`` baseline).

WHY A DEDICATED COSTER (not ``eval.silver.analysis.escalation_breakdown``)
-----------------------------------------------------------------
``escalation_breakdown.py``'s ``cost_frac`` drops the L1 term as "constant" and
charges the FULL-ensemble per-tier prices (L2=9.6, L3=18). That is valid ONLY
for ``voter_subset="all"``. Dropping a voter changes the tier's voter set, so:

  * dropping an **L1** voter LOWERS the L1 price paid on *every* chunk — a real
    saving ``escalation_breakdown`` ignores entirely (it omits the L1 term), and
  * dropping an **L2** voter lowers the L2 price on the escalated chunks.

So for the rows that actually matter (the drops), ``cost_frac`` mis-ranks them.
This module charges the **reduced** tier price for each dropped-voter row, so a
drop's cost is comparable to ``all`` on the same axis.

COST MODEL (per row)
--------------------
    active L1 price = Σ_{v∈L1} price(v) − price(dropped L1 voter, if drop_l1_i)
    active L2 price = Σ_{v∈L2} price(v) − price(dropped L2 voter, if drop_l2_i)
    cost_per_chunk  = (n_l1_invoked·L1p + n_l2_invoked·L2p + n_l3_invoked·L3p) / n_chunks

``price(v) = input_per_1m + output_per_1m`` from ``configs/model_prices.json``
(USD/1M tok). Same justification as E09 cost-quality: every tier summarises the
SAME chunk into the SAME AuditableSummary schema, so input tokens are ≈ equal
and cost ≈ price × call-count = the invocation index (and these are offline
replays, so per-call token logs don't exist). L3 (single Sonnet voter) is never
dropped by the grid, so its price is always full.

DERIVED COLUMNS appended to the sidecar (idempotent — re-running overwrites them):
    l1_escalate_rate      = n_l2_invoked / n_l1_invoked     (L1 disagreed → L2)
    l2_escalate_rate_cond = n_l3_invoked / n_l2_invoked     (of L2-reached, → L3)
    l1_active_price, l2_active_price                        (reduced tier prices)
    cost_per_chunk                                          (the voter-aware cost)
    sf1_delta_vs_all      strict_f1_optimal − the (variant,θ,reject) 'all' baseline
    cost_pct_vs_all       (cost − all.cost) / all.cost × 100
    dominates_all         "yes" iff sf1_delta>0 AND cost<all.cost (better+cheaper)

Domination is judged WITHIN a (variant, theta, reject_theta) group so the drop
and its 'all' baseline are at the same operating point. NB: on the fixed-θ E06b
screen this is single-θ — a drop that dominates here need NOT dominate the full
θ frontier (that is what E06c voter_subset_refine tests).

Usage:
  python -m eval.silver.experiments.E06b_voter_subset_cost.voter_subset_cost           # latest E06b CSV
  python -m eval.silver.experiments.E06b_voter_subset_cost.voter_subset_cost <csv>     # explicit CSV
  python -m eval.silver.experiments.E06b_voter_subset_cost.voter_subset_cost --print   # table only, no sidecar
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.analysis.map_theta_sweep import REPORTS_DIR, _make_voters

_DERIVED = (
    "l1_escalate_rate", "l2_escalate_rate_cond", "l1_active_price", "l2_active_price",
    "cost_per_chunk", "sf1_delta_vs_all", "cost_pct_vs_all", "dominates_all",
)


def _f(r: dict, k: str) -> float:
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return 0.0


def _price_lookup():
    """model id → blended (input+output) USD/1M tok, from configs/model_prices.json.

    Mirrors E09 cost_quality_frontier: exact key, else longest prefix match
    (handles date-suffixed ids like claude-sonnet-4-6-20251001)."""
    from nlp_histo.pipeline.stages.knowledge_extraction.costing.pricing import (
        default_price_path,
    )

    raw = json.loads(default_price_path().read_text(encoding="utf-8"))["models"]

    def price(model: str) -> float:
        if model in raw:
            m = raw[model]
        else:
            cand = sorted((k for k in raw if model.startswith(k) or k.startswith(model)),
                          key=len, reverse=True)
            if not cand:
                raise SystemExit(f"no price for voter model {model!r} in model_prices.json")
            m = raw[cand[0]]
        return float(m["input_per_1m"]) + float(m["output_per_1m"])

    return price


def _tier_prices() -> tuple[list[float], list[float], float]:
    """Per-voter prices: (L1 list, L2 list, L3 scalar) from the `real` cascade."""
    price = _price_lookup()
    l1, l2, l3 = _make_voters()
    l3v = (l3[0] if isinstance(l3, (list, tuple)) else l3)
    return ([price(v.model) for v in l1], [price(v.model) for v in l2], price(l3v.model))


def _active_tier_prices(subset: str, L1: list[float], L2: list[float]) -> tuple[float, float]:
    """(active L1 price, active L2 price) for a voter_subset, dropped voter removed."""
    l1p, l2p = sum(L1), sum(L2)
    if subset.startswith("drop_l1_"):
        l1p -= L1[int(subset.rsplit("_", 1)[1])]
    elif subset.startswith("drop_l2_"):
        l2p -= L2[int(subset.rsplit("_", 1)[1])]
    return l1p, l2p


def _cost_per_chunk(r: dict, L1: list[float], L2: list[float], L3: float) -> tuple[float, float, float]:
    n = _f(r, "n_chunks") or 1.0
    l1p, l2p = _active_tier_prices(r.get("voter_subset", "all"), L1, L2)
    cost = (_f(r, "n_l1_invoked") * l1p + _f(r, "n_l2_invoked") * l2p
            + _f(r, "n_l3_invoked") * L3) / n
    return cost, l1p, l2p


def _latest_e06b() -> Path:
    hits = [p for p in glob.glob(str(REPORTS_DIR / "E06b_voter_subset_screen" / "sweep_*.csv"))
            if "checkpoint" not in p]
    if not hits:
        raise SystemExit("no E06b sweep CSV found — run --stage voter_subset_screen first, "
                         "or pass an explicit CSV path.")
    return Path(sorted(hits)[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=None,
                    help="voter-subset sweep CSV (default: latest E06b_voter_subset_screen/sweep_*.csv).")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the domination table only; do not write the <stem>_cost.csv sidecar.")
    ap.add_argument("--cost-lambda", type=float, default=0.20,
                    help="knee weight for the reported quality−λ·cost note (display only).")
    args = ap.parse_args()

    src = Path(args.csv) if args.csv else _latest_e06b()
    if not src.exists():
        raise SystemExit(f"CSV not found: {src}")
    rows = list(csv.DictReader(open(src)))
    if not rows:
        raise SystemExit(f"{src} is empty.")
    if not rows[0].get("n_l3_invoked"):
        raise SystemExit(f"{src} has no n_l*_invoked columns — needs a per-tier sweep CSV.")

    L1, L2, L3 = _tier_prices()
    print(f"input: {src}")
    print(f"per-voter blended $/1M tok — L1={[round(x,2) for x in L1]} (Σ {sum(L1):.2f})  "
          f"L2={[round(x,2) for x in L2]} (Σ {sum(L2):.2f})  L3={L3:.2f}")

    # ── derive cost + escalation columns per row ──
    for r in rows:
        cost, l1p, l2p = _cost_per_chunk(r, L1, L2, L3)
        nl1, nl2 = _f(r, "n_l1_invoked"), _f(r, "n_l2_invoked")
        r["l1_escalate_rate"] = round(nl2 / nl1, 4) if nl1 else ""
        r["l2_escalate_rate_cond"] = round(_f(r, "n_l3_invoked") / nl2, 4) if nl2 else ""
        r["l1_active_price"] = round(l1p, 4)
        r["l2_active_price"] = round(l2p, 4)
        r["cost_per_chunk"] = round(cost, 4)

    # ── domination vs 'all' within each (variant, theta, reject_theta) group ──
    def gkey(r):
        return (r.get("variant", ""), r.get("theta", ""), r.get("reject_theta", ""))

    groups: dict = {}
    for r in rows:
        groups.setdefault(gkey(r), []).append(r)

    for grp in groups.values():
        base = next((x for x in grp if x.get("voter_subset") == "all"), None)
        bf = _f(base, "strict_f1_optimal") if base else 0.0
        bc = base["cost_per_chunk"] if base else 0.0
        for r in grp:
            df = _f(r, "strict_f1_optimal") - bf
            r["sf1_delta_vs_all"] = round(df, 4)
            r["cost_pct_vs_all"] = round((r["cost_per_chunk"] - bc) / bc * 100, 1) if bc else ""
            r["dominates_all"] = ("yes" if (df > 0 and r["cost_per_chunk"] < bc
                                            and r.get("voter_subset") != "all") else "")

    # ── print per-config table ──
    for (variant, theta, reject), grp in groups.items():
        base = next((x for x in grp if x.get("voter_subset") == "all"), None)
        bf = _f(base, "strict_f1_optimal") if base else 0.0
        bc = base["cost_per_chunk"] if base else 0.0
        print(f"\n=== {variant or '(no variant)'}  θ{theta}/r{reject}  "
              f"(all: sf1={bf:.4f}, cost/chunk={bc:.2f}) ===")
        print(f"  {'voter_subset':11s} {'model':26s} {'sf1':>7s} {'Δsf1':>8s} "
              f"{'cost':>7s} {'Δcost%':>7s}  verdict")
        for r in sorted(grp, key=lambda r: -_f(r, "strict_f1_optimal")):
            model = (r.get("dropped_voter_model") or "—").split("/")[-1]
            df, c = _f(r, "sf1_delta_vs_all"), r["cost_per_chunk"]
            if r.get("voter_subset") == "all":
                verdict = "— baseline"
            elif r["dominates_all"] == "yes":
                verdict = "DOMINATES all"
            elif df > 0:
                verdict = "better but costs more"
            elif c < bc:
                verdict = "cheaper but worse"
            else:
                verdict = "dominated"
            pct = r["cost_pct_vs_all"]
            print(f"  {str(r.get('voter_subset')):11s} {model:26s} {_f(r,'strict_f1_optimal'):7.4f} "
                  f"{df:+8.4f} {c:7.2f} {('' if pct=='' else f'{pct:+.1f}'):>7s}  {verdict}")

    dominators = [r for r in rows if r.get("dominates_all") == "yes"]
    print(f"\n{len(dominators)} (config, drop) pair(s) dominate 'all' at fixed θ"
          + (": " + ", ".join(f"{r.get('variant')}/{r.get('voter_subset')}" for r in dominators) if dominators else ""))
    if dominators:
        print("  → fixed-θ domination only; confirm on the full θ grid with --stage voter_subset_refine (E06c).")

    if args.print_only:
        print("\n(--print: no sidecar written)")
        return

    # ── write sidecar NEXT TO the input CSV ──
    out = src.with_name(f"{src.stem}_cost.csv")
    fields = list(rows[0].keys())
    for c in _DERIVED:
        if c not in fields:
            fields.append(c)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nsidecar → {out}")


if __name__ == "__main__":
    main()
