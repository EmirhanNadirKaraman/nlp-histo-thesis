#!/usr/bin/env python3
"""
Fit a 1D deferral-score threshold for the MAP-stage router.

Usage
-----
python scripts/fit_routing_threshold.py \\
    --records out/routing_records.jsonl \\
    --out-dir out/threshold_fitting/ \\
    --max-false-accept 0.10 \\
    --min-recall 0.70 \\
    [--plot] \\
    [--boundary-lo 0.5] \\
    [--boundary-hi 0.85]

Input
-----
A JSONL file of RoutingRecord objects exported by MapStage with a RoutingDataset
collector.  Records must have keep_ok labels filled (True/False) to contribute to
precision/recall metrics.  Unlabeled records (keep_ok=None) contribute only to
dataset statistics and escalation_rate.

Output
------
out_dir/metrics.csv     — per-threshold metrics table
out_dir/summary.json    — recommended theta + dataset statistics
out_dir/curve.png       — optional curves (--plot)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Bootstrap the repo root onto sys.path so the repository-local imports below resolve
# when this script is run directly (`python scripts/…`) — B-095.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.knowledge_extraction.routing.routing_dataset import (
    RoutingDataset,
    RoutingRecord,
)

load_dotenv()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--records", required=True, help="Path to routing_records.jsonl")
    p.add_argument("--out-dir", default="out/threshold_fitting/", help="Output directory")
    p.add_argument(
        "--max-false-accept", type=float, default=0.10,
        help="Max allowed false-accept rate (default: 0.10)",
    )
    p.add_argument(
        "--min-recall", type=float, default=0.70,
        help="Min required recall (default: 0.70)",
    )
    p.add_argument("--plot", action="store_true", help="Save precision/recall/cost curves")
    p.add_argument("--boundary-lo", type=float, default=0.5,
                   help="Lower bound of calibration-critical region (default: 0.5)")
    p.add_argument("--boundary-hi", type=float, default=0.85,
                   help="Upper bound of calibration-critical region (default: 0.85)")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_fitting_data(
    records_path: str | Path,
) -> tuple[
    list[RoutingRecord],  # ag_records
    list[RoutingRecord],  # labeled
    list[RoutingRecord],  # acceptable  (keep_ok=True)
    list[RoutingRecord],  # unacceptable (keep_ok=False)
]:
    """
    Partition records into the four populations used for fitting.

    Only agreement-gate records with a non-null deferral_score are meaningful
    for threshold calibration.  Records from schema/grounding gates are always
    hard decisions that theta cannot influence — they are excluded here and
    counted separately in dataset_stats.
    """
    all_records = list(RoutingDataset.iter_records(records_path))
    ag_records = [
        r for r in all_records
        if r.gate_origin == "agreement_gate" and r.deferral_score is not None
    ]
    labeled = [r for r in ag_records if r.keep_ok is not None]
    acceptable = [r for r in labeled if r.keep_ok is True]
    unacceptable = [r for r in labeled if r.keep_ok is False]
    return all_records, ag_records, labeled, acceptable, unacceptable


# ── Metrics ────────────────────────────────────────────────────────────────────

def _fmt(v: float | None) -> str:
    """Format float for CSV; empty string for NaN/None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.6f}"


def compute_metrics(
    ag_records: list[RoutingRecord],
    labeled: list[RoutingRecord],
    acceptable: list[RoutingRecord],
    thresholds: list[float],
) -> list[dict]:
    """
    Compute per-threshold metrics.

    Precision/recall/FAR are computed over *labeled* records only (keep_ok filled).
    escalation_rate is computed over *all* ag_records (realistic production cost proxy).
    keep_rate is n_kept / n_labeled (coverage of the cheap path within labeled set).
    """
    n_labeled = len(labeled)
    n_acceptable = len(acceptable)
    n_ag = len(ag_records)

    rows = []
    for theta in thresholds:
        kept_l = [r for r in labeled if (r.deferral_score or 0.0) >= theta]
        tp = sum(1 for r in kept_l if r.keep_ok is True)
        fp = sum(1 for r in kept_l if r.keep_ok is False)
        n_kept = len(kept_l)
        n_escalated = n_labeled - n_kept

        keep_precision = tp / n_kept if n_kept > 0 else float("nan")
        false_accept_rate = fp / n_kept if n_kept > 0 else float("nan")
        recall = tp / n_acceptable if n_acceptable > 0 else float("nan")
        keep_rate = n_kept / n_labeled if n_labeled > 0 else float("nan")

        n_escalated_ag = sum(
            1 for r in ag_records if (r.deferral_score or 0.0) < theta
        )
        escalation_rate = n_escalated_ag / n_ag if n_ag > 0 else float("nan")

        rows.append({
            "theta": theta,
            "n_kept": n_kept,
            "n_escalated": n_escalated,
            "keep_rate": keep_rate,
            "keep_precision": keep_precision,
            "false_accept_rate": false_accept_rate,
            "recall": recall,
            "escalation_rate": escalation_rate,
        })
    return rows


# ── Threshold selection ────────────────────────────────────────────────────────

def select_theta(
    metrics: list[dict],
    max_false_accept: float,
    min_recall: float,
) -> tuple[float, dict, bool, list[str]]:
    """
    Select theta* that maximises coverage (highest theta) subject to:
      1. false_accept_rate <= max_false_accept  (primary — safety)
      2. recall >= min_recall                   (secondary — cost)

    When infeasible, relax (2) first.  Returns (theta*, row, feasible, warnings).
    """
    warnings: list[str] = []

    candidates = [
        r for r in metrics
        if not math.isnan(r["false_accept_rate"])
        and r["false_accept_rate"] <= max_false_accept
        and not math.isnan(r["recall"])
        and r["recall"] >= min_recall
    ]

    if candidates:
        best = max(candidates, key=lambda r: r["theta"])
        return best["theta"], best, True, warnings

    # Infeasible — relax min_recall
    warnings.append(
        f"Constraints infeasible: no theta satisfies both "
        f"false_accept_rate ≤ {max_false_accept} and recall ≥ {min_recall}."
    )
    far_only = [
        r for r in metrics
        if not math.isnan(r["false_accept_rate"])
        and r["false_accept_rate"] <= max_false_accept
    ]
    if far_only:
        best = max(far_only, key=lambda r: r["theta"])
        warnings.append(
            f"Relaxed min_recall. Best theta under false-accept constraint: "
            f"{best['theta']:.2f} (recall={_fmt(best['recall'])})"
        )
        return best["theta"], best, False, warnings

    # Cannot satisfy FAR alone — return minimum-FAR theta
    warnings.append("Cannot satisfy false_accept_rate constraint at any theta.")
    valid = [r for r in metrics if not math.isnan(r["false_accept_rate"])]
    best = min(valid, key=lambda r: r["false_accept_rate"]) if valid else metrics[-1]
    return best["theta"], best, False, warnings


# ── I/O ────────────────────────────────────────────────────────────────────────

def write_csv(metrics: list[dict], out_dir: Path) -> None:
    path = out_dir / "metrics.csv"
    fieldnames = [
        "theta", "n_kept", "n_escalated", "keep_rate",
        "keep_precision", "false_accept_rate", "recall", "escalation_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({
                "theta": f"{row['theta']:.4f}",
                "n_kept": row["n_kept"],
                "n_escalated": row["n_escalated"],
                "keep_rate": _fmt(row["keep_rate"]),
                "keep_precision": _fmt(row["keep_precision"]),
                "false_accept_rate": _fmt(row["false_accept_rate"]),
                "recall": _fmt(row["recall"]),
                "escalation_rate": _fmt(row["escalation_rate"]),
            })
    print(f"[metrics]  {path}")


def write_summary(
    theta_star: float,
    row: dict,
    feasible: bool,
    dataset_stats: dict,
    warnings: list[str],
    constraints: dict,
    out_dir: Path,
) -> None:
    def _null(v: float) -> float | None:
        return None if (isinstance(v, float) and math.isnan(v)) else round(v, 6)

    summary = {
        "recommended_theta": round(theta_star, 4) if not math.isnan(theta_star) else None,
        "false_accept_rate_at_theta": _null(row["false_accept_rate"]),
        "recall_at_theta": _null(row["recall"]),
        "keep_rate_at_theta": _null(row["keep_rate"]),
        "escalation_rate_at_theta": _null(row["escalation_rate"]),
        "n_kept_at_theta": row["n_kept"],
        "constraints": constraints,
        "feasible": feasible,
        "dataset_stats": dataset_stats,
        "warnings": warnings,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"[summary]  {path}")


def plot_curves(
    metrics: list[dict],
    theta_star: float,
    max_false_accept: float,
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping plot.")
        return

    thetas = [r["theta"] for r in metrics]

    def _series(key: str) -> list[float | None]:
        return [None if math.isnan(r[key]) else r[key] for r in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thetas, _series("false_accept_rate"), color="crimson",
            label="false_accept_rate", linewidth=2)
    ax.plot(thetas, _series("recall"), color="steelblue",
            label="recall", linewidth=2)
    ax.plot(thetas, _series("escalation_rate"), color="darkorange",
            label="escalation_rate (all AG records)", linewidth=1.5, linestyle="--")
    ax.axvline(theta_star, color="black", linestyle=":", linewidth=1.5,
               label=f"θ* = {theta_star:.2f}")
    ax.axhline(max_false_accept, color="crimson", linestyle=":", linewidth=1, alpha=0.5,
               label=f"max_false_accept = {max_false_accept}")
    ax.set_xlabel("theta")
    ax.set_ylabel("rate")
    ax.set_title("MAP-stage router: deferral-score threshold calibration")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot]     {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records, ag_records, labeled, acceptable, unacceptable = load_fitting_data(
        args.records
    )

    gate_counts = Counter(r.gate_origin for r in all_records)
    n_boundary = sum(
        1 for r in labeled
        if args.boundary_lo <= (r.deferral_score or 0.0) <= args.boundary_hi
    )
    dataset_stats = {
        "n_total_records": len(all_records),
        "n_agreement_gate": len(ag_records),
        "n_labeled": len(labeled),
        "n_acceptable": len(acceptable),
        "n_unacceptable": len(unacceptable),
        "n_boundary_labeled": n_boundary,
        "boundary_region": [args.boundary_lo, args.boundary_hi],
        "gate_breakdown": dict(gate_counts),
    }

    print("\n=== Dataset ===")
    print(f"  total records   : {dataset_stats['n_total_records']}")
    print(f"  agreement_gate  : {dataset_stats['n_agreement_gate']}")
    print(f"  labeled         : {dataset_stats['n_labeled']}")
    print(f"  acceptable      : {dataset_stats['n_acceptable']}")
    print(f"  unacceptable    : {dataset_stats['n_unacceptable']}")
    print(f"  boundary region : {n_boundary} labeled in "
          f"[{args.boundary_lo}, {args.boundary_hi}]")

    if not labeled:
        print("\n[WARNING] No labeled agreement-gate records found. "
              "Fill keep_ok labels before fitting.\n")
        write_csv([], out_dir)
        write_summary(
            float("nan"),
            {"false_accept_rate": float("nan"), "recall": float("nan"),
             "keep_rate": float("nan"), "escalation_rate": float("nan"), "n_kept": 0},
            feasible=False,
            dataset_stats=dataset_stats,
            warnings=["No labeled records found."],
            constraints={
                "max_false_accept": args.max_false_accept,
                "min_recall": args.min_recall,
            },
            out_dir=out_dir,
        )
        return

    if n_boundary < 20:
        print(f"\n[WARNING] Only {n_boundary} labeled records in boundary region "
              f"[{args.boundary_lo}, {args.boundary_hi}] — theta fit may be unreliable.")

    # Build threshold grid: observed breakpoints + dense 0.01 grid
    observed = sorted({r.deferral_score for r in ag_records if r.deferral_score is not None})
    grid = [round(i / 100, 2) for i in range(101)]
    thresholds = sorted(set(observed + grid))

    metrics = compute_metrics(ag_records, labeled, acceptable, thresholds)
    theta_star, star_row, feasible, warnings = select_theta(
        metrics, args.max_false_accept, args.min_recall
    )

    for w in warnings:
        print(f"\n[WARNING] {w}")

    print("\n=== Result ===")
    print(f"  theta*            : {theta_star:.4f}")
    print(f"  false_accept_rate : {_fmt(star_row['false_accept_rate'])}")
    print(f"  recall            : {_fmt(star_row['recall'])}")
    print(f"  keep_rate         : {_fmt(star_row['keep_rate'])}")
    print(f"  escalation_rate   : {_fmt(star_row['escalation_rate'])}")
    print(f"  feasible          : {feasible}\n")

    write_csv(metrics, out_dir)
    write_summary(
        theta_star, star_row, feasible, dataset_stats, warnings,
        constraints={
            "max_false_accept": args.max_false_accept,
            "min_recall": args.min_recall,
        },
        out_dir=out_dir,
    )
    if args.plot:
        plot_curves(metrics, theta_star, args.max_false_accept, out_dir)


if __name__ == "__main__":
    main()
