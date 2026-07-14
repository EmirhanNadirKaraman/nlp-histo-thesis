#!/usr/bin/env python3
"""E14 — held-out theta-sweep plot, with the related15 calibration curve overlaid.

Plots strict-F1 vs the agreement threshold theta on heldout15 (the descriptive
theta-frontier from heldout_eval.py --theta-frontier), and overlays the related15
calibration curve (E08b, escalate gate) on the same axes. The two curves share
the monotone, diminishing-returns shape -- i.e. the cost-quality trade-off
generalizes to the held-out set.

Reads:
  eval/reports/E14_heldout/heldout_*.csv            (regime=theta_frontier)
  eval/reports/E08b_map_theta_shipped/sweep_*.csv   (related15, escalate gate)
Writes:
  eval/reports/E14_heldout/theta_strict_f1_heldout.{pdf,png}   (NEW file)

Does NOT touch the related15 figure (E07_map_theta/theta_strict_f1.*).
Offline -- CSV reports only, no API/DB.

    python -m eval.silver.experiments.E14_heldout.plot_theta_curve_heldout
"""
from __future__ import annotations

import csv

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = _REPO_ROOT / "eval" / "reports"
E14_DIR = REPORTS / "E14_heldout"
E08B_DIR = REPORTS / "E08b_map_theta_shipped"
REJECT = "0.2"


def _heldout_curve():
    """(theta, strict_f1) from the newest E14 CSV that has a theta_frontier."""
    for path in sorted(E14_DIR.glob("heldout_*.csv"), reverse=True):
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
                if r.get("regime") == "theta_frontier"]
        if rows:
            rows.sort(key=lambda r: float(r["theta"]))
            return ([float(r["theta"]) for r in rows],
                    [float(r["strict_f1_optimal"]) for r in rows], path.name)
    raise RuntimeError("no theta_frontier rows found — run heldout_eval.py --theta-frontier")


def _related_curve() -> tuple[dict[float, float], str]:
    """theta -> strict_f1 from the related15 shipped theta-curve (E08b, escalate)."""
    paths = sorted(E08B_DIR.glob("sweep_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no sweep_*.csv in {E08B_DIR}")
    pts = {}
    for r in csv.DictReader(open(paths[-1], encoding="utf-8")):
        if (r.get("reject_theta") or "") == REJECT:
            pts[round(float(r["theta"]), 2)] = float(r["strict_f1_optimal"])
    return pts, paths[-1].name


def main() -> int:
    hx, hy, hf = _heldout_curve()
    rpts, rf = _related_curve()
    lo = min(hx)
    rx = sorted(x for x in rpts if x >= lo)   # match the heldout theta range
    ry = [rpts[x] for x in rx]
    print(f"heldout   ({hf}): {[ (t, round(s,4)) for t,s in zip(hx,hy) ]}")
    print(f"related15 ({rf}): {[ (t, round(rpts[t],4)) for t in rx ]}")

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(rx, ry, "--s", color="#888888", lw=1.3, ms=3.8, label="related15 (calibration)")
    ax.plot(hx, hy, "-o", color="#1f77b4", lw=1.8, ms=4.5, label="heldout15 (test)")

    ax.axvline(0.9, color="#d62728", lw=0.8, ls=":", alpha=0.7)
    ax.annotate(f"shipped $\\theta=0.9$\nstrict-F1 {hy[-1]:.4f}",
                xy=(0.9, hy[-1]), xytext=(0.50, hy[-1] - 0.085), fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

    ax.set_xlabel(r"agreement threshold $\theta$")
    ax.set_ylabel("strict-F1 (optimal matcher)")
    ax.set_title(r"Held-out $\theta$-sweep vs calibration "
                 r"(at $\theta_{\mathrm{reject}}=0.2$)", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = E14_DIR / f"theta_strict_f1_heldout.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
