#!/usr/bin/env python3
"""E03 — combined grounding figure: (a) score distribution + (b) threshold effect.

One 1x2 figure, both panels reconstructed from the E03 grounding **sweep CSV**
(no recompute):

  (a) distribution of grounding entailment scores -- bimodal: a large mode at
      >= 0.9 (well-grounded) and a separated tail < 0.3 (claim not entailed by
      its cited paragraph). Bin counts = differences of the cumulative n_kept.
  (b) precision / recall / strict-F1 against Opus silver vs the threshold:
      raising it trades recall (and strict-F1) for groundedness while precision
      stays flat -- the filter drops recall, not precision.

The pinned 0.5 threshold is marked in both panels.

    python -m eval.silver.experiments.E03_grounding.plot_grounding_combined

No API/DB/NLI -- reads the sweep report only.
"""
from __future__ import annotations

import csv
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = _REPO_ROOT / "eval" / "reports" / "E03_grounding"
PINNED = 0.5


def _latest_sweep() -> Path:
    c = sorted(OUT_DIR.glob("sweep_*.csv"))
    if not c:
        raise FileNotFoundError(f"no sweep_*.csv in {OUT_DIR}")
    return c[-1]


def _load(path: Path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["grounding_threshold"] not in ("None", "", "none")]
    rows.sort(key=lambda r: float(r["grounding_threshold"]))
    kept = {round(float(r["grounding_threshold"]), 3): int(r["n_kept"]) for r in rows}
    th = [float(r["grounding_threshold"]) for r in rows]
    rec = [float(r["recall_optimal"]) for r in rows]
    f1 = [float(r["strict_f1_optimal"]) for r in rows]
    prec = [float(r["precision_optimal"]) for r in rows]
    return kept, th, rec, f1, prec


def main() -> int:
    kept, th, rec, f1, prec = _load(_latest_sweep())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.8, 3.9))

    # ── (a) score distribution (bimodal) ──────────────────────────────────────
    counts, centers = [], []
    for i in range(10):
        t = round(0.1 * i, 1)
        hi = kept.get(round(t + 0.1, 1))
        counts.append(kept[t] - (hi if hi is not None else 0))
        centers.append(round(t + 0.05, 2))
    axL.bar(centers, counts, width=0.092, color="#4c78a8", edgecolor="white", zorder=2)
    axL.axvline(PINNED, color="#d62728", lw=1.3, ls="--", zorder=3,
                label=f"pinned 0.5 (retain {kept[0.5] / kept[0.0] * 100:.1f}%)")
    axL.set_xlabel("grounding entailment score (cited paragraph $\\rightarrow$ claim)")
    axL.set_ylabel("number of MAP findings")
    axL.set_xticks([round(0.1 * i, 1) for i in range(11)])
    axL.set_title("(a) Score distribution (bimodal)", fontsize=9.5)
    axL.grid(True, axis="y", alpha=0.25)
    axL.legend(fontsize=8, loc="upper center")

    # ── (b) precision / recall / strict-F1 vs threshold ───────────────────────
    axR.plot(th, rec, "-o", color="#4c78a8", lw=1.7, ms=4.5, label="recall (vs silver)")
    axR.plot(th, prec, "-s", color="#54a24b", lw=1.7, ms=4.0, label="precision (vs silver)")
    axR.plot(th, f1, "-^", color="#e45756", lw=1.7, ms=4.5, label="strict-F1 (vs silver)")
    axR.axvline(PINNED, color="#888888", lw=1.1, ls="--", zorder=1)
    axR.text(PINNED + 0.013, 0.605, "pinned 0.5", color="#555555", fontsize=8,
             rotation=90, va="bottom")
    axR.set_xlabel("grounding threshold")
    axR.set_ylabel("score (vs Opus silver)")
    axR.set_ylim(0.60, 1.00)  # headroom above the precision line for the legend
    axR.set_xlim(-0.02, 0.92)
    axR.set_title("(b) Effect on the silver metric", fontsize=9.5)
    axR.grid(True, alpha=0.25)
    axR.legend(fontsize=8, loc="upper right")

    fig.tight_layout(w_pad=4.0)  # extra horizontal gap between the two panels
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"grounding_combined.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
