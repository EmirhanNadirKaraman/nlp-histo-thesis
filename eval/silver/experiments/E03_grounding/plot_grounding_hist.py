#!/usr/bin/env python3
"""E03 — grounding entailment-score distribution (the bimodality figure).

Reconstructs the per-bin score distribution directly from the E03 grounding
**sweep CSV** -- no recompute. The sweep records ``n_kept`` (findings with
entailment score >= threshold) at each threshold in {0.0, 0.1, ..., 0.9}, so the
count in bin ``[t, t+0.1)`` is just ``n_kept(t) - n_kept(t+0.1)`` (the top bin
``[0.9, 1.0]`` is ``n_kept(0.9)``). The distribution is bimodal: a large mode at
>= 0.9 (well-grounded) and a separated tail < 0.3 (claim not entailed by its
cited paragraph); the pinned 0.5 threshold drops the tail.

Reads the latest ``eval/reports/E03_grounding/sweep_*.csv`` and writes
``grounding_hist.{pdf,png}`` there. No API/DB/NLI.

    python -m eval.silver.experiments.E03_grounding.plot_grounding_hist
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = _REPO_ROOT / "eval" / "reports" / "E03_grounding"


def _latest_sweep() -> Path:
    c = sorted(OUT_DIR.glob("sweep_*.csv"))
    if not c:
        raise FileNotFoundError(f"no sweep_*.csv in {OUT_DIR}")
    return c[-1]


def _kept_by_threshold(path: Path) -> dict[float, int]:
    """threshold -> n_kept (cumulative count of findings with score >= threshold)."""
    kept: dict[float, int] = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        t = r["grounding_threshold"]
        if t in ("None", "", "none"):
            continue
        kept[round(float(t), 3)] = int(r["n_kept"])
    return kept


def main() -> int:
    kept = _kept_by_threshold(_latest_sweep())
    total = kept[0.0]
    # Per-bin counts by differencing the cumulative n_kept.
    counts, centers = [], []
    for i in range(10):
        t = round(0.1 * i, 1)
        nxt = round(t + 0.1, 1)
        hi = kept.get(nxt)  # None above 0.9 -> top bin is everything >= 0.9
        counts.append(kept[t] - (hi if hi is not None else 0))
        centers.append(round(t + 0.05, 2))
    print("bin counts:", dict(zip([round(0.1 * i, 1) for i in range(10)], counts)),
          "sum =", sum(counts))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(centers, counts, width=0.092, color="#4c78a8", edgecolor="white", zorder=2)
    ax.axvline(0.5, color="#d62728", lw=1.4, ls="--", zorder=3,
               label=f"pinned threshold 0.5 (retain {kept[0.5] / total * 100:.1f}%)")
    ax.set_xlabel("grounding entailment score (cited paragraph $\\rightarrow$ claim)")
    ax.set_ylabel("number of MAP findings")
    # No in-plot title: this panel is a subfigure; the subcaption labels it.
    ax.set_xticks([round(0.1 * i, 1) for i in range(11)])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"grounding_hist.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
