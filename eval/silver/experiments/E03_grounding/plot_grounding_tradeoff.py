#!/usr/bin/env python3
"""E03 — grounding trade-off curve vs threshold (the "deliberate trade" figure).

Reads the E03 grounding sweep CSV and plots how the grounding threshold
trades silver agreement for groundedness:

  * retention  (fraction of MAP findings kept) -- falls
  * strict-F1  (vs Opus silver)                -- falls
  * precision  (vs Opus silver)                -- essentially flat

i.e. raising the threshold drops *recall*, not precision: the findings it removes
are ungrounded, not silver-false. The pinned 0.5 threshold is marked.

No recompute -- reads the existing sweep report only. No API/DB/NLI.

    python -m eval.silver.experiments.E03_grounding.plot_grounding_tradeoff
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


def _series(path: Path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["grounding_threshold"] not in ("None", "", "none")]
    rows.sort(key=lambda r: float(r["grounding_threshold"]))
    th = [float(r["grounding_threshold"]) for r in rows]
    rec = [float(r["recall_optimal"]) for r in rows]
    f1 = [float(r["strict_f1_optimal"]) for r in rows]
    prec = [float(r["precision_optimal"]) for r in rows]
    return th, rec, f1, prec


def main() -> int:
    th, rec, f1, prec = _series(_latest_sweep())

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.plot(th, rec, "-o", color="#4c78a8", lw=1.7, ms=4.5, label="recall (vs silver)")
    ax.plot(th, prec, "-s", color="#54a24b", lw=1.7, ms=4.0, label="precision (vs silver)")
    ax.plot(th, f1, "-^", color="#e45756", lw=1.7, ms=4.5, label="strict-F1 (vs silver)")

    ax.axvline(PINNED, color="#888888", lw=1.1, ls="--", zorder=1)
    ax.text(PINNED + 0.012, 0.62, "pinned 0.5", color="#555555", fontsize=8, rotation=90,
            va="bottom")

    ax.set_xlabel("grounding threshold")
    ax.set_ylabel("score (vs Opus silver)")
    # No in-plot title: this panel is a subfigure; the subcaption labels it.
    ax.set_ylim(0.60, 1.00)
    ax.set_xlim(-0.02, 0.92)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"grounding_tradeoff.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
