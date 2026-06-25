#!/usr/bin/env python3
"""E09/E10 — combined strict-F1 vs cost plot: cascade frontier + single models.

Overlays, on one strict-F1-versus-cost plane:

  * the five-voter cascade's cost-Pareto frontier (the theta sweep, E09), and
  * each constituent voter run alone (E10), as points.

This is the visual statement of the RQ3 conclusion: single-Sonnet sits just
below-left of the cascade's quality point (cheaper at near-equal strict-F1), so
it lies on the Pareto front while the cascade does not; the cheaper single models
trail far below in quality.

Reads (latest timestamped CSV in each):
  eval/reports/E09_cost_quality/frontier_*.csv   (cascade theta sweep)
  eval/reports/E10_baselines/baselines_*.csv     (single-model baselines)
Writes:
  eval/reports/E09_cost_quality/cost_quality_plane.{pdf,png}

Offline -- CSV reports only, no API/DB.

    python -m eval.silver.experiments.E09_cost_quality.plot_cost_quality
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

REPORTS = _REPO_ROOT / "eval" / "reports"
E09_DIR = REPORTS / "E09_cost_quality"
E10_DIR = REPORTS / "E10_baselines"

_PRETTY = {
    "claude/claude-sonnet-4-6": "Sonnet",
    "claude/claude-haiku-4-5-20251001": "Haiku",
    "openai/gpt-4.1-mini": "GPT-4.1-mini",
    "gemini/gemini-2.5-flash": "Gemini-Flash",
    "gemini/gemini-2.5-flash-lite": "Gemini-Flash-Lite",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "openai/gpt-4.1-nano": "GPT-4.1-nano",
}


def _latest(d: Path, pat: str) -> Path:
    c = sorted(d.glob(pat))
    if not c:
        raise FileNotFoundError(f"no {pat} in {d}")
    return c[-1]


def _pareto_frontier(path: Path):
    """Cost-Pareto frontier from the theta sweep: keep points that, going from
    cheap to expensive, strictly improve strict-F1 (maximise F1, minimise cost)."""
    pts = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            pts.append({
                "cost": float(r["cost_per_chunk"]),
                "sf1": float(r["strict_f1_optimal"]),
                "op": (r.get("operating_point") or "").strip(),
            })
        except (KeyError, ValueError, TypeError):
            continue
    pts.sort(key=lambda p: (p["cost"], -p["sf1"]))
    frontier, best = [], -1.0
    for p in pts:
        if p["sf1"] > best + 1e-12:
            frontier.append(p)
            best = p["sf1"]
    return frontier


def _singles(path: Path):
    """Single-model points (exclude the cascade row -- it is the frontier top)."""
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if (r.get("system") or "").upper().startswith("CASCADE"):
            continue
        model = (r.get("model") or "").strip()
        try:
            out.append({
                "name": _PRETTY.get(model, model),
                "cost": float(r["cost_per_chunk"]),
                "sf1": float(r["strict_f1_optimal"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def main() -> int:
    fr = _pareto_frontier(_latest(E09_DIR, "frontier_*.csv"))
    sg = _singles(_latest(E10_DIR, "baselines_*.csv"))
    if not fr or not sg:
        print("missing frontier or baseline points")
        return 1
    print(f"frontier points: {len(fr)} | single models: {len(sg)}")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    # Cascade cost-Pareto frontier (line + markers).
    ax.plot([p["cost"] for p in fr], [p["sf1"] for p in fr], "-o",
            color="#1f77b4", lw=1.8, ms=4.5, zorder=3,
            label="5-voter cascade ($\\theta$ frontier)")
    # Spread the operating-point labels so the top-right cluster
    # (knee/quality, next to single-Sonnet) does not overlap.
    _OP_OFFSET = {  # (dx, dy, ha)
        "economy": (0, 8, "center"),
        "knee": (9, -13, "left"),   # bottom-right, off the rising frontier line
        "quality": (5, 7, "left"),
    }
    for p in fr:
        if p["op"] in _OP_OFFSET:
            dx, dy, ha = _OP_OFFSET[p["op"]]
            ax.annotate(p["op"], xy=(p["cost"], p["sf1"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color="#1f77b4", ha=ha)

    # Single models: Sonnet highlighted, the rest as one group.
    others = [s for s in sg if s["name"] != "Sonnet"]
    sonnet = next((s for s in sg if s["name"] == "Sonnet"), None)
    if others:
        ax.scatter([s["cost"] for s in others], [s["sf1"] for s in others],
                   color="#555555", marker="s", s=30, zorder=4, label="single models")
    if sonnet:
        ax.scatter([sonnet["cost"]], [sonnet["sf1"]], color="#d62728", marker="D",
                   s=52, zorder=5, label="single Sonnet")
    for s in sg:
        if s["name"] == "Sonnet":  # label to the LEFT, away from knee/quality
            ax.annotate(s["name"], xy=(s["cost"], s["sf1"]),
                        xytext=(-8, 1), textcoords="offset points",
                        fontsize=7.5, color="#d62728", ha="right")
        else:
            ax.annotate(s["name"], xy=(s["cost"], s["sf1"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color="#333333")

    ax.set_xscale("log")
    # Headroom so the top-right corner labels (quality, Sonnet) don't clip.
    _all_sf1 = [p["sf1"] for p in fr] + [s["sf1"] for s in sg]
    ax.set_ylim(min(_all_sf1) - 0.02, max(_all_sf1) + 0.028)
    ax.set_xlim(right=34)
    ax.set_xlabel("cost per chunk (price-weighted invocations, log scale)")
    ax.set_ylabel("strict-F1 (optimal matcher)")
    ax.set_title("Cost--quality plane: 5-voter cascade frontier vs single models",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = E09_DIR / f"cost_quality_plane.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
