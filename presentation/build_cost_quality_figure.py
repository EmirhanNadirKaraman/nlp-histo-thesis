#!/usr/bin/env python3
"""Regenerate the cost–quality figure with the `balanced` operating point.

`assets/fig_cost_quality_plane.png` predates the 2026-08-09 ε-constraint result, so it shows
three operating points instead of four. It also encodes the single-Sonnet baseline in red, which
is a reserved status colour — the comparison it carries is categorical, not an alert.

Form: the data's job is the *relationship* between two measures across a frontier, so a
cost-vs-quality scatter with the cascade's θ-curve drawn as a connected line is the right form.
Colour job is categorical: cascade frontier vs the one baseline that matters vs everything else.
Palette is TUM blue + TUM orange (validated: all six checks pass, worst adjacent CVD ΔE 26.8);
grey is recessive only — axes, grid, and the non-focal single models — never a series identity.

Reads the frozen E09 frontier plus RESULTS.md's single-model numbers. No network, no API, no cost.

    python3 presentation/build_cost_quality_figure.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
REPO = R.parent
FRONTIER = REPO / "eval/reports/E09_cost_quality/frontier_20260809T133949.csv"
OUT = R / "assets/fig_cost_quality_plane_v2.png"

TUM_BLUE = "#0065BD"
TUM_ORANGE = "#E37222"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
RECESSIVE = "#9a9a9a"

# E10 single-model baselines (RESULTS.md). Only Sonnet is focal — it is the one that
# ties the cascade, which is the thesis's central negative result.
SINGLES = [
    ("GPT-4.1-nano", 0.50, 0.4570),
    ("GPT-4o-mini", 0.75, 0.5394),
    ("Gemini-Flash-Lite", 0.50, 0.5721),
    ("Gemini-Flash", 2.80, 0.5761),
    ("GPT-4.1-mini", 2.00, 0.5957),
    ("Claude Haiku", 4.80, 0.6558),
]
SONNET = ("single Claude Sonnet", 18.00, 0.7129)

# Label offsets tuned by eye after rendering (the validator checks colour, not layout).
OP_LABELS = {
    "economy": (-4, -16, "left"),
    "balanced": (8, -14, "left"),
    "knee": (6, -16, "left"),
    "quality": (-10, 10, "left"),
}


def load_frontier():
    rows = []
    for r in csv.DictReader(FRONTIER.open()):
        rows.append({
            "cost": float(r["cost_per_chunk"]),
            "f1": float(r["strict_f1_optimal"]),
            "op": r.get("operating_point", ""),
        })
    best = {}
    for r in rows:                                  # one point per cost
        k = round(r["cost"], 4)
        if k not in best or r["f1"] > best[k]["f1"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: r["cost"])


def main() -> int:
    if not FRONTIER.exists():
        print(f"error: {FRONTIER} not found", file=sys.stderr)
        return 1
    fr = load_frontier()

    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=200)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # recessive grid, 2px-equivalent thin marks
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cfcfcf")

    # non-focal single models — recessive grey, labelled directly so identity is never colour-alone
    for name, cost, f1 in SINGLES:
        ax.scatter(cost, f1, s=52, marker="s", color=RECESSIVE, zorder=3,
                   edgecolors="white", linewidths=1.4)
        ax.annotate(name, (cost, f1), textcoords="offset points", xytext=(8, -3),
                    fontsize=8.5, color=MUTED, va="center")

    # the cascade frontier
    ax.plot([r["cost"] for r in fr], [r["f1"] for r in fr],
            color=TUM_BLUE, linewidth=2.0, marker="o", markersize=6.5,
            markeredgecolor="white", markeredgewidth=1.4, zorder=4,
            label="cascade — θ frontier")

    # the four operating points, direct-labelled
    for r in fr:
        for tag in [t for t in r["op"].split("|") if t]:
            dx, dy, ha = OP_LABELS.get(tag, (8, 8, "left"))
            ax.annotate(tag, (r["cost"], r["f1"]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=10, color=TUM_BLUE,
                        fontweight="bold", ha=ha)

    # the baseline that carries the argument
    ax.scatter(SONNET[1], SONNET[2], s=150, marker="D", color=TUM_ORANGE, zorder=5,
               edgecolors="white", linewidths=1.8, label="single Claude Sonnet")
    ax.annotate("single Sonnet\nties the cascade, 24% cheaper",
                (SONNET[1], SONNET[2]), textcoords="offset points", xytext=(-14, 16),
                fontsize=9.5, color=TUM_ORANGE, fontweight="bold", ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("cost per chunk  (price-weighted invocations, log scale)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("strict-F1  (Hungarian matcher)", fontsize=10.5, color=INK)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.set_ylim(0.43, 0.75)

    leg = ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    leg.get_frame().set_edgecolor("#e0e0e0")
    leg.get_frame().set_facecolor("white")
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(REPO)}  ({len(fr)} frontier points, "
          f"{len(SINGLES) + 1} single models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
