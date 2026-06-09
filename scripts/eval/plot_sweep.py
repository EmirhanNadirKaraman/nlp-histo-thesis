#!/usr/bin/env python3
"""Generate the document-extraction sweep figures from a score_pdf_variants
JSON report (`--json-out`).

Produces two PNGs:
  1. <out>/sweep_pr_scatter.png   — table crop precision vs recall, one point
     per variant, coloured by detector family, winner annotated (the §9.1
     figure the thesis plan calls for).
  2. <out>/sweep_strict_f1_bar.png — table strict F1 per variant, sorted,
     winner highlighted.

Tables (not figures) are plotted because figure cropping is detector-invariant
— only the table dimension discriminates between configs.

Usage:
    python3 scripts/eval/plot_sweep.py \
        [reports/post_exclusion_PMC11863705/variants_PR.json] \
        [--out-dir reports/figures] [--winner 18_hybrid_best_family_fixes_footnote_expand_1_2]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_JSON = _REPO / "reports" / "post_exclusion_PMC11863705" / "variants_PR.json"
_WINNER = "18_hybrid_best_family_fixes_footnote_expand_1_2"

_FAMILY_COLOR = {"docling": "#1f77b4", "tatr": "#d62728", "hybrid": "#2ca02c"}


def family(variant: str) -> str:
    v = variant.lower()
    if "docling" in v:
        return "docling"
    if "tatr" in v:
        return "tatr"
    return "hybrid"  # hybrid_* and best_* (final family is hybrid)


def short(variant: str) -> str:
    """Compact label: leading number only (e.g. '18')."""
    m = re.match(r"(\d+)", variant)
    return m.group(1) if m else variant


def load_tables(json_path: Path) -> list[dict]:
    rows = json.loads(json_path.read_text())
    return [r for r in rows if r["kind"] == "tables"]


def plot_pr_scatter(tables: list[dict], out: Path, winner: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in tables:
        v = r["variant"]
        crop = r["by_dim"]["crop"]
        p, rec = crop["precision"], crop["recall"]
        fam = family(v)
        is_win = v == winner
        ax.scatter(rec, p, s=180 if is_win else 70,
                   c=_FAMILY_COLOR[fam],
                   marker="*" if is_win else "o",
                   edgecolors="black", linewidths=1.1 if is_win else 0.4,
                   zorder=3 if is_win else 2)
        ax.annotate(short(v), (rec, p), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    # F1 iso-contours for context
    import numpy as np
    rr = np.linspace(0.5, 1.0, 100)
    for f1 in (0.8, 0.85, 0.9):
        pp = (f1 * rr) / (2 * rr - f1)
        ok = (pp > 0) & (pp <= 1.0)
        ax.plot(rr[ok], pp[ok], "--", color="grey", lw=0.5, alpha=0.6)
        if ok.any():
            ax.annotate(f"F1={f1}", (rr[ok][-1], pp[ok][-1]),
                        fontsize=6, color="grey")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f)
               for f, c in _FAMILY_COLOR.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", color="black",
                              label=f"winner ({short(winner)})"))
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_xlabel("Table crop recall")
    ax.set_ylabel("Table crop precision")
    ax.set_title("Document-extraction sweep — table crop precision vs recall")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def plot_strict_bar(tables: list[dict], out: Path, winner: str) -> None:
    rows = sorted(tables, key=lambda r: r["strict"]["f1"] or 0, reverse=True)
    labels = [short(r["variant"]) for r in rows]
    vals = [(r["strict"]["f1"] or 0) * 100 for r in rows]
    colors = ["#ff7f0e" if r["variant"] == winner else _FAMILY_COLOR[family(r["variant"])]
              for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Table strict F1 (%)")
    ax.set_title("Document-extraction sweep — table strict F1 by variant "
                 "(orange = winner)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", type=Path, default=_DEFAULT_JSON)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "reports" / "figures")
    ap.add_argument("--winner", default=_WINNER)
    args = ap.parse_args()

    if not args.json.exists():
        print(f"error: {args.json} not found — run score_pdf_variants.py "
              f"--json-out first.")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(args.json)
    print(f"loaded {len(tables)} variants from {args.json}")
    plot_pr_scatter(tables, args.out_dir / "sweep_pr_scatter.png", args.winner)
    plot_strict_bar(tables, args.out_dir / "sweep_strict_f1_bar.png", args.winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
