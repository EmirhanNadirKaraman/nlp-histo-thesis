"""Layer A — plot the grounding-threshold sweep results.

Reads ``eval/results/grounding_sweep.csv`` (produced by
``eval/sweeps/grounding.py``) for the retention curve, and re-loads the
summary artifacts via ``gather_findings_with_grounding`` for the score
distribution histogram. Writes two PNG files to ``eval/results/``.

Layer A invariant: pure-stdlib + matplotlib. No API calls, no NLI re-inference,
no pipeline re-execution. The matplotlib backend is forced to ``Agg`` so the
script works headless (CI, SSH, etc.) and never opens an interactive window.

Usage::

    python eval/sweeps/grounding_plot.py \\
        [--input out/summaries] \\
        [--sweep-csv eval/results/grounding_sweep.csv] \\
        [--out-retention eval/results/grounding_retention.png] \\
        [--out-distribution eval/results/grounding_score_distribution.png] \\
        [--threshold 0.5]

``--threshold`` draws a dashed vertical reference line on both plots so the
current producer threshold is visible at a glance.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

# Force headless matplotlib before pyplot import so the script works without
# a display server. Must come before any seaborn / plotnine etc.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the gather helper from the sweep library.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from eval.sweeps import _lib as sweeps_lib  # noqa: E402

DEFAULT_INPUT = Path("out/summaries")
DEFAULT_SWEEP_CSV = Path("eval/results/grounding_sweep.csv")
DEFAULT_SWEEP_MD = Path("eval/results/grounding_sweep.md")
DEFAULT_OUT_RETENTION = Path("eval/results/grounding_retention.png")
DEFAULT_OUT_DISTRIBUTION = Path("eval/results/grounding_score_distribution.png")

MD_PLOT_BEGIN = "<!-- BEGIN: grounding_plot.py auto-generated -->"
MD_PLOT_END = "<!-- END: grounding_plot.py auto-generated -->"

logger = logging.getLogger("eval.sweeps.grounding_plot")


# ---------------------------------------------------------------------------
# CSV reader (skips leading ``#`` comment metadata lines)
# ---------------------------------------------------------------------------


def read_sweep_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Sweep CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        non_comment = (line for line in f if not line.startswith("#"))
        reader = csv.DictReader(non_comment)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                if v == "" or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


# ---------------------------------------------------------------------------
# Plot 1 — retention curve
# ---------------------------------------------------------------------------


def plot_retention(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    producer_threshold: float | None = None,
) -> None:
    if not rows:
        raise ValueError("No sweep rows to plot.")
    rows_sorted = sorted(rows, key=lambda r: r["threshold"])
    thresholds = [r["threshold"] for r in rows_sorted]
    kept_pct = [r["kept_pct_of_scored"] for r in rows_sorted]
    mean_kept = [r.get("mean_grounding_score_kept") for r in rows_sorted]
    mean_rejected = [r.get("mean_grounding_score_rejected") for r in rows_sorted]

    fig, ax_left = plt.subplots(figsize=(8.5, 5.0))

    color_left = "tab:blue"
    ax_left.plot(thresholds, kept_pct, color=color_left, marker="o",
                 linewidth=2, label="kept_pct_of_scored")
    ax_left.set_xlabel("Grounding threshold")
    ax_left.set_ylabel("Retention (% of scored)", color=color_left)
    ax_left.tick_params(axis="y", labelcolor=color_left)
    ax_left.grid(True, alpha=0.3)
    ax_left.set_ylim(0, 105)
    ax_left.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)

    ax_right = ax_left.twinx()
    color_kept = "tab:green"
    color_rej = "tab:red"
    ax_right.plot(thresholds, mean_kept, color=color_kept, marker="s",
                  linestyle="--", linewidth=1.5, label="mean_score_kept")
    # Drop None entries before plotting (matplotlib accepts them but emits warnings).
    if any(v is not None for v in mean_rejected):
        ax_right.plot(thresholds, mean_rejected, color=color_rej, marker="^",
                      linestyle="--", linewidth=1.5, label="mean_score_rejected")
    ax_right.set_ylabel("Mean grounding score")
    ax_right.set_ylim(0, 1.05)

    if producer_threshold is not None:
        ax_left.axvline(producer_threshold, color="gray", linestyle=":",
                        linewidth=1.2,
                        label=f"producer threshold ({producer_threshold:g})")

    # Merge legends from both axes.
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right,
                   loc="lower left", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Grounding-threshold retention sweep\n"
        "(this is a retention plot, not an accuracy plot)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ``metadata={...}`` strips matplotlib's default timestamp embedded in the
    # PNG so repeated runs produce byte-identical output (helpful for diffs).
    fig.savefig(out_path, dpi=150, metadata={"Software": "matplotlib"})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — score distribution histogram
# ---------------------------------------------------------------------------


def plot_score_distribution(
    records: list[sweeps_lib.FindingRecord],
    out_path: Path,
    *,
    producer_threshold: float | None = None,
    bins: int = 20,
) -> None:
    scored = [r.grounding_score for r in records if r.grounding_score is not None]
    if not scored:
        raise ValueError("No scored findings to plot.")

    # Separate by producer-side keep/reject status so the bimodal pattern is
    # readable.
    kept_scores = [r.grounding_score for r in records
                   if r.grounding_score is not None and r.kept_at_producer_threshold]
    rejected_scores = [r.grounding_score for r in records
                       if r.grounding_score is not None and not r.kept_at_producer_threshold]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    edges = [i / bins for i in range(bins + 1)]
    ax.hist(
        [kept_scores, rejected_scores],
        bins=edges,
        stacked=True,
        color=["tab:green", "tab:red"],
        alpha=0.85,
        label=[
            f"kept by producer (N={len(kept_scores)})",
            f"rejected by producer (N={len(rejected_scores)})",
        ],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Grounding score")
    ax.set_ylabel("Finding count")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    if producer_threshold is not None:
        ax.axvline(producer_threshold, color="gray", linestyle=":",
                   linewidth=1.5,
                   label=f"producer threshold ({producer_threshold:g})")

    ax.legend(loc="upper center", fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Distribution of {len(scored)} persisted grounding scores\n"
        "(stacked by producer-side keep/reject)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, metadata={"Software": "matplotlib"})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown injection (idempotent)
# ---------------------------------------------------------------------------


def inject_plot_refs_into_markdown(
    md_path: Path,
    *,
    retention_png: Path,
    distribution_png: Path,
) -> bool:
    """Insert (or replace) a fenced block in ``md_path`` that embeds both
    PNGs as relative-path image references.

    Idempotent: re-running the plotter replaces the block in place rather
    than appending duplicates. The fence tokens are HTML comments
    (:data:`MD_PLOT_BEGIN` / :data:`MD_PLOT_END`) so they're invisible when
    the markdown is rendered.

    Returns ``True`` when the file was changed, ``False`` when no update
    was needed (markdown missing → returns ``False`` and logs a warning).
    """
    if not md_path.exists():
        logger.warning(
            "Markdown report %s not found — skipping plot embedding. "
            "Run `python eval/sweeps/grounding.py` first to produce it.",
            md_path,
        )
        return False

    import os as _os
    rel_ret = _os.path.relpath(retention_png, start=md_path.parent)
    rel_dist = _os.path.relpath(distribution_png, start=md_path.parent)

    block_lines = [
        MD_PLOT_BEGIN,
        "",
        "## Plots",
        "",
        f"![Retention curve at varying thresholds]({rel_ret})",
        "",
        "*Retention curve: blue line is the fraction of scored findings the "
        "filter would keep at each candidate threshold; green and red lines "
        "are the mean grounding score of survivors and rejects respectively.*",
        "",
        f"![Distribution of grounding scores]({rel_dist})",
        "",
        "*Distribution: stacked histogram of every persisted grounding "
        "score, coloured by what the producer run actually did at write "
        "time. A bimodal distribution means the filter has little gray "
        "zone to discriminate.*",
        "",
        MD_PLOT_END,
    ]
    new_block = "\n".join(block_lines)

    text = md_path.read_text(encoding="utf-8")
    if MD_PLOT_BEGIN in text and MD_PLOT_END in text:
        # Replace existing block in place.
        prefix, _, rest = text.partition(MD_PLOT_BEGIN)
        _, _, suffix = rest.partition(MD_PLOT_END)
        new_text = prefix + new_block + suffix
    else:
        # Insert before "## Notes" if present, else append at end.
        anchor = "\n## Notes"
        if anchor in text:
            new_text = text.replace(anchor, "\n" + new_block + "\n" + anchor, 1)
        else:
            sep = "" if text.endswith("\n") else "\n"
            new_text = text + sep + "\n" + new_block + "\n"

    if new_text == text:
        return False
    md_path.write_text(new_text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--out-retention", type=Path, default=DEFAULT_OUT_RETENTION)
    parser.add_argument("--out-distribution", type=Path, default=DEFAULT_OUT_DISTRIBUTION)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Drawn as a dashed reference line on both plots.")
    parser.add_argument(
        "--sweep-md", type=Path, default=DEFAULT_SWEEP_MD,
        help="Markdown report to embed the plots into (idempotent — repeated "
             "runs replace the auto-generated block rather than appending). "
             "Pass an empty string to skip the embedding step.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # --- Plot 1: retention curve from CSV ---
    try:
        rows = read_sweep_csv(args.sweep_csv)
    except FileNotFoundError:
        logger.error(
            "Sweep CSV not found at %s. Run `python eval/sweeps/grounding.py` first.",
            args.sweep_csv,
        )
        return 1
    if not rows:
        logger.error("Sweep CSV %s has no data rows.", args.sweep_csv)
        return 1
    plot_retention(rows, args.out_retention, producer_threshold=args.threshold)
    logger.info("Wrote retention curve → %s (%d threshold points)",
                args.out_retention, len(rows))

    # --- Plot 2: score distribution from summary artifacts ---
    summaries_dir = args.input / "summaries"
    if not summaries_dir.exists():
        logger.error("Summaries directory not found: %s", summaries_dir)
        return 1
    records, _warnings = sweeps_lib.gather_findings_with_grounding(summaries_dir)
    if not records:
        logger.error("No findings gathered from %s.", summaries_dir)
        return 1
    plot_score_distribution(records, args.out_distribution,
                            producer_threshold=args.threshold)
    logger.info(
        "Wrote score distribution → %s (%d findings, %d with score)",
        args.out_distribution, len(records),
        sum(1 for r in records if r.grounding_score is not None),
    )

    # --- Embed plot references into the markdown report (idempotent) ---
    sweep_md = args.sweep_md
    if sweep_md and str(sweep_md):
        changed = inject_plot_refs_into_markdown(
            sweep_md,
            retention_png=args.out_retention,
            distribution_png=args.out_distribution,
        )
        if changed:
            logger.info("Embedded plot references into %s", sweep_md)
        else:
            logger.info("Plot block in %s already up to date (no change)", sweep_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
