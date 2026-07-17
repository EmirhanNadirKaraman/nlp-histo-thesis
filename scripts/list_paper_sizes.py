"""List every paper in the database with its text-element / figure / table count.

Useful for spotting outliers (textbook chapters, theses, etc.) before kicking
off a pipeline run. Sorts by ``n_text_elements`` descending by default — the
papers at the top are the ones most likely to dominate runtime and cost.

Usage
-----
    PYTHONPATH=. python scripts/list_paper_sizes.py

    # Top 20 longest only
    PYTHONPATH=. python scripts/list_paper_sizes.py --limit 20

    # Filter by minimum/maximum size
    PYTHONPATH=. python scripts/list_paper_sizes.py --min-elements 500
    PYTHONPATH=. python scripts/list_paper_sizes.py --max-elements 700

    # CSV for piping into other tools
    PYTHONPATH=. python scripts/list_paper_sizes.py --csv > sizes.csv

    # Sort by figures / tables instead
    PYTHONPATH=. python scripts/list_paper_sizes.py --sort figures
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from statistics import mean, median

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("list_paper_sizes")


def fetch_sizes() -> list[dict]:
    """Return one row per Document with text_element / figure / table counts."""
    from sqlalchemy import func

    from nlp_histo.database import get_db_connection
    from nlp_histo.database.models import Document, Figure, Table, TextElement

    db = get_db_connection()
    with db.session_scope() as session:
        # Pull counts as aggregate queries so we don't fetch 1M rows.
        te_counts = dict(
            session.query(TextElement.document_id, func.count(TextElement.id))
            .group_by(TextElement.document_id).all()
        )
        fig_counts = dict(
            session.query(Figure.document_id, func.count(Figure.id))
            .group_by(Figure.document_id).all()
        )
        tab_counts = dict(
            session.query(Table.document_id, func.count(Table.id))
            .group_by(Table.document_id).all()
        )
        docs = session.query(Document.id, Document.pmcid, Document.title).all()

    rows: list[dict] = []
    for doc_id, pmcid, title in docs:
        rows.append({
            "pmcid": pmcid,
            "n_text_elements": te_counts.get(doc_id, 0),
            "n_figures":       fig_counts.get(doc_id, 0),
            "n_tables":        tab_counts.get(doc_id, 0),
            "title":           (title or "")[:80],
        })
    return rows


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    idx = max(0, min(len(values) - 1, int(round(pct / 100.0 * (len(values) - 1)))))
    return sorted(values)[idx]


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("(no papers)")
        return
    te = [r["n_text_elements"] for r in rows]
    fig = [r["n_figures"] for r in rows]
    tab = [r["n_tables"] for r in rows]

    print()
    print(f"Summary across {len(rows)} papers:")
    print(f"  text_elements: min={min(te):>5d} mean={mean(te):>7.1f} "
          f"median={median(te):>5.0f} p90={_percentile(te, 90):>5d} "
          f"p99={_percentile(te, 99):>5d} max={max(te):>5d}")
    print(f"  figures:       min={min(fig):>5d} mean={mean(fig):>7.1f} "
          f"median={median(fig):>5.0f} p90={_percentile(fig, 90):>5d} "
          f"max={max(fig):>5d}")
    print(f"  tables:        min={min(tab):>5d} mean={mean(tab):>7.1f} "
          f"median={median(tab):>5.0f} p90={_percentile(tab, 90):>5d} "
          f"max={max(tab):>5d}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Show only the first N rows after sorting.")
    parser.add_argument("--sort",
                        choices=("elements", "figures", "tables", "pmcid"),
                        default="elements",
                        help="Sort key (default: elements, descending).")
    parser.add_argument("--ascending", action="store_true",
                        help="Reverse the sort order.")
    parser.add_argument("--min-elements", type=int, default=None,
                        help="Drop rows with fewer text elements than this.")
    parser.add_argument("--max-elements", type=int, default=None,
                        help="Drop rows with more text elements than this.")
    parser.add_argument("--csv", action="store_true",
                        help="Emit CSV to stdout instead of a table.")
    parser.add_argument("--no-title", action="store_true",
                        help="Omit the (truncated) title column.")
    args = parser.parse_args()

    logger.info("Querying database…")
    rows = fetch_sizes()
    logger.info("Loaded %d papers", len(rows))

    if args.min_elements is not None:
        rows = [r for r in rows if r["n_text_elements"] >= args.min_elements]
    if args.max_elements is not None:
        rows = [r for r in rows if r["n_text_elements"] <= args.max_elements]

    sort_key = {
        "elements": "n_text_elements",
        "figures":  "n_figures",
        "tables":   "n_tables",
        "pmcid":    "pmcid",
    }[args.sort]
    reverse = not args.ascending
    rows.sort(key=lambda r: r[sort_key], reverse=reverse)

    if args.limit is not None:
        display_rows = rows[:args.limit]
    else:
        display_rows = rows

    if args.csv:
        cols = ["pmcid", "n_text_elements", "n_figures", "n_tables"]
        if not args.no_title:
            cols.append("title")
        writer = csv.DictWriter(sys.stdout, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in display_rows:
            writer.writerow(r)
        return 0

    title_col = "" if args.no_title else "  title"
    title_w = 0 if args.no_title else 80
    print()
    print(f"{'pmcid':<48} {'tx_el':>6} {'figs':>5} {'tabs':>5}{title_col}")
    print("-" * (48 + 6 + 5 + 5 + 4 + title_w))
    for r in display_rows:
        title = "" if args.no_title else f"  {r['title']}"
        print(f"{r['pmcid']:<48} {r['n_text_elements']:>6d} "
              f"{r['n_figures']:>5d} {r['n_tables']:>5d}{title}")

    if args.limit is not None and args.limit < len(rows):
        print(f"\n({args.limit}/{len(rows)} shown — pass --limit 0 or omit "
              f"to see everything)")

    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
