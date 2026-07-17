#!/usr/bin/env python
"""
Regenerate the knowledge_extraction pipeline cost report from a saved usage JSONL.

Inputs
------
- ``llm_usage_records.jsonl`` — the canonical record stream the runner writes.
- a price book (``configs/model_prices.json`` by default).

Outputs
-------
- ``<out_dir>/cost_report.md``
- ``<out_dir>/cost_report.json``

The runtime path writes these automatically at the end of every pipeline run.
Use this script when you want to (a) replay an old run, (b) re-run the report
with a different price book, or (c) merge JSONLs from a multi-paper batch.

Usage
-----
    python scripts/estimate_cost_savings.py \\
        --usage out/cost/<run_id>/llm_usage_records.jsonl \\
        --prices configs/model_prices.json \\
        --out out/cost/<run_id>

Multiple --usage args are accepted and concatenated, so you can regenerate
one aggregate report across many runs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make repo root importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nlp_histo.pipeline.stages.knowledge_extraction.costing import (  # noqa: E402
    PriceBook, load_records, write_cost_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usage", required=True, action="append",
        help="Path(s) to llm_usage_records.jsonl. May be passed multiple times.",
    )
    parser.add_argument(
        "--prices", default=None,
        help="Path to model_prices.json. Defaults to configs/model_prices.json.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for cost_report.md and cost_report.json.",
    )
    parser.add_argument(
        "--no-markdown", action="store_true", help="Skip writing cost_report.md",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="Skip writing cost_report.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )

    records = []
    for p in args.usage:
        path = Path(p)
        chunk = load_records(path)
        logging.info("loaded %d records from %s", len(chunk), path)
        records.extend(chunk)
    if not records:
        logging.error("no records loaded — nothing to do")
        return 1

    book = PriceBook.load(args.prices)
    paths = write_cost_reports(
        records, book, Path(args.out),
        write_markdown=not args.no_markdown,
        write_json=not args.no_json,
    )
    for kind, path in paths.items():
        logging.info("wrote %s → %s", kind, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
