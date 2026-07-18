#!/usr/bin/env python3
"""E01 — flatten the document-extraction sweep results to a tidy CSV.

DESCRIPTIVE (RQ1, §9.1), not a calibration. The sweep itself is run by
``scripts/eval/run_all_sweeps.py`` (generate crops) + ``scripts/eval/
score_pdf_variants.py`` (score against ``eval/label_rubric.yaml``); the scorer
emits a nested ``*_PR.json`` (one record per variant x kind, each holding
``by_dim`` / ``strict`` / ``counts`` blocks) plus a human-readable ``*_PR.md``.

This module is a pure file transform: it reads that nested JSON and writes a
flat, one-row-per-(variant, kind) CSV — the same tabular view the other
numbered experiments (E02-E14) already ship alongside their narrative, so E01
has a machine-readable artefact too. No API, no model loads, no pipeline import.

The CSV columns are, per row:
  variant, kind,
  {crop,caption,footnote,mask}_{tp,fp,fn,precision,recall,f1},
  strict_{tp,fp,fn,precision,recall,f1},
  emitted, labelled, unlabelled, unrecognised, icons

``footnote`` is ``null`` for figure rows (figures carry no body footnote); those
cells are left blank. Numbers are copied verbatim from the JSON — this is a
lossless reshape, every cell cross-checks against the ``*_PR.md`` table.

Usage::

  # default: flatten the pinned 27-PDF artefact next to itself (<stem>.csv)
  python -m eval.silver.experiments.E01_doc_extraction.flatten_to_csv

  # any sweep JSON, explicit output path
  python -m eval.silver.experiments.E01_doc_extraction.flatten_to_csv \
      --json eval/reports/E01_doc_extraction/<run>_PR.json \
      --csv-out eval/reports/E01_doc_extraction/<run>_PR.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

# Pure file transform — define the reports dir locally rather than importing
# eval.silver.analysis.map_theta_sweep (which would pull in the summarisation pipeline
# and its load-time stdout noise) just for a Path constant.
_REPORTS_DIR = _REPO_ROOT / "eval" / "reports" / "E01_doc_extraction"
_DEFAULT_JSON = _REPORTS_DIR / "figtable_extraction_sweep_rerun_27pdf_20260604_PR.json"

_DIMS = ("crop", "caption", "footnote", "mask")
_DIM_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1")
_COUNT_FIELDS = (
    ("emitted", "emitted"),
    ("labelled", "labelled"),
    ("unlabelled", "unlabelled"),
    ("unrecognised", "unrecognised"),
    ("icons", "icon"),          # JSON key is "icon"; column is "icons"
)


def _columns() -> list[str]:
    cols = ["variant", "kind"]
    for dim in _DIMS:
        cols += [f"{dim}_{f}" for f in _DIM_FIELDS]
    cols += [f"strict_{f}" for f in _DIM_FIELDS]
    cols += [c for c, _ in _COUNT_FIELDS]
    return cols


def _flatten_record(e: dict) -> dict:
    row: dict = {"variant": e.get("variant"), "kind": e.get("kind")}
    by_dim = e.get("by_dim") or {}
    for dim in _DIMS:
        m = by_dim.get(dim) or {}        # footnote is null on figure rows
        for f in _DIM_FIELDS:
            row[f"{dim}_{f}"] = m.get(f)
    strict = e.get("strict") or {}
    for f in _DIM_FIELDS:
        row[f"strict_{f}"] = strict.get(f)
    counts = e.get("counts") or {}
    for col, key in _COUNT_FIELDS:
        row[col] = counts.get(key)
    return row


def flatten(json_path: Path, csv_path: Path) -> int:
    """Read the nested sweep JSON at ``json_path``; write the flat CSV at
    ``csv_path``. Returns the number of data rows written."""
    records = json.loads(json_path.read_text())
    if not isinstance(records, list):
        raise ValueError(
            f"expected a list of variant records, got {type(records).__name__}: {json_path}"
        )
    rows = [_flatten_record(e) for e in records]
    # Stable order: variant asc, then kind (figures before tables alphabetically).
    rows.sort(key=lambda r: (r["variant"] or "", r["kind"] or ""))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_columns())
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flatten an E01 doc-extraction sweep JSON to CSV.")
    ap.add_argument("--json", type=Path, default=_DEFAULT_JSON,
                    help="nested sweep-results JSON from score_pdf_variants.py "
                         f"(default: {_DEFAULT_JSON.name})")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="output CSV path (default: same stem as --json with .csv)")
    args = ap.parse_args(argv)

    if not args.json.exists():
        print(f"error: json not found: {args.json}", file=sys.stderr)
        return 2
    csv_path = args.csv_out or args.json.with_suffix(".csv")
    n = flatten(args.json, csv_path)
    # Report a repo-relative path when the output is under the repo, else the path as given.
    # `relative_to` needs both sides absolute and nested; a plain relative --csv-out (what a
    # reader naturally passes) is neither, and the bare call raised after the CSV was
    # already written — a crash on success (B-126).
    try:
        shown = csv_path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        shown = csv_path
    print(f"E01 flatten: {n} rows -> {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
