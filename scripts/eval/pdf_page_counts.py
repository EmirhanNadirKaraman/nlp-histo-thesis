#!/usr/bin/env python3
"""
pdf_page_counts.py — per-PDF page count + file size for a directory of PDFs.

Quick eyeball tool for spotting outliers (very long or very large papers
that disproportionately blow up sweep wall-time).  Defaults to
``eval/pdfs/`` — the 30-PDF labelled evaluation set.

Usage::

    python scripts/eval/pdf_page_counts.py
    python scripts/eval/pdf_page_counts.py --pdf-dir files/organized_pdfs
    python scripts/eval/pdf_page_counts.py --threshold 30
    python scripts/eval/pdf_page_counts.py --json-out reports/page_counts.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional


_DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "eval" / "pdfs"
_DEFAULT_LONG_THRESHOLD = 20   # highlight PDFs with > 20 pages
_DEFAULT_HUGE_THRESHOLD = 50   # double-flag PDFs with > 50 pages


def count_pages(pdf_path: Path) -> int:
    """Open a PDF and return its page count.  Caller catches exceptions."""
    import fitz  # type: ignore
    with fitz.open(str(pdf_path)) as doc:
        return len(doc)


def analyze(pdf_dir: Path) -> List[dict]:
    """Return ``[{name, path, pages, bytes, error}]`` for every PDF in
    ``pdf_dir``, sorted by descending page count."""
    rows: List[dict] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        try:
            pages = count_pages(pdf)
            error = None
        except Exception as exc:  # pragma: no cover — exercised on bad PDFs
            pages = None
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "name":  pdf.name,
            "path":  str(pdf),
            "pages": pages,
            "bytes": pdf.stat().st_size,
            "error": error,
        })
    rows.sort(key=lambda r: -(r["pages"] or 0))
    return rows


def render_table(rows: List[dict], *,
                 long_threshold: int = _DEFAULT_LONG_THRESHOLD,
                 huge_threshold: int = _DEFAULT_HUGE_THRESHOLD) -> str:
    """Pretty-printed page-count table with summary stats."""
    out: List[str] = []
    out.append(f"{'file':<55s} {'pages':>6s} {'MB':>8s}  note")
    out.append("-" * 80)
    for r in rows:
        pages = r["pages"]
        mb = r["bytes"] / 1e6
        if r["error"]:
            note = r["error"]
        elif pages is None:
            note = "—"
        elif pages > huge_threshold:
            note = f"⚠⚠ HUGE (> {huge_threshold} pages)"
        elif pages > long_threshold:
            note = f"⚠ long (> {long_threshold} pages)"
        else:
            note = ""
        pages_str = "err" if pages is None else str(pages)
        out.append(f"{r['name']:<55s} {pages_str:>6s} {mb:>8.2f}  {note}")

    valid_pages = [r["pages"] for r in rows if r["pages"] is not None]
    valid_bytes = [r["bytes"] for r in rows if r["pages"] is not None]
    if valid_pages:
        out.append("")
        out.append(f"PDFs:                 {len(valid_pages)}")
        out.append(f"total pages:          {sum(valid_pages)}")
        out.append(f"page count mean:      {statistics.mean(valid_pages):.1f}")
        out.append(f"page count median:    {statistics.median(valid_pages):.0f}")
        out.append(f"page count range:     {min(valid_pages)} – {max(valid_pages)}")
        out.append(f"total disk size:      {sum(valid_bytes)/1e6:.1f} MB")
        out.append(f"size mean / median:   {statistics.mean(valid_bytes)/1e6:.2f} MB  /  "
                   f"{statistics.median(valid_bytes)/1e6:.2f} MB")
        long_pdfs = [r for r in rows if r["pages"] and r["pages"] > long_threshold]
        huge_pdfs = [r for r in rows if r["pages"] and r["pages"] > huge_threshold]
        out.append("")
        out.append(f"PDFs with > {long_threshold} pages:  {len(long_pdfs)}")
        out.append(f"PDFs with > {huge_threshold} pages:  {len(huge_pdfs)} (consider excluding)")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf-dir", type=Path, default=_DEFAULT_PDF_DIR,
                   help="Directory of PDFs to scan.  Default: eval/pdfs/")
    p.add_argument("--threshold", "--long-threshold", type=int,
                   dest="long_threshold", default=_DEFAULT_LONG_THRESHOLD,
                   help="Flag PDFs with > this many pages.")
    p.add_argument("--huge-threshold", type=int, default=_DEFAULT_HUGE_THRESHOLD,
                   help="Double-flag PDFs with > this many pages.")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Also write structured rows to this JSON file.")
    args = p.parse_args(argv)

    if not args.pdf_dir.exists():
        print(f"error: pdf dir not found: {args.pdf_dir}", file=sys.stderr)
        return 2

    rows = analyze(args.pdf_dir)
    if not rows:
        print(f"no PDFs in {args.pdf_dir}", file=sys.stderr)
        return 1

    print(render_table(rows,
                        long_threshold=args.long_threshold,
                        huge_threshold=args.huge_threshold))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"\nwrote JSON → {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
