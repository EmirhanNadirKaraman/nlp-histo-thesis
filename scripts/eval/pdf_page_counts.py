#!/usr/bin/env python3
"""
pdf_page_counts.py — per-PDF page count + file size for a directory of PDFs,
plus a corpus page-count distribution, a page-cap trade-off table, and an
optional LaTeX export.

Quick eyeball tool for spotting outliers (very long or very large papers that
disproportionately blow up extraction wall-time), and for justifying the
extraction page cap (``runner.py --max-pages``).  Defaults to ``eval/pdfs/`` —
the 30-PDF labelled evaluation set.

Usage::

    python scripts/eval/pdf_page_counts.py
    python scripts/eval/pdf_page_counts.py --pdf-dir files/organized_pdfs
    python scripts/eval/pdf_page_counts.py --threshold 30

    # Corpus cap analysis (single main PDF per paper) + LaTeX table:
    python scripts/eval/pdf_page_counts.py --pdf-dir files/organized_pdfs \\
        --single-pdf-only --summary --latex-out reports/page_caps.tex
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional


_DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "eval" / "pdfs"
_DEFAULT_LONG_THRESHOLD = 20   # highlight PDFs with > 20 pages
_DEFAULT_HUGE_THRESHOLD = 50   # double-flag PDFs with > 50 pages
_DEFAULT_CAPS = "15,20,25,30,40,50"


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


def single_pdf_only(rows: List[dict]) -> List[dict]:
    """Keep one PDF per PMC id, dropping any package that shipped >1 PDF
    (supplemented).  Mirrors the extraction runner's ``--main-pdf-only`` default
    so the analysis reflects the actual corpus candidate pool."""
    by: dict = defaultdict(list)
    for r in rows:
        pmcid = Path(r["name"]).stem.split("_", 1)[0]
        by[pmcid].append(r)
    return [grp[0] for grp in by.values() if len(grp) == 1]


def _pct(sorted_vals: List[int], q: float) -> int:
    n = len(sorted_vals)
    return sorted_vals[min(n - 1, int(round(q * (n - 1))))]


def render_table(rows: List[dict], *,
                 long_threshold: int = _DEFAULT_LONG_THRESHOLD,
                 huge_threshold: int = _DEFAULT_HUGE_THRESHOLD) -> str:
    """Pretty-printed per-PDF page-count table with summary stats."""
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


def render_distribution(pages: List[int]) -> str:
    """Percentile summary of a page-count list."""
    s = sorted(pages)
    n = len(s)
    return (f"Page-count distribution (n={n}, total={sum(s)}, mean={sum(s)/n:.1f}):\n"
            f"  min={s[0]}  p25={_pct(s,.25)}  median={_pct(s,.5)}  p75={_pct(s,.75)}  "
            f"p90={_pct(s,.9)}  p95={_pct(s,.95)}  p99={_pct(s,.99)}  max={s[-1]}")


def cap_rows(pages: List[int], caps: List[int]) -> List[dict]:
    """For each cap T, how many papers / how much page-volume survive ``<= T``,
    plus a final 'no cap' row."""
    total = sum(pages)
    n = len(pages)
    rows = []
    for T in caps:
        kept = [p for p in pages if p <= T]
        rows.append({
            "cap": T,
            "papers_kept": len(kept),
            "pct_papers": 100 * len(kept) / n,
            "pages_kept": sum(kept),
            "pct_pages": 100 * sum(kept) / total,
        })
    rows.append({"cap": None, "papers_kept": n, "pct_papers": 100.0,
                 "pages_kept": total, "pct_pages": 100.0})
    return rows


def render_cap_text(rows: List[dict]) -> str:
    out = ["Page-cap trade-off (keep papers with <= T pages;",
           " page-volume ~= extraction cost):",
           f"  {'cap':>5} {'papers':>8} {'%pap':>6} {'pages':>9} {'%pages':>7}"]
    for r in rows:
        cap = "none" if r["cap"] is None else str(r["cap"])
        out.append(f"  {cap:>5} {r['papers_kept']:>8} {r['pct_papers']:>5.1f}% "
                   f"{r['pages_kept']:>9} {r['pct_pages']:>6.1f}%")
    return "\n".join(out)


def render_cap_latex(rows: List[dict], n: int, total: int, *, highlight: int = 30) -> str:
    """Render the cap trade-off as a booktabs LaTeX table."""
    def num(x: int) -> str:
        return f"{x:,}".replace(",", r"\,")   # thin-space digit grouping

    body: List[str] = []
    for r in rows:
        cap = "none" if r["cap"] is None else rf"$\le {r['cap']}$"
        cells = [cap, num(r["papers_kept"]), f"{r['pct_papers']:.1f}",
                 num(r["pages_kept"]), f"{r['pct_pages']:.1f}"]
        if r["cap"] == highlight:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        body.append("    " + " & ".join(cells) + r" \\")

    return "\n".join([
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Page-count cap trade-off over the single-PDF candidate pool "
        rf"($n={num(n)}$ papers, {num(total)} pages total). Page-volume is a proxy "
        rf"for document-extraction cost.}}",
        r"  \label{tab:page-cap-tradeoff}",
        r"  \begin{tabular}{@{}r r r r r@{}}",
        r"    \toprule",
        r"    Cap & Papers & \% papers & Page-volume & \% volume \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


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
    p.add_argument("--single-pdf-only", action="store_true",
                   help="Keep one PDF per PMC id, dropping multi-PDF (supplemented) "
                        "packages — matches the extraction runner's corpus definition.")
    p.add_argument("--caps", type=str, default=_DEFAULT_CAPS,
                   help=f"Comma-separated page caps for the trade-off table "
                        f"(default: {_DEFAULT_CAPS}).")
    p.add_argument("--highlight", type=int, default=30,
                   help="Bold this cap's row in the LaTeX table (default: 30).")
    p.add_argument("--latex-out", type=Path, default=None,
                   help="Write the cap trade-off as a booktabs LaTeX table to this path.")
    p.add_argument("--summary", action="store_true",
                   help="Skip the per-PDF listing; show distribution + cap table only.")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Also write structured per-PDF rows to this JSON file.")
    args = p.parse_args(argv)

    if not args.pdf_dir.exists():
        print(f"error: pdf dir not found: {args.pdf_dir}", file=sys.stderr)
        return 2

    rows = analyze(args.pdf_dir)
    if not rows:
        print(f"no PDFs in {args.pdf_dir}", file=sys.stderr)
        return 1

    if args.single_pdf_only:
        before = len(rows)
        rows = single_pdf_only(rows)
        print(f"# single-PDF filter: {len(rows)} papers kept "
              f"(dropped {before - len(rows)} files in multi-PDF packages)\n",
              file=sys.stderr)

    if not args.summary:
        print(render_table(rows,
                            long_threshold=args.long_threshold,
                            huge_threshold=args.huge_threshold))

    pages = sorted(r["pages"] for r in rows if r["pages"] is not None)
    if pages:
        caps = [int(x) for x in args.caps.split(",") if x.strip()]
        crows = cap_rows(pages, caps)
        print()
        print(render_distribution(pages))
        print()
        print(render_cap_text(crows))
        if args.latex_out:
            tex = render_cap_latex(crows, len(pages), sum(pages), highlight=args.highlight)
            args.latex_out.parent.mkdir(parents=True, exist_ok=True)
            args.latex_out.write_text(tex)
            print(f"\nwrote LaTeX → {args.latex_out}", file=sys.stderr)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"\nwrote JSON → {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
