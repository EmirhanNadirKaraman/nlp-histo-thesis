"""
compare_prefilter.py

Runs HierarchicalTextAssembler on a set of PDFs with pre_filter_relevance=True
(original) and pre_filter_relevance=False (new) and prints a diff summary so
you can decide which mode is better for rule extraction.

Usage:
    python scripts/compare_prefilter.py [PDF_DIR] [--n N]

    PDF_DIR  directory of PDFs (default: files/organized_pdfs)
    --n N    number of papers to sample (default: 5)

Outputs are written to:
    out/prefilter_compare/{pmcid}_filtered.txt     (pre_filter=True)
    out/prefilter_compare/{pmcid}_unfiltered.txt   (pre_filter=False)
    out/prefilter_compare/summary.txt              (diff stats for all papers)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so the repository-local imports below resolve
# when this script is run directly (`python scripts/…`) — B-095.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nlp_histo.pipeline.stages.pdf_text_extraction.components.layout_extractor import DoclingLayoutExtractor
from nlp_histo.pipeline.stages.pdf_text_extraction.components.text_assembler import HierarchicalTextAssembler
from nlp_histo.pipeline.stages.pdf_text_extraction.config import TextAssemblyConfig

CACHE_DIR  = Path("out/prefilter_compare/layout_cache")
OUTPUT_DIR = Path("out/prefilter_compare")

FILTERED_SUFFIX   = "_filtered.txt"
UNFILTERED_SUFFIX = "_unfiltered.txt"


def _write_rows(rows, path: Path, pmcid: str, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {pmcid}  [{mode}]  {len(rows)} paragraphs\n\n")
        for row in rows:
            stitch_note = f"  [stitched from {len(row.source_chunks)} chunks]" if len(row.source_chunks) > 1 else ""
            f.write(f"[{row.path_string}]{stitch_note}\n{row.text}\n\n")


def _diff_summary(pmcid: str, filtered, unfiltered) -> str:
    f_texts = {r.text for r in filtered}
    u_texts = {r.text for r in unfiltered}
    only_in_filtered   = f_texts - u_texts   # dropped by un-filtered (shouldn't happen)
    only_in_unfiltered = u_texts - f_texts   # paragraphs rescued by removing pre-filter

    lines = [
        f"\n{'='*60}",
        f"  {pmcid}",
        f"{'='*60}",
        f"  filtered   (pre_filter=True) : {len(filtered):>4} paragraphs",
        f"  unfiltered (pre_filter=False): {len(unfiltered):>4} paragraphs",
        f"  extra in unfiltered          : {len(only_in_unfiltered):>4}",
        f"  missing from unfiltered      : {len(only_in_filtered):>4}",
    ]
    if only_in_unfiltered:
        lines.append("\n  --- Paragraphs rescued when pre_filter=False (sample up to 10) ---")
        for t in sorted(only_in_unfiltered)[:10]:
            lines.append(f"    • {t[:120]!r}")
    if only_in_filtered:
        lines.append("\n  --- Paragraphs lost in unfiltered mode (shouldn't happen) ---")
        for t in sorted(only_in_filtered)[:10]:
            lines.append(f"    • {t[:120]!r}")
    return "\n".join(lines)


def process_pdf(pdf_path: Path, extractor: DoclingLayoutExtractor) -> dict:
    pmcid = pdf_path.stem.split("_")[0]

    layout = extractor.extract(pdf_path)

    assembler_filtered = HierarchicalTextAssembler(
        config=TextAssemblyConfig(pre_filter_relevance=True)
    )
    assembler_unfiltered = HierarchicalTextAssembler(
        config=TextAssemblyConfig(pre_filter_relevance=False)
    )

    filtered   = assembler_filtered.assemble(layout)
    unfiltered = assembler_unfiltered.assemble(layout)

    _write_rows(filtered,   OUTPUT_DIR / f"{pmcid}{FILTERED_SUFFIX}",   pmcid, "pre_filter=True")
    _write_rows(unfiltered, OUTPUT_DIR / f"{pmcid}{UNFILTERED_SUFFIX}", pmcid, "pre_filter=False")

    return {"pmcid": pmcid, "filtered": filtered, "unfiltered": unfiltered}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre_filter_relevance modes")
    parser.add_argument("pdf_dir", nargs="?", default="files/organized_pdfs")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(
        p for p in pdf_dir.glob("*.pdf") if "-SD" not in p.name and "-s0" not in p.name
    )[: args.n]

    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    print(f"Processing {len(pdfs)} PDFs …")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extractor = DoclingLayoutExtractor(cache_dir=CACHE_DIR)

    summaries = []
    for pdf in pdfs:
        print(f"  {pdf.name}")
        result = process_pdf(pdf, extractor)
        summaries.append(_diff_summary(result["pmcid"], result["filtered"], result["unfiltered"]))

    summary_text = "\n".join(summaries)
    print(summary_text)

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(summary_text)
    print(f"\nFull outputs → {OUTPUT_DIR}/")
    print(f"Summary      → {summary_path}")


if __name__ == "__main__":
    main()
