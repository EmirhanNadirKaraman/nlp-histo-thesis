"""
compare_docling_options.py

Runs the full text-extraction pipeline on a set of PDFs under multiple named
DoclingConfig variants and writes per-variant text files plus a summary table
so you can compare the effect of Docling settings on extraction quality.

Usage
-----
    python scripts/compare_docling_options.py [PDF_DIR] [--n N] [--variants v1,v2,...]

    PDF_DIR      directory of PDFs (default: files/organized_pdfs)
    --n N        number of papers to sample (default: 3)
    --variants   comma-separated subset of variant names to run
                 (default: all variants defined in VARIANTS below)

Outputs
-------
    out/docling_compare/<variant_name>/<pmcid>.txt   — extracted paragraphs
    out/docling_compare/layout_cache/               — per-variant Docling caches
    out/docling_compare/summary.txt                 — diff stats across all variants
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

# Bootstrap the repo root onto sys.path so the repository-local imports below resolve
# when this script is run directly (`python scripts/…`) — B-095.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.pdf_text_extraction.components.layout_extractor import DoclingLayoutExtractor
from pipeline.stages.pdf_text_extraction.components.text_assembler import HierarchicalTextAssembler
from pipeline.stages.pdf_text_extraction.config import DoclingConfig, OcrEngine, TextAssemblyConfig

OUTPUT_DIR = Path("out/docling_compare")
CACHE_ROOT = OUTPUT_DIR / "layout_cache"

# ── Variant definitions ───────────────────────────────────────────────────────
# Each entry: (name, DoclingConfig).  Add or remove entries freely.
# The first variant is treated as the baseline for diff comparisons.

_BASE = DoclingConfig()

VARIANTS: Dict[str, DoclingConfig] = {
    # ── Baseline ──────────────────────────────────────────────────────────────
    # OCR disabled — relies entirely on embedded PDF text layer (production default)
    "no_ocr": _BASE,

    # ── OCR engine comparisons ────────────────────────────────────────────────
    # OCR on non-text regions only (native text is used where available)
    "ocr_easyocr": replace(_BASE, do_ocr=True, ocr_engine=OcrEngine.EASYOCR),
    "ocr_tesseract": replace(_BASE, do_ocr=True, ocr_engine=OcrEngine.TESSERACT),

    # Force OCR on every page — ignores embedded text layer entirely
    "ocr_full_page_easyocr": replace(
        _BASE, do_ocr=True, force_full_page_ocr=True, ocr_engine=OcrEngine.EASYOCR
    ),
    "ocr_full_page_tesseract": replace(
        _BASE, do_ocr=True, force_full_page_ocr=True, ocr_engine=OcrEngine.TESSERACT
    ),

    # ── Image resolution comparisons ─────────────────────────────────────────
    # Higher resolution improves OCR quality at the cost of speed
    "scale_1x": replace(_BASE, images_scale=1.0),
    "scale_3x": replace(_BASE, images_scale=3.0),

    # ── Table structure analysis ──────────────────────────────────────────────
    # Disabling skips table-structure model; tables become plain text blocks
    "no_table_structure": replace(_BASE, do_table_structure=False),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_rows(rows, path: Path, pmcid: str, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {pmcid}  [variant={variant}]  {len(rows)} paragraphs\n\n")
        for row in rows:
            stitch = (
                f"  [stitched from {len(row.source_chunks)} chunks]"
                if len(row.source_chunks) > 1 else ""
            )
            f.write(f"[{row.path_string}]{stitch}\n{row.text}\n\n")


def _diff_summary(pmcid: str, results: Dict[str, list], baseline_name: str) -> str:
    baseline = results.get(baseline_name, [])
    baseline_texts = {r.text for r in baseline}

    lines = [
        f"\n{'='*70}",
        f"  {pmcid}",
        f"{'='*70}",
        f"  {'variant':<25}  {'paras':>6}  {'vs baseline':>12}",
        f"  {'-'*25}  {'-'*6}  {'-'*12}",
    ]

    for name, rows in results.items():
        count = len(rows)
        if name == baseline_name:
            delta = "—  (baseline)"
        else:
            texts = {r.text for r in rows}
            extra   = len(texts - baseline_texts)
            missing = len(baseline_texts - texts)
            delta = f"+{extra} / -{missing}"
        lines.append(f"  {name:<25}  {count:>6}  {delta:>12}")

    # Show sample paragraphs rescued / lost relative to baseline
    for name, rows in results.items():
        if name == baseline_name:
            continue
        texts = {r.text for r in rows}
        rescued = texts - baseline_texts
        lost    = baseline_texts - texts
        if rescued:
            lines.append(f"\n  [{name}] paragraphs gained vs {baseline_name} (up to 5):")
            for t in sorted(rescued)[:5]:
                lines.append(f"    + {t[:120]!r}")
        if lost:
            lines.append(f"\n  [{name}] paragraphs lost vs {baseline_name} (up to 5):")
            for t in sorted(lost)[:5]:
                lines.append(f"    - {t[:120]!r}")

    return "\n".join(lines)


def process_pdf(
    pdf_path: Path,
    variant_extractors: Dict[str, DoclingLayoutExtractor],
    assembler_cfg: TextAssemblyConfig,
) -> Dict[str, list]:
    results: Dict[str, list] = {}
    assembler = HierarchicalTextAssembler(config=assembler_cfg)
    for name, extractor in variant_extractors.items():
        try:
            layout = extractor.extract(pdf_path)
            rows = assembler.assemble(layout)
            results[name] = rows
        except Exception as exc:
            print(f"    [{name}] ERROR: {exc}")
            results[name] = []
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Docling extraction options")
    parser.add_argument("pdf_dir", nargs="?", default="files/organized_pdfs")
    parser.add_argument("--n", type=int, default=3, help="Number of PDFs to sample")
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant names (default: all)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs: List[Path] = sorted(
        p for p in pdf_dir.glob("*.pdf") if "-SD" not in p.name and "-s0" not in p.name
    )[: args.n]

    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    selected_names = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else list(VARIANTS.keys())
    )
    unknown = [n for n in selected_names if n not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}.  Available: {list(VARIANTS.keys())}")
        sys.exit(1)

    selected: Dict[str, DoclingConfig] = {n: VARIANTS[n] for n in selected_names}
    # Use "no_ocr" as the baseline if it's among the selected variants,
    # otherwise fall back to the first selected variant.
    baseline_name = "no_ocr" if "no_ocr" in selected_names else selected_names[0]

    print(f"Variants  : {', '.join(selected_names)}")
    print(f"Baseline  : {baseline_name}")
    print(f"PDFs      : {len(pdfs)}")
    print(f"Cache root: {CACHE_ROOT}\n")

    # Build one extractor per variant (each gets its own cache subdir)
    variant_extractors: Dict[str, DoclingLayoutExtractor] = {
        name: DoclingLayoutExtractor(
            config=cfg,
            cache_dir=CACHE_ROOT / name,
        )
        for name, cfg in selected.items()
    }

    assembler_cfg = TextAssemblyConfig()

    summaries: List[str] = []
    for pdf_path in pdfs:
        pmcid = pdf_path.stem.split("_")[0]
        print(f"  Processing {pdf_path.name} …")

        results = process_pdf(pdf_path, variant_extractors, assembler_cfg)

        for name, rows in results.items():
            _write_rows(rows, OUTPUT_DIR / name / f"{pmcid}.txt", pmcid, name)
            print(f"    [{name}] {len(rows)} paragraphs")

        summaries.append(_diff_summary(pmcid, results, baseline_name))

    summary_text = "\n".join(summaries)
    print(summary_text)

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nFull outputs → {OUTPUT_DIR}/")
    print(f"Summary      → {summary_path}")


if __name__ == "__main__":
    main()
