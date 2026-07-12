"""
two_pass_extract.py

Demonstration and CLI entry point for the TwoPassTextExtractor pipeline.

Usage
─────
    python scripts/two_pass_extract.py path/to/paper.pdf [--out-dir out/twopass]
    python scripts/two_pass_extract.py files/organized_pdfs/ --batch --n 5

Outputs per PDF
───────────────
    out/twopass/<pmcid>_text.txt          — extracted body text, hierarchical
    out/twopass/<pmcid>_rejected.txt      — Pass-1 nodes rejected as ghost text
    out/twopass/masked_pdfs/              — header-masked PDFs (intermediate)
    out/twopass/layout_cache/             — Docling JSON caches (optional)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Bootstrap the repo root onto sys.path so the repository-local imports below resolve
# when this script is run directly (`python scripts/…`) — B-095.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.pdf_text_extraction.components.two_pass_extractor import (
    TwoPassTextExtractor,
)
from pipeline.stages.pdf_text_extraction.config import (
    DoclingConfig,
    TwoPassConfig,
)
from pipeline.stages.pdf_text_extraction.models.scored_node import TwoPassResult
from parsers.layout_utils import extract_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("two_pass_extract")


# ── Config helpers ────────────────────────────────────────────────────────────

def make_two_pass_config() -> TwoPassConfig:
    """
    Return a TwoPassConfig tuned for typical histopathology PDFs.

    Adjust thresholds here or construct TwoPassConfig() directly with your own
    values.  All thresholds are documented on the dataclass.
    """
    return TwoPassConfig(
        # Pixel rendering
        render_dpi=150,

        # Blank-space detection (Rule R1) ─────────────────────────────────────
        # A region with mean brightness ≥ 245 AND < 2% dark pixels is blank.
        blank_brightness_threshold=245.0,
        blank_dark_pixel_max_fraction=0.02,

        # Word-count fallback (Rule R2, only when rendering unavailable) ───────
        # Trigger when fitz char coverage < 5% for text with ≥ 8 chars.
        min_char_coverage_threshold=0.05,
        min_text_chars_for_word_check=8,

        # Dense-text heuristic (Rule R3) ─────────────────────────────────────
        max_chars_per_bbox_pt=15.0,

        # Header-zone hint ────────────────────────────────────────────────────
        max_top_fraction_header=0.15,  # top 15% of page = "header zone"

        # Body-anchor finding ─────────────────────────────────────────────────
        min_anchor_word_count=5,

        # Header-mask geometry ────────────────────────────────────────────────
        header_mask_margin_pt=3.0,
    )


def make_docling_config() -> DoclingConfig:
    return DoclingConfig(
        do_table_structure=True,
        do_ocr=False,
        images_scale=2.0,
        accelerator_device="cpu",
    )


# ── Processing ────────────────────────────────────────────────────────────────

def process_one(
    pdf_path: Path,
    extractor: TwoPassTextExtractor,
    out_dir: Path,
) -> None:
    """Run the pipeline on one PDF and write text + rejection reports."""
    pmcid = pdf_path.stem
    logger.info("Processing %s", pmcid)

    try:
        result: TwoPassResult = extractor.process(pdf_path)
    except Exception as exc:
        logger.error("FAILED %s: %s", pmcid, exc, exc_info=True)
        return

    # ── Write body text ───────────────────────────────────────────────────────
    # Convert Pass-2 text elements to the dict format expected by extract_text()
    element_dicts = [el.to_dict() for el in result.text_elements]
    rows, n_skipped = extract_text(element_dicts, nlp=None, pre_filter=True)

    text_path = out_dir / f"{pmcid}_text.txt"
    with open(text_path, "w", encoding="utf-8") as fh:
        fh.write(f"Document: {pmcid}  |  two-pass extraction  |  document order\n")
        fh.write("=" * 80 + "\n\n")
        current_path = None
        for path_str, _, _, para in rows:
            if path_str != current_path:
                fh.write(f"[{path_str}]\n" + "-" * 80 + "\n")
                current_path = path_str
            fh.write(para + "\n\n")

    logger.info(
        "  %s → %d paragraphs written (%d filtered), %d figure(s), %d table(s)",
        pmcid, len(rows), n_skipped,
        len(result.figure_elements), len(result.table_elements),
    )

    # ── Write rejection report ────────────────────────────────────────────────
    if result.rejected_nodes:
        rej_path = out_dir / f"{pmcid}_rejected.txt"
        with open(rej_path, "w", encoding="utf-8") as fh:
            fh.write(f"Rejected Pass-1 nodes for {pmcid}\n{'=' * 60}\n\n")
            for node in result.rejected_nodes:
                el = node.element
                ev = node.evidence
                fh.write(
                    f"page={el.page}  type={el.type}\n"
                    f"  reason:     {node.rejection_reason}\n"
                    f"  text:       {(el.text or '')[:120]!r}\n"
                    f"  brightness: mean={ev.pixel_brightness_mean:.1f}  "
                    f"dark_frac={ev.dark_pixel_fraction:.4f}  "
                    f"blank={ev.visually_blank}\n"
                    f"  fitz words: {ev.fitz_word_count}  "
                    f"char_coverage={ev.char_coverage:.3f}\n"
                    f"  render_skipped={ev.render_skipped}\n\n"
                )
        logger.info("  Rejection report → %s", rej_path.name)

    # ── Anchor summary ────────────────────────────────────────────────────────
    if result.header_anchors:
        for page_no, anchor in sorted(result.header_anchors.items()):
            logger.debug(
                "  Anchor page=%d  fitz_y0=%.1f  text=%.60r",
                page_no, anchor.top_y_fitz,
                (anchor.element.text or "")[:60],
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def collect_pdfs(target: Path, n: int) -> List[Path]:
    if target.is_file():
        return [target]
    pdfs = sorted(target.glob("*.pdf"))
    if n > 0:
        pdfs = pdfs[:n]
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the two-pass PDF text-cleanup pipeline."
    )
    parser.add_argument("target", type=Path, help="PDF file or directory of PDFs")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out/twopass"),
        help="Output directory (default: out/twopass)",
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Number of PDFs to process when target is a directory (0 = all)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable Docling layout JSON cache",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Set log level to DEBUG",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir    = None if args.no_cache else args.out_dir / "layout_cache"
    masked_dir   = args.out_dir / "masked_pdfs"

    extractor = TwoPassTextExtractor(
        config=make_two_pass_config(),
        docling_config=make_docling_config(),
        cache_dir=cache_dir,
        masked_pdf_dir=masked_dir,
    )

    pdfs = collect_pdfs(args.target, args.n)
    if not pdfs:
        logger.error("No PDFs found at %s", args.target)
        sys.exit(1)

    logger.info("Processing %d PDF(s) → %s", len(pdfs), args.out_dir)
    for pdf in pdfs:
        process_one(pdf, extractor, args.out_dir)

    logger.info("Done.")


# ── Inline usage example (import API) ────────────────────────────────────────

def inline_example() -> None:
    """
    Minimal programmatic usage — copy-paste this into a notebook or script.

        from scripts.two_pass_extract import inline_example
        inline_example()
    """

    pdf = Path("files/organized_pdfs/PMC1234567.pdf")
    if not pdf.exists():
        print("Set `pdf` to an existing PDF path before running.")
        return

    # ── 1. Build config ───────────────────────────────────────────────────────
    two_pass_cfg = TwoPassConfig(
        render_dpi=96,
        blank_brightness_threshold=245.0,
        blank_dark_pixel_max_fraction=0.02,
        min_anchor_word_count=5,
        header_mask_margin_pt=3.0,
    )
    docling_cfg = DoclingConfig(do_ocr=False)

    # ── 2. Instantiate and run ─────────────────────────────────────────────────
    extractor = TwoPassTextExtractor(
        config=two_pass_cfg,
        docling_config=docling_cfg,
        masked_pdf_dir=Path("out/twopass/masked"),
    )
    result = extractor.process(pdf)

    # ── 3. Inspect results ─────────────────────────────────────────────────────
    print(f"Pass-1: {len(result.pass1_layout)} elements, "
          f"{result.n_rejected} rejected as ghost text")

    print(f"\nBody anchors found on pages: {sorted(result.header_anchors)}")

    print("\nPass-2 (masked re-extraction):")
    print(f"  text elements  : {len(result.text_elements)}")
    print(f"  figure elements: {len(result.figure_elements)}")
    print(f"  table elements : {len(result.table_elements)}")

    # ── 4. Convert to hierarchical rows via existing extract_text() ───────────
    element_dicts = [el.to_dict() for el in result.text_elements]
    rows, n_skipped = extract_text(element_dicts, nlp=None, pre_filter=True)

    print(f"\nHierarchical rows: {len(rows)} paragraphs "
          f"({n_skipped} skipped by pre-filter)")
    for path_str, path_list, depth, para in rows[:3]:
        print(f"\n  [{path_str}]\n  {para[:120]}")

    # ── 5. Inspect rejected ghost-text nodes ─────────────────────────────────
    print(f"\nRejected ghost-text nodes ({result.n_rejected}):")
    for node in result.rejected_nodes[:5]:
        el, ev = node.element, node.evidence
        print(
            f"  page={el.page}  {el.type:20s}  "
            f"brightness={ev.pixel_brightness_mean:.0f}  "
            f"dark_frac={ev.dark_pixel_fraction:.3f}  "
            f"text={repr((el.text or '')[:60])}"
        )


if __name__ == "__main__":
    main()
