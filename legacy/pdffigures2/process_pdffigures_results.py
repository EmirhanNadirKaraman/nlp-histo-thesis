#!/usr/bin/env python3
"""
Process PDFFigures2 JSON results to create masked PDFs.

Run this AFTER you've run PDFFigures2 to mask tables/figures in PDFs.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
import sys
import logging
import shutil

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def mask_pdf_from_json(pdf_path: Path, json_path: Path, output_dir: Path):
    """
    Mask tables/figures in PDF using PDFFigures2 JSON output.

    Args:
        pdf_path: Original PDF
        json_path: PDFFigures2 JSON with bounding boxes
        output_dir: Where to save masked PDF
    """
    logger.info(f"Processing: {pdf_path.name}")

    # Load JSON
    if not json_path.exists():
        logger.warning(f"  No JSON found: {json_path}")
        return None

    with open(json_path) as f:
        figures_data = json.load(f)

    if not figures_data:
        logger.info("  No figures/tables detected")
        return None

    logger.info(f"  Found {len(figures_data)} figures/tables")

    # Open PDF
    doc = fitz.open(str(pdf_path))
    masked_count = 0

    # Mask each figure/table
    for item in figures_data:
        try:
            page_num = item.get('page', 0)
            region = item.get('regionBoundary', {})

            if not region:
                continue

            bbox = [
                region.get('x1', 0),
                region.get('y1', 0),
                region.get('x2', 0),
                region.get('y2', 0)
            ]

            if page_num < len(doc):
                page = doc[page_num]
                rect = fitz.Rect(bbox)

                # Use redaction to actually REMOVE content (not just cover it)
                page.add_redact_annot(rect, fill=(1, 1, 1))
                masked_count += 1

                fig_name = item.get('name', 'unknown')
                logger.info(f"    ✓ Masked {fig_name} on page {page_num + 1}")

        except Exception as e:
            logger.warning(f"    Failed to mask region: {e}")

    # Apply all redactions (actually removes content from PDF)
    for page_num in range(len(doc)):
        page = doc[page_num]
        page.apply_redactions()

    # Save masked PDF
    masked_pdf = output_dir / f"{pdf_path.stem}_masked.pdf"
    doc.save(str(masked_pdf))
    doc.close()

    logger.info(f"  ✓ Saved masked PDF: {masked_pdf}")
    logger.info(f"  ✓ Masked {masked_count} regions\n")

    return masked_pdf


def extract_text_from_masked(masked_pdf: Path, output_dir: Path):
    """Extract text from masked PDF using Docling."""
    logger.info(f"Extracting text from: {masked_pdf.name}")

    try:
        from legacy.pdf_parsers.docling_parser import DoclingParser

        parser = DoclingParser()
        text_elements = parser.extract(str(masked_pdf))

        # Save JSON
        json_output = output_dir / f"{masked_pdf.stem}_text.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(text_elements, f, indent=2, ensure_ascii=False)

        # Save readable text
        text_output = output_dir / f"{masked_pdf.stem}_text.txt"
        with open(text_output, 'w', encoding='utf-8') as f:
            for elem in text_elements:
                f.write(elem.get('text', '') + '\n\n')

        logger.info(f"  ✓ Extracted {len(text_elements)} text elements")
        logger.info(f"  ✓ Saved: {json_output}")
        logger.info(f"  ✓ Saved: {text_output}\n")

    except Exception as e:
        logger.error(f"  ✗ Text extraction failed: {e}\n")


def main():
    """Process all PDFFigures2 results."""
    import time

    pdf_dir = Path('files/organized_pdfs')
    json_dir = Path('out/data')
    output_base = Path('files/masked_pdfs')
    output_base.mkdir(exist_ok=True)

    # Find all JSON files
    json_files = sorted(json_dir.glob('*.json'))

    if not json_files:
        logger.error("No JSON files found in out/data")
        logger.error("Did you run PDFFigures2 first?")
        return

    logger.info(f"{'='*70}")
    logger.info(f"Processing {len(json_files)} PDFFigures2 results")
    logger.info(f"{'='*70}\n")

    # Track statistics
    stats = {
        'processed': 0,
        'skipped_no_pdf': 0,
        'skipped_already_done': 0,
        'copied_no_figures': 0,
        'total_masked_regions': 0,
        'errors': 0
    }

    start_time = time.time()

    for idx, json_file in enumerate(json_files, 1):
        logger.info(f"\n[{idx}/{len(json_files)}] Processing: {json_file.stem}")

        # Find corresponding PDF
        pdf_name = json_file.stem + '.pdf'
        pdf_path = pdf_dir / pdf_name

        if not pdf_path.exists():
            logger.warning(f"  ✗ PDF not found: {pdf_name}")
            stats['skipped_no_pdf'] += 1
            continue

        # Check if already processed
        masked_pdf_path = output_base / f"{pdf_path.stem}_masked.pdf"
        if masked_pdf_path.exists():
            logger.info("  ✓ Already processed, skipping")
            stats['skipped_already_done'] += 1
            continue

        try:
            # Mask PDF (save directly to output_base)
            masked_pdf = mask_pdf_from_json(pdf_path, json_file, output_base)

            if masked_pdf:
                stats['processed'] += 1
            else:
                # No figures detected - copy original PDF
                output_pdf = output_base / f"{pdf_path.stem}_masked.pdf"
                shutil.copy2(pdf_path, output_pdf)
                logger.info("  ℹ No figures detected - copied original PDF")
                logger.info(f"  ✓ Saved: {output_pdf}\n")
                stats['copied_no_figures'] += 1

        except Exception as e:
            logger.error(f"  ✗ Error processing {json_file.stem}: {e}")
            stats['errors'] += 1

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*70}")
    logger.info("✓ Processing complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Total JSON files: {len(json_files)}")
    logger.info(f"Masked (with tables): {stats['processed']}")
    logger.info(f"Copied (no tables): {stats['copied_no_figures']}")
    logger.info(f"Skipped (already done): {stats['skipped_already_done']}")
    logger.info(f"Skipped (no PDF): {stats['skipped_no_pdf']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info(f"Results saved to: {output_base.resolve()}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
