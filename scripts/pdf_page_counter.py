#!/usr/bin/env python3
"""
PDF Page Counter for Unprocessed Documents

Counts pages in PDF files that are:
- Not already in the database
- Not in the blacklist
- Located in the organized_pdfs directory

Outputs a sorted list by page count (smallest to largest).

Usage:
    python scripts/pdf_page_counter.py
    python scripts/pdf_page_counter.py --pdf-dir files/organized_pdfs
    python scripts/pdf_page_counter.py --output results.csv
"""

import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Set

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("ERROR: PyMuPDF is required. Install with: pip install PyMuPDF")
    sys.exit(1)

from database import get_db_connection, Document


def load_blacklist(blacklist_file: Path) -> Set[str]:
    """
    Load blacklisted PMCIDs from JSON file.

    Returns:
        Set of blacklisted PMCIDs
    """
    if not blacklist_file.exists():
        return set()

    try:
        with open(blacklist_file, 'r') as f:
            blacklist_data = json.load(f)
        return set(blacklist_data.keys())
    except Exception as e:
        print(f"Warning: Could not load blacklist: {e}")
        return set()


def get_processed_pmcids() -> Set[str]:
    """
    Query database for PMCIDs that have already been processed.

    Returns:
        Set of PMCIDs in the database
    """
    db = get_db_connection()

    with db.session_scope() as session:
        documents = session.query(Document.pmcid).all()
        return {doc.pmcid for doc in documents}


def count_pdf_pages(pdf_path: Path) -> int:
    """
    Count the number of pages in a PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages, or -1 if unable to read
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            return len(doc)
    except Exception as e:
        print(f"Warning: Could not read {pdf_path.name}: {e}")
        return -1


def analyze_unprocessed_pdfs(
    pdf_dir: str = "files/organized_pdfs",
    blacklist_file: str = "out/failed_pdfs_blacklist.json",
    output_file: str = None
) -> List[Dict]:
    """
    Analyze unprocessed PDFs and return sorted list by page count.

    Args:
        pdf_dir: Directory containing PDF files
        blacklist_file: Path to blacklist JSON file
        output_file: Optional CSV output file

    Returns:
        List of dicts with PDF information, sorted by page count
    """
    print("="*80)
    print("PDF Page Counter - Unprocessed Documents")
    print("="*80)
    print(f"PDF Directory: {pdf_dir}")
    print(f"Blacklist File: {blacklist_file}")
    print("="*80 + "\n")

    # Load blacklist
    print("Loading blacklist...")
    blacklist_path = Path(blacklist_file)
    blacklisted_pmcids = load_blacklist(blacklist_path)
    print(f"Found {len(blacklisted_pmcids)} blacklisted PMCIDs\n")

    # Get processed PMCIDs from database
    print("Querying database for processed documents...")
    processed_pmcids = get_processed_pmcids()
    print(f"Found {len(processed_pmcids)} documents in database\n")

    # Collect all PDF files
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        return []

    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}\n")

    # Filter and count pages
    print("Analyzing unprocessed PDFs...")
    print("-"*80)

    unprocessed_pdfs = []
    skipped_processed = 0
    skipped_blacklisted = 0
    skipped_unreadable = 0

    for pdf_file in pdf_files:
        # Extract PMCID from filename
        stem = pdf_file.stem
        if not stem.startswith("PMC"):
            print(f"Skipping {pdf_file.name} - cannot extract PMCID")
            continue

        pmcid = stem.split('_')[0]

        # Check if already processed
        if pmcid in processed_pmcids:
            skipped_processed += 1
            continue

        # Check if blacklisted
        if pmcid in blacklisted_pmcids:
            skipped_blacklisted += 1
            continue

        # Count pages
        page_count = count_pdf_pages(pdf_file)

        if page_count < 0:
            skipped_unreadable += 1
            continue

        unprocessed_pdfs.append({
            'pmcid': pmcid,
            'filename': pdf_file.name,
            'page_count': page_count,
            'file_size_mb': pdf_file.stat().st_size / (1024 * 1024),
            'file_path': str(pdf_file)
        })

    # Sort by page count (smallest first)
    unprocessed_pdfs.sort(key=lambda x: x['page_count'])

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"Total PDF files:           {len(pdf_files)}")
    print(f"Already processed:         {skipped_processed}")
    print(f"Blacklisted:               {skipped_blacklisted}")
    print(f"Unreadable:                {skipped_unreadable}")
    print(f"Unprocessed (remaining):   {len(unprocessed_pdfs)}")
    print("="*80 + "\n")

    if unprocessed_pdfs:
        # Page count distribution
        page_counts = [pdf['page_count'] for pdf in unprocessed_pdfs]
        total_pages = sum(page_counts)
        avg_pages = total_pages / len(unprocessed_pdfs)
        min_pages = min(page_counts)
        max_pages = max(page_counts)

        print("Page Count Distribution")
        print("-"*80)
        print(f"Minimum:      {min_pages:6d} pages")
        print(f"Maximum:      {max_pages:6d} pages")
        print(f"Average:      {avg_pages:6.1f} pages")
        print(f"Total:        {total_pages:6d} pages")
        print("-"*80 + "\n")

        # Print first 20 documents
        print("First 20 Unprocessed Documents (Sorted by Page Count)")
        print("-"*80)
        print(f"{'Rank':<6} {'PMCID':<15} {'Pages':<8} {'Size (MB)':<12} {'Filename'}")
        print("-"*80)

        for i, pdf in enumerate(unprocessed_pdfs[:20], 1):
            print(f"{i:<6} {pdf['pmcid']:<15} {pdf['page_count']:<8} {pdf['file_size_mb']:<12.2f} {pdf['filename']}")

        if len(unprocessed_pdfs) > 20:
            print(f"\n... and {len(unprocessed_pdfs) - 20} more documents")

        print("-"*80 + "\n")

        # Last 10 documents (largest)
        if len(unprocessed_pdfs) > 10:
            print("Last 10 Unprocessed Documents (Largest)")
            print("-"*80)
            print(f"{'Rank':<6} {'PMCID':<15} {'Pages':<8} {'Size (MB)':<12} {'Filename'}")
            print("-"*80)

            for i, pdf in enumerate(unprocessed_pdfs[-10:], len(unprocessed_pdfs) - 9):
                print(f"{i:<6} {pdf['pmcid']:<15} {pdf['page_count']:<8} {pdf['file_size_mb']:<12.2f} {pdf['filename']}")

            print("-"*80 + "\n")

        # Save to CSV if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['rank', 'pmcid', 'page_count', 'file_size_mb', 'filename', 'file_path'])
                writer.writeheader()

                for i, pdf in enumerate(unprocessed_pdfs, 1):
                    writer.writerow({
                        'rank': i,
                        'pmcid': pdf['pmcid'],
                        'page_count': pdf['page_count'],
                        'file_size_mb': round(pdf['file_size_mb'], 2),
                        'filename': pdf['filename'],
                        'file_path': pdf['file_path']
                    })

            print(f"✓ Saved detailed results to: {output_path}")
            print(f"  ({len(unprocessed_pdfs)} documents)\n")

    return unprocessed_pdfs


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Count and order unprocessed PDF files by page count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage (files/organized_pdfs/)
  python scripts/pdf_page_counter.py

  # Custom PDF directory
  python scripts/pdf_page_counter.py --pdf-dir path/to/pdfs

  # Save results to CSV
  python scripts/pdf_page_counter.py --output results/unprocessed_pdfs.csv

  # Custom blacklist location
  python scripts/pdf_page_counter.py --blacklist out/my_blacklist.json
        """
    )

    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='files/organized_pdfs',
        help='Directory containing PDF files (default: files/organized_pdfs)'
    )
    parser.add_argument(
        '--blacklist',
        type=str,
        default='out/failed_pdfs_blacklist.json',
        help='Path to blacklist JSON file (default: out/failed_pdfs_blacklist.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Optional CSV output file for detailed results'
    )

    args = parser.parse_args()

    try:
        analyze_unprocessed_pdfs(
            pdf_dir=args.pdf_dir,
            blacklist_file=args.blacklist,
            output_file=args.output
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
