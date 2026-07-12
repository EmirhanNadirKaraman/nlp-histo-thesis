"""
PyMuPDF4LLM Parser

Extracts hierarchical text from PDFs using PyMuPDF4LLM's markdown conversion.
PyMuPDF4LLM converts PDFs to markdown format optimized for LLMs, which we then
parse to extract hierarchical sections.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict
import logging

# Handle both direct execution and module import
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from legacy.pdf_parsers.base_parser import BasePDFParser, HierarchicalPathBuilder
else:
    from .base_parser import BasePDFParser, HierarchicalPathBuilder


class PyMuPDF4LLMParser(BasePDFParser):
    """
    PDF parser using PyMuPDF4LLM's markdown conversion.

    PyMuPDF4LLM excels at:
    - Converting PDFs to clean markdown
    - Preserving document structure
    - Fast extraction
    - Good handling of tables and formatting
    """

    def __init__(self):
        """Initialize the PyMuPDF4LLM parser."""
        self.logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """
        Check if pymupdf4llm is installed.

        Returns:
            True if available, False otherwise
        """
        try:
            import pymupdf4llm  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_hierarchy(self, pdf_path: str) -> List[Dict]:
        """
        Extract hierarchical text from PDF using PyMuPDF4LLM.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries with hierarchical structure

        Raises:
            ImportError: If pymupdf4llm is not installed
            FileNotFoundError: If PDF doesn't exist
        """
        if not self.is_available():
            raise ImportError(
                "pymupdf4llm is not installed. "
                "Install with: pip install pymupdf4llm"
            )

        pdf_file = self._ensure_pdf_exists(pdf_path)

        try:
            import pymupdf4llm

            self.logger.info(f"Extracting from {pdf_file.name} with PyMuPDF4LLM")

            # Convert PDF to markdown
            md_text = pymupdf4llm.to_markdown(str(pdf_file))

            # Parse markdown to hierarchical structure
            hierarchy = self._parse_markdown_to_hierarchy(md_text)

            self.logger.info(f"Extracted {len(hierarchy)} text elements")

            return hierarchy

        except Exception as e:
            self.logger.error(f"PyMuPDF4LLM extraction failed: {e}")
            raise

    def _parse_markdown_to_hierarchy(self, md_text: str) -> List[Dict]:
        """
        Parse markdown text to hierarchical structure.

        Identifies headers (# ## ###) and paragraphs, building hierarchical paths.
        Skips sections that contain "References" in their header.

        Args:
            md_text: Markdown text from PyMuPDF4LLM

        Returns:
            List of hierarchical text elements
        """
        lines = md_text.split('\n')
        path_builder = HierarchicalPathBuilder()
        hierarchy = []

        current_paragraph = []
        last_header = None  # Track last header to avoid storing title twice
        in_references = False  # Track if we're in a References section
        references_depth = None  # Track depth of References section

        for line in lines:
            line = line.rstrip()

            # Skip empty lines
            if not line:
                # If we have accumulated paragraph text, save it (unless in References)
                if current_paragraph and not in_references:
                    text = ' '.join(current_paragraph).strip()
                    # Skip if accumulated text matches last header (avoid duplication)
                    if text and text != last_header:
                        hierarchy.append(path_builder.get_current_path_dict(text))
                current_paragraph = []
                continue

            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save any accumulated paragraph before processing header
                if current_paragraph and not in_references:
                    text = ' '.join(current_paragraph).strip()
                    # Skip if accumulated text matches last header (avoid duplication)
                    if text and text != last_header:
                        hierarchy.append(path_builder.get_current_path_dict(text))
                current_paragraph = []

                # Process header
                hashes, header_text = header_match.groups()
                depth = len(hashes)

                # Clean header text (remove markdown formatting)
                header_text = self._clean_markdown(header_text)

                # Check if this is a References section or if we're exiting one
                if 'reference' in header_text.lower():
                    in_references = True
                    references_depth = depth
                elif in_references and references_depth is not None and depth <= references_depth:
                    # We've exited the References section (new section at same or shallower depth)
                    in_references = False
                    references_depth = None

                # Update path (even if in references, to maintain structure)
                path_builder.process_header(header_text, depth)

                # Remember this header to avoid storing it as text
                last_header = header_text

            else:
                # Regular paragraph line - skip if in References section
                if not in_references:
                    # Clean markdown formatting
                    clean_line = self._clean_markdown(line)

                    # Skip if this line exactly matches the last header (avoid duplication)
                    # Don't clear last_header - the same title might repeat later
                    if clean_line and clean_line != last_header:
                        current_paragraph.append(clean_line)

        # Save any remaining paragraph (if not in References)
        if current_paragraph and not in_references:
            text = ' '.join(current_paragraph).strip()
            # Skip if accumulated text matches last header (avoid duplication)
            if text and text != last_header:
                hierarchy.append(path_builder.get_current_path_dict(text))

        return hierarchy

    def _clean_markdown(self, text: str) -> str:
        """
        Remove markdown formatting from text.

        Args:
            text: Text with markdown formatting

        Returns:
            Cleaned text
        """
        # Remove bold/italic markers
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic*
        text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__
        text = re.sub(r'_(.+?)_', r'\1', text)        # _italic_

        # Remove inline code markers
        text = re.sub(r'`(.+?)`', r'\1', text)

        # Remove links but keep text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

        return text.strip()


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = "files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf"

    if Path(pdf_file).exists():
        parser = PyMuPDF4LLMParser()

        if not parser.is_available():
            print("ERROR: pymupdf4llm is not installed")
            print("Install with: pip install pymupdf4llm")
            sys.exit(1)

        print(f"\nTesting PyMuPDF4LLM Parser on: {pdf_file}\n")

        try:
            hierarchy = parser.extract_hierarchy(pdf_file)

            print(f"Extracted {len(hierarchy)} elements\n")

            # Show first 10 elements
            print("First 10 elements:")
            print("-" * 80)
            for i, elem in enumerate(hierarchy[:10], 1):
                print(f"\n{i}. Path: {elem['path_string']} (depth {elem['depth']})")
                print(f"   Text: {elem['text'][:150]}...")

            if len(hierarchy) > 10:
                print(f"\n... and {len(hierarchy) - 10} more elements")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"File not found: {pdf_file}")
        print("Usage: python pymupdf4llm_parser.py <pdf_file>")
        sys.exit(1)
