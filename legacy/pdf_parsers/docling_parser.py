"""
Docling Parser

Extracts hierarchical text from PDFs using IBM's Docling library.
Docling provides state-of-the-art document understanding with excellent
handling of complex layouts, tables, and document structure.
"""

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


class DoclingParser(BasePDFParser):
    """
    PDF parser using IBM's Docling library.

    Docling excels at:
    - Complex document layouts
    - Table extraction and understanding
    - Document structure recognition
    - High-quality text extraction
    - Scientific and technical documents
    """

    def __init__(self):
        """Initialize the Docling parser."""
        self.logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """
        Check if docling is installed.

        Returns:
            True if available, False otherwise
        """
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_hierarchy(self, pdf_path: str) -> List[Dict]:
        """
        Extract hierarchical text from PDF using Docling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries with hierarchical structure

        Raises:
            ImportError: If docling is not installed
            FileNotFoundError: If PDF doesn't exist
        """
        if not self.is_available():
            raise ImportError(
                "docling is not installed. "
                "Install with: pip install docling"
            )

        pdf_file = self._ensure_pdf_exists(pdf_path)

        try:
            from docling.document_converter import DocumentConverter

            self.logger.info(f"Extracting from {pdf_file.name} with Docling")

            # Initialize Docling converter
            converter = DocumentConverter()

            # Convert PDF
            result = converter.convert(str(pdf_file))

            # Parse Docling output to hierarchical structure
            hierarchy = self._parse_docling_to_hierarchy(result)

            self.logger.info(f"Extracted {len(hierarchy)} text elements")

            return hierarchy

        except Exception as e:
            self.logger.error(f"Docling extraction failed: {e}")
            raise

    def _parse_docling_to_hierarchy(self, doc_result) -> List[Dict]:
        """
        Parse Docling document result to hierarchical structure.

        Docling provides a structured document with sections, paragraphs,
        tables, etc. We extract these and build hierarchical paths.

        Args:
            doc_result: Docling document conversion result

        Returns:
            List of hierarchical text elements
        """
        path_builder = HierarchicalPathBuilder()
        hierarchy = []

        try:
            # Get the document
            doc = doc_result.document

            # Export to markdown to get structured content
            md_text = doc.export_to_markdown()

            # Parse markdown (similar to PyMuPDF4LLM)
            hierarchy = self._parse_markdown(md_text, path_builder)

            # Alternative: If Docling provides structured elements directly
            # we could iterate through them instead of markdown
            # This would be more robust but depends on Docling's API

        except Exception as e:
            self.logger.error(f"Error parsing Docling output: {e}")
            # Fallback: try to get plain text
            try:
                text_content = str(doc_result.document)
                if text_content:
                    hierarchy.append({
                        'path_list': [],
                        'path_string': '',
                        'depth': 0,
                        'text': text_content
                    })
            except Exception:
                pass

        return hierarchy

    def _parse_markdown(self, md_text: str, path_builder: HierarchicalPathBuilder) -> List[Dict]:
        """
        Parse markdown text to hierarchical structure.

        Args:
            md_text: Markdown text from Docling
            path_builder: Path builder instance

        Returns:
            List of hierarchical text elements
        """
        import re

        lines = md_text.split('\n')
        hierarchy = []
        current_paragraph = []
        last_header = None  # Track last header to avoid storing title twice

        for line in lines:
            line = line.rstrip()

            # Skip empty lines
            if not line:
                if current_paragraph:
                    text = ' '.join(current_paragraph).strip()
                    # Skip if accumulated text matches last header (avoid duplication)
                    if text and text != last_header:
                        hierarchy.append(path_builder.get_current_path_dict(text))
                    current_paragraph = []
                continue

            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save accumulated paragraph
                if current_paragraph:
                    text = ' '.join(current_paragraph).strip()
                    # Skip if accumulated text matches last header (avoid duplication)
                    if text and text != last_header:
                        hierarchy.append(path_builder.get_current_path_dict(text))
                    current_paragraph = []

                # Process header
                hashes, header_text = header_match.groups()
                depth = len(hashes)

                # Clean header
                header_text = self._clean_markdown(header_text)
                path_builder.process_header(header_text, depth)

                # Remember this header to avoid storing it as text
                last_header = header_text

            else:
                # Regular paragraph
                clean_line = self._clean_markdown(line)

                # Skip if this line exactly matches the last header (avoid duplication)
                # Don't clear last_header - the same title might repeat later
                if clean_line and clean_line != last_header:
                    current_paragraph.append(clean_line)

        # Save remaining paragraph
        if current_paragraph:
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
        import re

        # Remove bold/italic markers
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)

        # Remove inline code
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
        parser = DoclingParser()

        if not parser.is_available():
            print("ERROR: docling is not installed")
            print("Install with: pip install docling")
            sys.exit(1)

        print(f"\nTesting Docling Parser on: {pdf_file}\n")

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
        print("Usage: python docling_parser.py <pdf_file>")
        sys.exit(1)
