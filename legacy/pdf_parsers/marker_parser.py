"""
Marker PDF Parser

Primary parser using Marker for high-quality PDF→Markdown conversion.
Converts PDF to markdown format, then parses headers to build hierarchical structure.
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Handle both direct execution and module import
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from legacy.pdf_parsers.base_parser import BasePDFParser, HierarchicalPathBuilder
else:
    from .base_parser import BasePDFParser, HierarchicalPathBuilder


class MarkerParser(BasePDFParser):
    """
    PDF parser using Marker for high-quality text extraction.

    Marker converts PDFs to markdown with preserved structure using headers.
    This parser then extracts hierarchical sections from the markdown output.

    Strategy:
        1. Convert PDF → Markdown using Marker
        2. Parse markdown line-by-line
        3. Detect headers (# ## ###) to build hierarchy
        4. Accumulate text between headers
        5. Return standardized hierarchical format
    """

    def __init__(self, max_pages: Optional[int] = None, langs: Optional[List[str]] = None):
        """
        Initialize Marker parser.

        Args:
            max_pages: Maximum number of pages to process (None = all)
            langs: List of languages (default: ["English"])
        """
        self.max_pages = max_pages
        self.langs = langs or ["English"]
        self.logger = logging.getLogger(__name__)

        # Cache for Marker models (loaded lazily)
        self._models = None

    def is_available(self) -> bool:
        """Check if marker-pdf is installed."""
        try:
            from marker.converters.pdf import PdfConverter  # noqa: F401
            from marker.models import create_model_dict  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_models(self):
        """Lazy load Marker models (heavy operation, do once)."""
        if self._models is None:
            try:
                from marker.models import create_model_dict
                self.logger.info("Loading Marker models...")
                self._models = create_model_dict()
                self.logger.info("Marker models loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Marker models: {e}")
                raise

    def extract_hierarchy(self, pdf_path: str) -> List[Dict]:
        """
        Extract hierarchical text structure using Marker.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of hierarchical text elements
        """
        # Validate PDF exists
        pdf_file = self._ensure_pdf_exists(pdf_path)

        # Convert PDF to markdown
        markdown_text = self._convert_to_markdown(str(pdf_file))

        # Parse markdown to hierarchical structure
        data = self._parse_markdown_structure(markdown_text)

        # Validate output
        if not self.validate_output(data):
            self.logger.warning(f"Output validation failed for {pdf_path}")

        return data

    def _convert_to_markdown(self, pdf_path: str) -> str:
        """
        Convert PDF to markdown using Marker.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Markdown text
        """
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered

        # Load models (cached after first call)
        self._load_models()

        try:
            self.logger.info(f"Converting PDF to markdown: {Path(pdf_path).name}")

            # Create converter with model artifacts
            converter = PdfConverter(artifact_dict=self._models)

            # Convert PDF using Marker (new API)
            rendered = converter(pdf_path)

            # Extract markdown text from rendered output
            markdown_text, _, _ = text_from_rendered(rendered)

            self.logger.info(f"Conversion complete. Text length: {len(markdown_text)} chars")
            return markdown_text

        except Exception as e:
            self.logger.error(f"Marker conversion failed: {e}")
            raise

    def _parse_markdown_structure(self, markdown: str) -> List[Dict]:
        """
        Parse markdown text to extract hierarchical structure.

        Strategy:
            - Detect headers using regex: ^#{1,6} Text
            - Build hierarchical paths using HierarchicalPathBuilder
            - Accumulate text content between headers
            - Return standardized format

        Args:
            markdown: Markdown text from Marker

        Returns:
            List of hierarchical text elements
        """
        data = []
        lines = markdown.split('\n')
        path_builder = HierarchicalPathBuilder()

        current_text = []

        for line in lines:
            # Check if line is a markdown header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save accumulated text at previous path
                if current_text and path_builder.current_path:
                    text_content = '\n'.join(current_text).strip()
                    if text_content:
                        data.append(path_builder.get_current_path_dict(text_content))
                    current_text = []

                # Process new header
                header_level = len(header_match.group(1))  # Count #'s
                header_text = header_match.group(2).strip()

                # Clean up header text (remove markdown formatting)
                header_text = self._clean_header_text(header_text)

                # Update path
                path_builder.process_header(header_text, header_level)

            else:
                # Accumulate text content (non-header lines)
                if line.strip():
                    current_text.append(line)

        # Save final text block
        if current_text and path_builder.current_path:
            text_content = '\n'.join(current_text).strip()
            if text_content:
                data.append(path_builder.get_current_path_dict(text_content))

        self.logger.info(f"Extracted {len(data)} hierarchical elements")
        return data

    def _clean_header_text(self, header_text: str) -> str:
        """
        Clean header text by removing markdown formatting.

        Args:
            header_text: Raw header text

        Returns:
            Cleaned header text
        """
        # Remove bold/italic markers
        header_text = re.sub(r'\*\*(.+?)\*\*', r'\1', header_text)  # **bold**
        header_text = re.sub(r'\*(.+?)\*', r'\1', header_text)      # *italic*
        header_text = re.sub(r'_(.+?)_', r'\1', header_text)        # _italic_

        # Remove inline code markers
        header_text = re.sub(r'`(.+?)`', r'\1', header_text)

        return header_text.strip()


# Example usage for testing
if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        # Default test file
        pdf_file = "files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf"

    if Path(pdf_file).exists():
        print(f"\n{'='*60}")
        print(f"Testing MarkerParser on: {pdf_file}")
        print(f"{'='*60}\n")

        parser = MarkerParser()

        if not parser.is_available():
            print("ERROR: Marker is not installed.")
            print("Install with: pip install marker-pdf")
            sys.exit(1)

        try:
            # Extract hierarchy
            data = parser.extract_hierarchy(pdf_file)

            print(f"\nExtracted {len(data)} hierarchical elements\n")

            # Display first 5 elements
            for i, elem in enumerate(data[:5], 1):
                print(f"Element {i}:")
                print(f"  Path: {elem['path_string']}")
                print(f"  Depth: {elem['depth']}")
                print(f"  Text preview: {elem['text'][:150]}...")
                print()

            if len(data) > 5:
                print(f"... and {len(data) - 5} more elements\n")

            # Show section structure
            print("\nSection Structure:")
            print("-" * 60)
            unique_paths = sorted(set(elem['path_string'] for elem in data))
            for path in unique_paths:
                depth = path.count(' > ') + 1 if path else 0
                indent = "  " * (depth - 1)
                section = path.split(' > ')[-1] if path else "Root"
                print(f"{indent}{section}")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"File not found: {pdf_file}")
        print("Usage: python marker_parser.py <pdf_file>")
        sys.exit(1)
