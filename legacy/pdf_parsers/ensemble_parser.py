"""
Medical-Grade Ensemble PDF Parser

Intelligent page-level routing optimized for histopathology papers.
Uses fast detection to route each page to the optimal parser.

Architecture:
- Lead (PyMuPDF4LLM): Narrative multi-column text (90% of pages)
- Specialist (Docling): Pages with tables (clean markdown tables)
- Archivist (Marker): Pages with images/figures (high-res extraction)

M1-Optimized: Lazy loads parsers only when needed.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Handle both direct execution and module import
if __name__ == "__main__" or not __package__:
    # Add parent directories to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from legacy.pdf_parsers.marker_parser import MarkerParser
    from legacy.pdf_parsers.nougat_parser import NougatParser
    from legacy.pdf_parsers.pdffigures_parser import PDFFiguresParser
    from legacy.pdf_parsers.docling_parser import DoclingParser
    from legacy.pdf_parsers.pymupdf4llm_parser import PyMuPDF4LLMParser
    from legacy.pdf_parsers.deduplicator import MedicalPaperDeduplicator
else:
    # Use relative imports when imported as module
    from .marker_parser import MarkerParser
    from .nougat_parser import NougatParser
    from .pdffigures_parser import PDFFiguresParser
    from .docling_parser import DoclingParser
    from .pymupdf4llm_parser import PyMuPDF4LLMParser
    from .deduplicator import MedicalPaperDeduplicator


class EnsemblePDFParser:
    """
    Medical-grade ensemble parser with intelligent page-level routing.

    Two Modes:
    1. ROUTING MODE (default, recommended for medical papers):
       - Fast PyMuPDF detection per page
       - Routes to specialist parser based on content
       - Hybrid output from multiple parsers
       - M1-optimized (lazy loading)

    2. FALLBACK MODE (legacy):
       - Tries entire document with each parser sequentially
       - Uses first successful parser
       - Single parser processes whole document
    """

    def __init__(
        self,
        mode: str = "routing",
        min_content_threshold: int = 3,
        extract_figures: bool = True,
        parser_priority: Optional[List[str]] = None,
        deduplicate: bool = True
    ):
        """
        Initialize ensemble parser.

        Args:
            mode: "routing" (page-level) or "fallback" (legacy sequential)
            min_content_threshold: Minimum elements for fallback mode success
            extract_figures: Whether to extract figures
            parser_priority: Custom parser order for fallback mode
            deduplicate: Whether to remove duplicate headers/footers (default: True)
        """
        self.mode = mode
        self.min_content_threshold = min_content_threshold
        self.extract_figures_enabled = extract_figures
        self.parser_priority = parser_priority
        self.deduplicate_enabled = deduplicate

        self.logger = logging.getLogger(__name__)

        # Initialize deduplicator if enabled
        if self.deduplicate_enabled:
            self.deduplicator = MedicalPaperDeduplicator()

        # Lazy loading: Initialize parsers only when needed
        self._marker = None
        self._nougat = None
        self._pdffigures = None
        self._docling = None
        self._pymupdf4llm = None

    @property
    def marker(self):
        """Lazy load Marker parser."""
        if self._marker is None:
            self._marker = MarkerParser()
        return self._marker

    @property
    def nougat(self):
        """Lazy load Nougat parser."""
        if self._nougat is None:
            self._nougat = NougatParser()
        return self._nougat

    @property
    def pdffigures(self):
        """Lazy load PDFFigures parser."""
        if self._pdffigures is None:
            self._pdffigures = PDFFiguresParser()
        return self._pdffigures

    @property
    def docling(self):
        """Lazy load Docling parser."""
        if self._docling is None:
            self._docling = DoclingParser()
        return self._docling

    @property
    def pymupdf4llm(self):
        """Lazy load PyMuPDF4LLM parser."""
        if self._pymupdf4llm is None:
            self._pymupdf4llm = PyMuPDF4LLMParser()
        return self._pymupdf4llm

    def extract_hierarchy(
        self,
        pdf_path: str,
        extract_figures: Optional[bool] = None
    ) -> Dict:
        """
        Extract hierarchical text from PDF.

        Routes to appropriate extraction method based on mode.

        Args:
            pdf_path: Path to PDF file
            extract_figures: Override instance setting

        Returns:
            Dictionary with:
                - text_elements (List[Dict]): Hierarchical text elements
                - figures (List[Dict]): Figure metadata (if enabled)
                - extraction_method (str or Dict): Parser(s) used
                - success (bool): Whether extraction succeeded
                - page_routing (Dict): Page-level routing info (routing mode only)
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if self.mode == "routing":
            return self._extract_with_routing(pdf_file, extract_figures)
        else:
            return self._extract_with_fallback(pdf_file, extract_figures)

    def _extract_with_routing(
        self,
        pdf_file: Path,
        extract_figures: Optional[bool]
    ) -> Dict:
        """
        Medical-grade extraction with page-level routing.

        Detection → Routing → Hybrid merge.
        """
        try:
            import fitz  # PyMuPDF
            import pymupdf4llm
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for routing mode. "
                "Install with: pip install PyMuPDF pymupdf4llm pymupdf-layout"
            )

        # Check if pymupdf-layout is installed (used automatically by PyMuPDF)
        # pymupdf-layout enhances PyMuPDF's table detection and layout analysis
        try:
            import pkg_resources
            pkg_resources.get_distribution('pymupdf-layout')
            use_enhanced_detection = True
            self.logger.info("pymupdf-layout detected - enhanced table/layout detection active")
        except (ImportError, pkg_resources.DistributionNotFound):
            use_enhanced_detection = True  # Still use our enhanced logic
            self.logger.info("Using enhanced detection with line-based table detection")

        self.logger.info(f"Starting routing mode extraction: {pdf_file.name}")

        doc = fitz.open(str(pdf_file))
        page_routes = []  # Track routing decisions
        all_markdown = []  # Collect markdown from all pages

        # Phase 1: Fast detection and routing
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Detect content type (millisecond speed)
            if use_enhanced_detection:
                # Enhanced detection with line-based table detection
                # Note: PyMuPDF automatically uses pymupdf-layout if installed
                tabs = page.find_tables()
                has_tables = len(tabs.tables) > 0

                # Get page text for keyword-based detection
                page_text = page.get_text()

                # MEDICAL-GRADE TABLE DETECTION
                # Docling is "Truth" for data - be aggressive about routing tables
                if not has_tables:
                    # 1. Check for grid-like structures (lowered threshold)
                    paths = page.get_drawings()
                    line_count = sum(1 for p in paths if p.get('type') == 'l')
                    has_tables = line_count > 15  # Lowered from 20

                    # 2. Keyword-based detection - medical data indicators
                    if not has_tables:
                        table_keywords = [
                            'Table', 'TABLE',
                            'Percentage', '%',
                            'n =', 'N =',
                            'p <', 'P <', 'p=', 'P=',
                            'Mean', 'SD', 'CI',
                            'Patient', 'Case',
                            'Median', 'Range',
                            'vs.', 'versus'
                        ]
                        has_tables = any(keyword in page_text for keyword in table_keywords)

                # Image detection
                images = page.get_images()
                has_images = len(images) > 0
                image_count = len(images)

                # MARKER ROUTING CONDITIONS
                # Detect complex academic layouts that need Marker's specialist handling

                # 1. Dense mathematical formulas
                math_symbols = ['∑', '√', '∫', '±', '≤', '≥', '×', '÷', 'σ', 'μ', 'α', 'β']
                has_complex_math = sum(page_text.count(sym) for sym in math_symbols) > 5

                # 2. Multi-panel micrographs (>3 images in close proximity)
                has_multi_panel = image_count > 3

                # 3. Footnote-heavy pages (small fonts at bottom)
                text_blocks = page.get_text("dict")["blocks"]
                small_font_blocks = []
                page_height = page.rect.height

                for block in text_blocks:
                    if block.get('type') == 0:  # Text block
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                font_size = span.get('size', 0)
                                y_pos = span.get('bbox', [0,0,0,0])[3]  # Bottom y-coordinate

                                # Small font (<8pt) in bottom 20% of page
                                if font_size < 8 and y_pos > page_height * 0.8:
                                    small_font_blocks.append(span)

                has_footnotes = len(small_font_blocks) > 3

                # Determine if Marker should be used
                needs_marker = has_complex_math or has_multi_panel or has_footnotes

                has_text = len(text_blocks) > 0
            else:
                # Basic detection (fallback)
                has_tables = len(page.find_tables().tables) > 0
                has_images = len(page.get_images()) > 0
                has_text = True  # Assume text present
                needs_marker = False
                image_count = len(page.get_images()) if has_images else 0

            # ROUTING DECISION - Priority order for medical papers:
            # 1. Docling for tables (Truth for data)
            # 2. Marker for complex academic layouts
            # 3. PyMuPDF4LLM for everything else (fast, multi-column)

            if has_tables:
                # Docling is Truth for data
                route = 'docling'
                reason = 'table_detected'
            elif needs_marker and self.marker.is_available():
                # Marker for complex academic layouts
                route = 'marker'
                if has_complex_math:
                    reason = 'complex_math_detected'
                elif has_multi_panel:
                    reason = 'multi_panel_figures'
                elif has_footnotes:
                    reason = 'footnote_heavy'
                else:
                    reason = 'complex_layout'
            elif has_images:
                # PyMuPDF4LLM with image extraction
                route = 'pymupdf4llm_images'
                reason = 'images_detected'
            else:
                # PyMuPDF4LLM for narrative multi-column text
                route = 'pymupdf4llm'
                reason = 'narrative_text'

            page_routes.append({
                'page': page_num,
                'route': route,
                'reason': reason,
                'has_tables': has_tables,
                'has_images': has_images,
                'has_text': has_text,
                'image_count': image_count,
                'needs_marker': needs_marker if use_enhanced_detection else False
            })

        doc.close()

        self.logger.info(f"Routing complete: {len(page_routes)} pages analyzed")

        # Phase 2: Execute extraction with routed parsers
        # Group consecutive pages by route for batch processing
        route_groups = self._group_consecutive_routes(page_routes)

        for route_type, page_nums in route_groups.items():
            self.logger.info(f"Processing {len(page_nums)} pages with {route_type}")

            if route_type == 'docling':
                # Use Docling for table pages (Truth for data)
                if self.docling.is_available():
                    md = self._extract_pages_with_docling(pdf_file, page_nums)
                else:
                    # Fallback to PyMuPDF4LLM
                    self.logger.warning("Docling not available, using PyMuPDF4LLM")
                    md = pymupdf4llm.to_markdown(str(pdf_file), pages=page_nums)
                all_markdown.append(md)

            elif route_type == 'marker':
                # Use Marker for complex academic layouts
                if self.marker.is_available():
                    # Marker processes full documents, extract and filter by pages
                    # For now, fall back to PyMuPDF4LLM with special handling
                    # TODO: Implement page-specific Marker extraction
                    self.logger.info("Marker routing detected, using PyMuPDF4LLM (Marker integration pending)")
                    md = pymupdf4llm.to_markdown(str(pdf_file), pages=page_nums)
                else:
                    self.logger.warning("Marker not available, using PyMuPDF4LLM")
                    md = pymupdf4llm.to_markdown(str(pdf_file), pages=page_nums)
                all_markdown.append(md)

            elif route_type == 'pymupdf4llm_images':
                # Use PyMuPDF4LLM with image extraction
                md = pymupdf4llm.to_markdown(
                    str(pdf_file),
                    pages=page_nums,
                    write_images=True,
                    image_path=str(pdf_file.parent / 'extracted_images')
                )
                all_markdown.append(md)

            else:  # route_type == 'pymupdf4llm'
                # Use PyMuPDF4LLM for narrative text
                md = pymupdf4llm.to_markdown(str(pdf_file), pages=page_nums)
                all_markdown.append(md)

        # Phase 3: Merge and parse markdown to hierarchy
        combined_markdown = "\n\n".join(all_markdown)
        hierarchy = self.pymupdf4llm._parse_markdown_to_hierarchy(combined_markdown)

        # Phase 4: Deduplicate (remove headers/footers)
        if self.deduplicate_enabled and hierarchy:
            original_count = len(hierarchy)
            hierarchy = self._deduplicate_with_page_info(hierarchy, page_routes)
            self.logger.info(
                f"Deduplication: {original_count} → {len(hierarchy)} elements "
                f"({original_count - len(hierarchy)} removed)"
            )

        # Extract figures if needed
        figures = []
        if extract_figures or (extract_figures is None and self.extract_figures_enabled):
            figures = self._extract_figures(pdf_file)

        result = {
            'text_elements': hierarchy,
            'figures': figures,
            'extraction_method': self._summarize_routes(page_routes),
            'success': len(hierarchy) > 0,
            'page_routing': {
                'total_pages': len(page_routes),
                'routes': page_routes,
                'summary': self._route_summary(page_routes)
            }
        }

        self.logger.info(
            f"Routing extraction complete: {len(hierarchy)} text elements, "
            f"{len(figures)} figures"
        )

        return result

    def _group_consecutive_routes(
        self,
        page_routes: List[Dict]
    ) -> Dict[str, List[int]]:
        """
        Group consecutive pages with same route for batch processing.

        Args:
            page_routes: List of page routing decisions

        Returns:
            Dict mapping route_type to list of page numbers
        """
        groups = {}
        current_route = None
        current_pages = []

        for route_info in page_routes:
            route = route_info['route']
            page = route_info['page']

            if route != current_route:
                # Save previous group
                if current_pages:
                    if current_route not in groups:
                        groups[current_route] = []
                    groups[current_route].extend(current_pages)

                # Start new group
                current_route = route
                current_pages = [page]
            else:
                current_pages.append(page)

        # Save last group
        if current_pages:
            if current_route not in groups:
                groups[current_route] = []
            groups[current_route].extend(current_pages)

        return groups

    def _extract_pages_with_docling(
        self,
        pdf_file: Path,
        page_nums: List[int]
    ) -> str:
        """
        Extract specific pages with Docling.

        Note: Docling may not support page-level extraction directly,
        so we extract full document and filter if needed.

        Args:
            pdf_file: PDF file path
            page_nums: Page numbers to extract

        Returns:
            Markdown text
        """
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(pdf_file))
            markdown = result.document.export_to_markdown()

            # TODO: If Docling supports page filtering, implement here
            # For now, return full markdown (will be filtered in merge)

            return markdown

        except Exception as e:
            self.logger.error(f"Docling extraction failed: {e}")
            # Fallback to PyMuPDF4LLM
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(pdf_file), pages=page_nums)

    def _extract_figures(self, pdf_file: Path) -> List[Dict]:
        """Extract figures using PDFFigures."""
        if not self.pdffigures.is_available():
            return []

        try:
            self.logger.info("Extracting figures with PDFFigures 2.0")
            return self.pdffigures.extract_figures(str(pdf_file))
        except Exception as e:
            self.logger.error(f"Figure extraction failed: {e}")
            return []

    def _summarize_routes(self, page_routes: List[Dict]) -> Dict[str, int]:
        """Summarize which parsers were used."""
        summary = {}
        for route_info in page_routes:
            route = route_info['route']
            summary[route] = summary.get(route, 0) + 1
        return summary

    def _route_summary(self, page_routes: List[Dict]) -> str:
        """Human-readable routing summary."""
        summary = self._summarize_routes(page_routes)
        parts = [f"{count} pages via {route}" for route, count in summary.items()]
        return ", ".join(parts)

    def _deduplicate_with_page_info(
        self,
        hierarchy: List[Dict],
        page_routes: List[Dict]
    ) -> List[Dict]:
        """
        Deduplicate text elements using medical paper rules.

        Args:
            hierarchy: List of text elements
            page_routes: Page routing information

        Returns:
            Deduplicated list
        """
        if not hierarchy:
            return hierarchy

        # Build occurrence map
        text_occurrences = {}
        for i, elem in enumerate(hierarchy):
            text = elem.get('text', '').strip()
            if not text:
                continue

            if text not in text_occurrences:
                text_occurrences[text] = []
            text_occurrences[text].append(i)

        # Determine which elements to keep
        to_delete = set()

        for text, indices in text_occurrences.items():
            count = len(indices)

            # Rule 1: Delete if occurs >2 times (likely header/footer/page number)
            # Simple heuristic: if exact same text appears many times, it's repetitive
            if count > 2:
                # Check if it's likely a page number or header
                if len(text) < 20 or text.isdigit() or self._is_likely_header(text):
                    for idx in indices:
                        to_delete.add(idx)
                    self.logger.debug(f"Removing duplicate: '{text[:50]}...' ({count}x)")

        # Build deduplicated list
        deduplicated = [elem for i, elem in enumerate(hierarchy) if i not in to_delete]

        return deduplicated

    def _is_likely_header(self, text: str) -> bool:
        """Check if text is likely a header/footer."""
        text_lower = text.lower()

        # Common header/footer patterns
        header_patterns = [
            'page',
            'doi:',
            '©',
            'copyright',
            'journal',
            'volume',
            'issue',
            'published',
            'correspondence:',
            'author',
            'et al',
        ]

        return any(pattern in text_lower for pattern in header_patterns)

    def _extract_with_fallback(
        self,
        pdf_file: Path,
        extract_figures: Optional[bool]
    ) -> Dict:
        """
        Legacy fallback mode: try parsers sequentially on whole document.
        """
        self.logger.info(f"Starting fallback mode extraction: {pdf_file.name}")

        result = {
            'text_elements': [],
            'figures': [],
            'extraction_method': None,
            'success': False
        }

        # Choose parser order
        if self.parser_priority:
            parser_map = {
                'marker': (self.marker, 'marker'),
                'nougat': (self.nougat, 'nougat'),
                'docling': (self.docling, 'docling'),
                'pymupdf4llm': (self.pymupdf4llm, 'pymupdf4llm')
            }
            parsers = [parser_map[name] for name in self.parser_priority if name in parser_map]
        else:
            # Default order
            parsers = [
                (self.pymupdf4llm, 'pymupdf4llm'),
                (self.docling, 'docling'),
                (self.marker, 'marker'),
                (self.nougat, 'nougat')
            ]

        # Try parsers in order
        for parser, parser_name in parsers:
            if not parser.is_available():
                self.logger.info(f"{parser_name} not available, skipping")
                continue

            try:
                self.logger.info(f"Attempting extraction with {parser_name}")
                text_data = parser.extract_hierarchy(str(pdf_file))

                if text_data and len(text_data) >= self.min_content_threshold:
                    result['text_elements'] = text_data
                    result['extraction_method'] = parser_name
                    result['success'] = True
                    self.logger.info(f"Success with {parser_name}: {len(text_data)} elements")
                    break
                else:
                    self.logger.warning(
                        f"{parser_name} insufficient content: {len(text_data) if text_data else 0}"
                    )

            except Exception as e:
                self.logger.error(f"{parser_name} failed: {e}")
                continue

        # Extract figures
        if extract_figures or (extract_figures is None and self.extract_figures_enabled):
            result['figures'] = self._extract_figures(pdf_file)

        if not result['success']:
            self.logger.error(f"All parsers failed for: {pdf_file.name}")

        return result

    def get_available_parsers(self) -> Dict[str, bool]:
        """
        Check which parsers are available.

        Returns:
            Dictionary with parser availability status
        """
        return {
            'pymupdf4llm': self.pymupdf4llm.is_available(),
            'docling': self.docling.is_available(),
            'marker': self.marker.is_available(),
            'nougat': self.nougat.is_available(),
            'pdffigures': self.pdffigures.is_available()
        }

    def print_availability(self):
        """Print status of all parsers."""
        availability = self.get_available_parsers()

        print("\nPDF Parser Availability:")
        print("-" * 40)
        for parser, available in availability.items():
            status = "✓ Available" if available else "✗ Not available"
            print(f"{parser.ljust(15)}: {status}")
        print()


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
        pdf_file = "files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf"

    if Path(pdf_file).exists():
        print(f"\n{'='*60}")
        print("Testing Medical-Grade Ensemble Parser")
        print(f"File: {pdf_file}")
        print(f"{'='*60}\n")

        # Create parser in routing mode
        parser = EnsemblePDFParser(mode="routing")

        # Check availability
        parser.print_availability()

        try:
            # Extract content
            print("Extracting with intelligent routing...\n")

            result = parser.extract_hierarchy(pdf_file)

            # Display results
            if result['success']:
                print(f"\n{'='*60}")
                print("Extraction Results")
                print(f"{'='*60}\n")

                print("Mode: routing (page-level)")
                print(f"Text elements: {len(result['text_elements'])}")
                print(f"Figures: {len(result['figures'])}\n")

                # Show routing info
                if 'page_routing' in result:
                    routing = result['page_routing']
                    print("Routing Summary:")
                    print(f"  Total pages: {routing['total_pages']}")
                    print(f"  {routing['summary']}\n")

                # Show parser usage
                print(f"Parsers used: {result['extraction_method']}\n")

                # Show first few text elements
                print("First 5 text elements:")
                print("-" * 60)
                for i, elem in enumerate(result['text_elements'][:5], 1):
                    print(f"\n{i}. {elem['path_string']} (depth {elem['depth']})")
                    print(f"   {elem['text'][:150]}...")

                if len(result['text_elements']) > 5:
                    print(f"\n... and {len(result['text_elements']) - 5} more elements")

            else:
                print("\nERROR: Extraction failed")
                sys.exit(1)

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"File not found: {pdf_file}")
        print("Usage: python ensemble_parser.py <pdf_file>")
        sys.exit(1)
