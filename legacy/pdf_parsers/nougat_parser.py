"""
Nougat PDF Parser

Fallback parser using Nougat neural OCR for scientific documents.
Particularly good for equations and complex layouts.
"""

import sys
import re
from pathlib import Path
from typing import List, Dict
import logging

# Handle both direct execution and module import
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from legacy.pdf_parsers.base_parser import BasePDFParser, HierarchicalPathBuilder
else:
    from .base_parser import BasePDFParser, HierarchicalPathBuilder


class NougatParser(BasePDFParser):
    """
    PDF parser using Nougat's neural OCR for scientific documents.

    Nougat (Neural Optical Understanding for Academic documents) is a
    transformer-based model trained on arXiv and PMC papers. It handles
    equations and complex layouts better than traditional OCR.

    Strategy:
        1. Convert PDF pages to images
        2. Process each image with Nougat model
        3. Nougat outputs mMarkdown format (similar to Marker)
        4. Parse markdown to extract hierarchical structure
        5. Return standardized format

    Note: ~10x slower than Marker, use as fallback only
    """

    def __init__(self, model_name: str = "facebook/nougat-base"):
        """
        Initialize Nougat parser.

        Args:
            model_name: Hugging Face model name (default: facebook/nougat-base)
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Cache for models (loaded lazily)
        self._processor = None
        self._model = None
        self._device = None

    def is_available(self) -> bool:
        """Check if Nougat dependencies are installed."""
        try:
            from transformers import NougatProcessor  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_models(self):
        """Lazy load Nougat models (heavy operation)."""
        if self._processor is None:
            try:
                from transformers import NougatProcessor, VisionEncoderDecoderModel
                import torch

                self.logger.info(f"Loading Nougat model: {self.model_name}...")

                self._processor = NougatProcessor.from_pretrained(self.model_name)
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

                # Use GPU if available
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._model = self._model.to(self._device)

                self.logger.info(f"Nougat model loaded successfully (device: {self._device})")

            except Exception as e:
                self.logger.error(f"Failed to load Nougat models: {e}")
                raise

    def extract_hierarchy(self, pdf_path: str) -> List[Dict]:
        """
        Extract hierarchical text structure using Nougat.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of hierarchical text elements
        """
        # Validate PDF exists
        pdf_file = self._ensure_pdf_exists(pdf_path)

        # Convert PDF to mMarkdown using Nougat
        markdown_text = self._convert_to_markdown(str(pdf_file))

        # Parse markdown structure (reuse logic from Marker)
        data = self._parse_markdown_structure(markdown_text)

        # Validate output
        if not self.validate_output(data):
            self.logger.warning(f"Output validation failed for {pdf_path}")

        return data

    def _convert_to_markdown(self, pdf_path: str) -> str:
        """
        Convert PDF to mMarkdown using Nougat neural OCR.

        Args:
            pdf_path: Path to PDF file

        Returns:
            mMarkdown text
        """
        # Load models (cached after first call)
        self._load_models()

        try:
            self.logger.info(f"Converting PDF with Nougat: {Path(pdf_path).name}")

            # Convert PDF pages to images
            images = self._pdf_to_images(pdf_path)

            self.logger.info(f"Processing {len(images)} pages with Nougat...")

            # Process each page
            markdown_pages = []
            for page_num, img in enumerate(images, 1):
                self.logger.debug(f"Processing page {page_num}/{len(images)}")

                # Prepare image for model
                pixel_values = self._processor(img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self._device)

                # Generate markdown
                outputs = self._model.generate(
                    pixel_values,
                    max_length=self._model.decoder.config.max_length,
                    bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                )

                # Decode output
                page_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
                markdown_pages.append(page_text)

            # Combine all pages
            full_markdown = '\n\n'.join(markdown_pages)

            self.logger.info(f"Conversion complete. Text length: {len(full_markdown)} chars")
            return full_markdown

        except Exception as e:
            self.logger.error(f"Nougat conversion failed: {e}")
            raise

    def _pdf_to_images(self, pdf_path: str) -> List:
        """
        Convert PDF pages to PIL Images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Image objects
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
        except ImportError as e:
            raise ImportError(f"Required dependency missing: {e}")

        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page to image (150 DPI for good quality)
            pix = page.get_pixmap(dpi=150)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        return images

    def _parse_markdown_structure(self, markdown: str) -> List[Dict]:
        """
        Parse mMarkdown to extract hierarchical structure.

        Nougat outputs similar markdown format to Marker, so we can reuse
        the same parsing logic.

        Args:
            markdown: mMarkdown text from Nougat

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

                # Clean up header text
                header_text = self._clean_header_text(header_text)

                # Update path
                path_builder.process_header(header_text, header_level)

            else:
                # Accumulate text content
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
        header_text = re.sub(r'\*\*(.+?)\*\*', r'\1', header_text)
        header_text = re.sub(r'\*(.+?)\*', r'\1', header_text)
        header_text = re.sub(r'_(.+?)_', r'\1', header_text)

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
        pdf_file = "files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf"

    if Path(pdf_file).exists():
        print(f"\n{'='*60}")
        print(f"Testing NougatParser on: {pdf_file}")
        print(f"{'='*60}\n")

        parser = NougatParser()

        if not parser.is_available():
            print("ERROR: Nougat is not installed.")
            print("Install with: pip install transformers torch")
            sys.exit(1)

        try:
            # Extract hierarchy
            print("WARNING: Nougat is slow (~10x slower than Marker)")
            print("This may take several minutes...\n")

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

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"File not found: {pdf_file}")
        print("Usage: python nougat_parser.py <pdf_file>")
        sys.exit(1)
