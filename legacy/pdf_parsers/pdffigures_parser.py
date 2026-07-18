"""
PDFFigures 2.0 Parser

Specialized parser for extracting figures, tables, and their captions from PDFs.
Uses Allen AI's PDFFigures 2.0 (Java-based tool) for bounding box detection.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging


class PDFFiguresParser:
    """
    Specialized parser for figure and table extraction.

    PDFFigures 2.0 is a Scala-based tool that detects figures and tables
    in PDFs using layout analysis and extracts their captions.

    Note: This is not a BasePDFParser subclass because it serves a different
    purpose (figure extraction rather than text extraction).

    Strategy:
        1. Run pdffigures2.jar command line tool
        2. Parse JSON metadata output
        3. Return figure information for database storage
    """

    def __init__(self, pdffigures_jar_path: Optional[str] = None):
        """
        Initialize PDFFigures parser.

        Args:
            pdffigures_jar_path: Path to pdffigures2.jar (auto-detected if None)
        """
        self.logger = logging.getLogger(__name__)
        self.jar_path = pdffigures_jar_path or self._find_jar()

    def _find_jar(self) -> Optional[str]:
        """
        Try to locate pdffigures2.jar in common locations.

        Returns:
            Path to jar file if found, None otherwise
        """
        search_paths = [
            Path.home() / '.local/bin/pdffigures2.jar',
            Path('/usr/local/bin/pdffigures2.jar'),
            Path('./pdffigures2.jar'),
            Path(__file__).parent.parent.parent / 'pdffigures2.jar',
        ]

        for path in search_paths:
            if path.exists():
                self.logger.info(f"Found pdffigures2.jar at: {path}")
                return str(path)

        self.logger.warning("pdffigures2.jar not found in standard locations")
        return None

    def is_available(self) -> bool:
        """
        Check if PDFFigures 2.0 is available.

        Returns:
            True if pdffigures2.jar exists and Java is available
        """
        if self.jar_path is None:
            return False

        if not Path(self.jar_path).exists():
            return False

        # Check if Java is available
        try:
            result = subprocess.run(
                ['java', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def extract_figures(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        save_images: bool = True
    ) -> List[Dict]:
        """
        Extract figures, tables, and captions from PDF using SBT.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save outputs (default: temp directory)
            save_images: Whether to save figure images

        Returns:
            List of dictionaries with figure metadata:
            {
                'figure_id': str,        # e.g., "Figure1", "Table2"
                'figure_type': str,      # "Figure" or "Table"
                'caption': str,          # Caption text
                'page': int,             # Page number (0-indexed)
                'bbox': List[float],     # [x0, y0, x1, y1] bounding box
                'image_path': str        # Path to extracted image (if saved)
            }
        """
        if not self.is_available():
            self.logger.warning("PDFFigures 2.0 not available, skipping figure extraction")
            return []

        pdf_file = Path(pdf_path).resolve()
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Prepare output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            # Use temp directory in same location as PDF
            output_path = pdf_file.parent / f".pdffigures_{pdf_file.stem}"

        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare metadata file
            metadata_file = output_path / f"{pdf_file.stem}.json"

            # Build SBT command for extraction (not visualization)
            # Use FigureExtractor for JSON output, not FigureExtractorVisualizationCli
            sbt_args = [str(pdf_file), '-m', str(metadata_file)]

            if save_images:
                sbt_args.extend(['-d', str(output_path)])

            # Create SBT command
            sbt_command = f'runMain org.allenai.pdffigures2.FigureExtractor {" ".join(sbt_args)}'

            self.logger.info(f"Running PDFFigures 2.0 on: {pdf_file.name}")
            self.logger.debug(f"SBT command: {sbt_command}")

            # Find pdffigures2 directory
            pdffigures_dir = self._find_pdffigures_dir()
            if not pdffigures_dir:
                self.logger.error("PDFFigures2 directory not found")
                return []

            # Run SBT with the command piped in
            # Using echo to avoid interactive mode
            result = subprocess.run(
                f'cd {pdffigures_dir} && echo "{sbt_command}" | sbt',
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                self.logger.error(f"PDFFigures failed: {result.stderr}")
                return []

            # Parse metadata JSON
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    raw_data = json.load(f)

                # Convert to standardized format
                figures = self._parse_metadata(raw_data, output_path, pdf_file.stem)

                self.logger.info(f"Extracted {len(figures)} figures/tables")
                return figures
            else:
                self.logger.warning(f"Metadata file not created: {metadata_file}")
                return []

        except subprocess.TimeoutExpired:
            self.logger.error(f"PDFFigures timed out processing: {pdf_file.name}")
            return []

        except Exception as e:
            self.logger.error(f"PDFFigures extraction failed: {e}")
            return []

    def _find_pdffigures_dir(self) -> Optional[str]:
        """Find pdffigures2 directory."""
        search_paths = [
            Path('pdffigures2'),
            Path(__file__).parent.parent.parent / 'pdffigures2',
            Path.home() / 'pdffigures2',
        ]

        for path in search_paths:
            if path.exists() and (path / 'build.sbt').exists():
                return str(path)

        return None

    def _parse_metadata(
        self,
        raw_data: List[Dict],
        output_dir: Path,
        pdf_stem: str
    ) -> List[Dict]:
        """
        Parse PDFFigures JSON output to standardized format.

        Args:
            raw_data: Raw JSON from PDFFigures
            output_dir: Directory where images are saved
            pdf_stem: PDF filename stem (for constructing image paths)

        Returns:
            List of figure metadata dictionaries
        """
        figures = []

        for item in raw_data:
            # Extract figure metadata
            figure_id = item.get('name', '')  # e.g., "Figure 1", "Table 2"
            figure_type = item.get('figType', 'Figure')  # "Figure" or "Table"
            caption = item.get('caption', '')

            # Page number
            page = item.get('page', 0)

            # Bounding box [x0, y0, x1, y1]
            region_boundary = item.get('regionBoundary', {})
            bbox = [
                region_boundary.get('x1', 0),
                region_boundary.get('y1', 0),
                region_boundary.get('x2', 0),
                region_boundary.get('y2', 0)
            ]

            # Construct image path if it was saved
            image_path = None
            if figure_id:
                # PDFFigures saves images as: {pdf_stem}-{figure_id}.png
                # Example: PMC10047158-Figure1.png
                image_filename = f"{pdf_stem}-{figure_id.replace(' ', '')}.png"
                candidate_path = output_dir / image_filename

                if candidate_path.exists():
                    image_path = str(candidate_path.absolute())

            # Create standardized figure dict
            figure_dict = {
                'figure_id': figure_id,
                'figure_type': figure_type,
                'caption': caption,
                'page': page,
                'bbox': bbox,
                'image_path': image_path
            }

            figures.append(figure_dict)

        return figures


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
        print(f"Testing PDFFiguresParser on: {pdf_file}")
        print(f"{'='*60}\n")

        parser = PDFFiguresParser()

        if not parser.is_available():
            print("ERROR: PDFFigures 2.0 is not available.")
            print("\nInstallation instructions:")
            print("1. Download: wget https://github.com/allenai/pdffigures2/releases/download/v0.1.0/pdffigures2.jar")
            print("2. Install to: ~/.local/bin/pdffigures2.jar")
            print("3. Ensure Java is installed: java -version")
            sys.exit(1)

        try:
            # Extract figures
            figures = parser.extract_figures(pdf_file, save_images=True)

            print(f"\nExtracted {len(figures)} figures/tables\n")

            if figures:
                for i, fig in enumerate(figures, 1):
                    print(f"Figure {i}:")
                    print(f"  ID: {fig['figure_id']}")
                    print(f"  Type: {fig['figure_type']}")
                    print(f"  Page: {fig['page']}")
                    print(f"  Caption: {fig['caption'][:100]}...")
                    print(f"  Image: {fig['image_path']}")
                    print()
            else:
                print("No figures found in this PDF")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"File not found: {pdf_file}")
        print("Usage: python pdffigures_parser.py <pdf_file>")
        sys.exit(1)
