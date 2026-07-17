"""
Base PDF Parser Interface

Abstract base class defining the common interface for all PDF parsers.
Ensures all parsers return consistent output format matching the XML parser.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path


class BasePDFParser(ABC):
    """
    Abstract base class for PDF parsers.

    All PDF parsers must implement this interface to ensure consistent output
    format that matches the XML parser's hierarchical structure.
    """

    @abstractmethod
    def extract_hierarchy(self, pdf_path: str) -> List[Dict]:
        """
        Extract hierarchical text structure from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries with keys:
                - path_list (List[str]): Hierarchical path as list
                - path_string (str): Hierarchical path as "> " separated string
                - depth (int): Nesting level (1, 2, 3, ...)
                - text (str): Text content of the element

        Example return value:
            [
                {
                    'path_list': ['Methods', '2.1 Staining'],
                    'path_string': 'Methods > 2.1 Staining',
                    'depth': 2,
                    'text': 'The tissues were stained with...'
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the parser's dependencies are installed and available.

        Returns:
            True if parser can be used, False otherwise
        """
        pass

    def validate_output(self, data: List[Dict]) -> bool:
        """
        Validate that parser output matches expected schema.

        Args:
            data: Parser output to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, list):
            return False

        required_keys = {'path_list', 'path_string', 'depth', 'text'}

        for item in data:
            if not isinstance(item, dict):
                return False

            # Check all required keys present
            if not all(key in item for key in required_keys):
                return False

            # Validate types
            if not isinstance(item['path_list'], list):
                return False
            if not isinstance(item['path_string'], str):
                return False
            if not isinstance(item['depth'], int):
                return False
            if not isinstance(item['text'], str):
                return False

            # Validate depth matches path_list length
            if item['depth'] != len(item['path_list']):
                return False

            # Validate path_string matches path_list
            expected_path_string = ' > '.join(item['path_list'])
            if item['path_string'] != expected_path_string:
                return False

        return True

    def _ensure_pdf_exists(self, pdf_path: str) -> Path:
        """
        Validate PDF file exists.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Path object

        Raises:
            FileNotFoundError: If PDF doesn't exist
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        return pdf_file


class HierarchicalPathBuilder:
    """
    Helper class to build hierarchical paths while parsing PDFs.

    Maintains state similar to the XML parser's recursive approach,
    but adapted for stateful line-by-line parsing of PDFs.
    """

    def __init__(self):
        """Initialize with empty path."""
        self.current_path = []

    def process_header(self, header_text: str, depth: int) -> List[str]:
        """
        Process a new section header and update the path.

        Args:
            header_text: The section header text
            depth: The hierarchical depth (1, 2, 3, ...)

        Returns:
            Copy of the updated path list
        """
        # Remove trailing elements deeper than current depth
        while len(self.current_path) > depth - 1:
            self.current_path.pop()

        # Add new section at current depth
        self.current_path.append(header_text.strip())

        return self.current_path.copy()

    def get_current_path_dict(self, text_content: str) -> Dict:
        """
        Get standardized output dictionary for current path and text.

        Args:
            text_content: The text content to include

        Returns:
            Dictionary matching the expected parser output format
        """
        return {
            'path_list': self.current_path.copy(),
            'path_string': ' > '.join(self.current_path),
            'depth': len(self.current_path),
            'text': text_content.strip()
        }

    def reset(self):
        """Reset path to empty."""
        self.current_path = []
