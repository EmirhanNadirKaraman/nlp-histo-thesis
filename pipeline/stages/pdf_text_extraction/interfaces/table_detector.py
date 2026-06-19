from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pipeline.stages.pdf_text_extraction.models.dto import TableDetectionResult


@runtime_checkable
class TableDetector(Protocol):
    """
    Contract for table detection components.

    Implementations (e.g., TATR, Docling, VLM, Hybrid) must return a
    TableDetectionResult for the given PDF path.
    """

    def detect(self, pdf_path: Path) -> TableDetectionResult:
        """Detect table regions in the given PDF."""
        ...