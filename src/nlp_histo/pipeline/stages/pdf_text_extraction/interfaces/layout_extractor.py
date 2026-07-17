from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import LayoutResult


@runtime_checkable
class LayoutExtractor(Protocol):
    """
    Contract for PDF layout extraction components.

    An implementation (e.g. DoclingLayoutExtractor) converts a raw PDF into
    a structured LayoutResult containing typed elements, bounding boxes, and
    per-page dimensions.
    """

    def extract(self, pdf_path: Path) -> LayoutResult:
        """Extract the full layout (typed elements + page metadata) from a PDF."""
        ...
