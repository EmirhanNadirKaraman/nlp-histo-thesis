from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox


@runtime_checkable
class RegionMasker(Protocol):
    """
    Contract for PDF region-masking components.

    An implementation draws opaque white rectangles over the supplied bounding
    boxes and writes the result to a new PDF file, returning its path.
    """

    def mask(self, pdf_path: Path, regions: List[BoundingBox]) -> Path:
        """Draw white rectangles over the regions (Docling coords) and return the masked PDF path."""
        ...
