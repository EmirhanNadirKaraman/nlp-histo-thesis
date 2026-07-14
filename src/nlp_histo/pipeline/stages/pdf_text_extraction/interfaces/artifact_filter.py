from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import LayoutElement


@runtime_checkable
class ArtifactFilter(Protocol):
    """
    Contract for layout-artifact filtering components.

    An implementation removes spurious elements (headers, footers, sidebars,
    single-character floats, etc.) from a list of LayoutElements.
    """

    def filter_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Drop artifact elements (headers, footers, single-char floats, …)."""
        ...
