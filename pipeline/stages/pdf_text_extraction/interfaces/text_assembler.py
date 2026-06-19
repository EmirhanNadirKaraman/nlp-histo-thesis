from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from pipeline.stages.pdf_text_extraction.models.dto import HierarchicalRow, LayoutResult


@runtime_checkable
class TextAssembler(Protocol):
    """
    Contract for hierarchical text assembly components.

    An implementation walks a LayoutResult in document order, tracks section
    headers to build a hierarchical path, and returns one HierarchicalRow per
    paragraph (after stitching and citation removal).
    """

    def assemble(self, layout: LayoutResult) -> List[HierarchicalRow]:
        """Assemble the layout into hierarchical rows in document order."""
        ...
