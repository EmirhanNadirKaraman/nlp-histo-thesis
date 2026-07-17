"""
DoclingTableDetector

Extracts table regions directly from an existing LayoutResult produced by the
DoclingLayoutExtractor.  No additional model inference is required — this
simply filters elements whose type is TABLE or RECONSTRUCTED_TABLE.
"""
from __future__ import annotations

from pathlib import Path

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import DetectedRegion, TableDetectionResult


class DoclingTableDetector:
    """
    Table detector that reads TABLE/RECONSTRUCTED_TABLE elements from a
    Docling LayoutResult.

    Usage::

        layout = DoclingLayoutExtractor(config).extract(pdf_path)
        detector = DoclingTableDetector()
        result = detector.detect_from_layout(layout)
    """

    TABLE_TYPES = frozenset({"TABLE", "RECONSTRUCTED_TABLE"})

    def detect(self, pdf_path: Path) -> TableDetectionResult:
        """
        Run a fresh Docling extraction and return table regions.

        Prefer ``detect_from_layout`` when you already have a LayoutResult to
        avoid redundant Docling processing.
        """
        from nlp_histo.pipeline.stages.pdf_text_extraction.components.layout_extractor import DoclingLayoutExtractor
        from nlp_histo.pipeline.stages.pdf_text_extraction.config import DoclingConfig

        layout = DoclingLayoutExtractor(DoclingConfig()).extract(pdf_path)
        return self.detect_from_layout(layout)

    def detect_from_layout(
        self,
        layout,  # LayoutResult — avoid circular import at module level
    ) -> TableDetectionResult:
        """
        Extract TABLE/RECONSTRUCTED_TABLE regions from an existing LayoutResult.

        Args:
            layout: A LayoutResult produced by any LayoutExtractor.

        Returns:
            TableDetectionResult with one DetectedRegion per table element.
        """
        regions = []
        for el in layout.elements:
            if el.type in self.TABLE_TYPES:
                regions.append(
                    DetectedRegion(
                        bbox=el.bbox,
                        score=1.0,
                        source="docling",
                        label="table",
                    )
                )

        return TableDetectionResult(
            regions=regions,
            source="docling",
            pdf_path=layout.pdf_path,
            page_dims=layout.page_dims,
        )
