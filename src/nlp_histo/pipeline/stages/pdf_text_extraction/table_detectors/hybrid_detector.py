"""
HybridTableDetector

Combines results from multiple TableDetector implementations — by default
DoclingTableDetector and TATRTableDetector — and merges overlapping bounding
boxes using iterative boolean-overlap union (parsers/layout_utils.merge_rects).
The merge is by any non-zero rectangle intersection (``Rect.intersects``), NOT
an IoU threshold — see ``merge_rects`` for the exact rule (B-087).

The merged regions come from the union of all source detectors, deduplicated by
page-level rectangle intersection.  Each merged region is tagged
``source='hybrid'``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from nlp_histo.pipeline.stages.pdf_text_extraction.config import TATRConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, DetectedRegion, TableDetectionResult

logger = logging.getLogger(__name__)


class HybridTableDetector:
    """
    Table detector that fuses outputs from Docling and TATR.

    Parameters
    ----------
    detectors:
        List of TableDetector-compatible objects.  Defaults to
        ``[DoclingTableDetector(), TATRTableDetector(tatr_config)]``.
    tatr_config:
        Passed to TATRTableDetector when using the default detector list.
    """

    def __init__(
        self,
        detectors: Optional[List] = None,
        tatr_config: Optional[TATRConfig] = None,
    ) -> None:
        if detectors is not None:
            self._detectors = detectors
        else:
            from nlp_histo.pipeline.stages.pdf_text_extraction.table_detectors.docling_detector import DoclingTableDetector
            from nlp_histo.pipeline.stages.pdf_text_extraction.table_detectors.tatr_detector import TATRTableDetector

            self._detectors = [
                DoclingTableDetector(),
                TATRTableDetector(tatr_config or TATRConfig()),
            ]

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, pdf_path: Path) -> TableDetectionResult:
        """
        Run every sub-detector and merge their results.

        Args:
            pdf_path: Path to the PDF to analyse.

        Returns:
            TableDetectionResult with overlap-merged regions tagged ``source='hybrid'``.
        """
        all_results = [d.detect(pdf_path) for d in self._detectors]
        return self._merge(all_results, pdf_path)

    def detect_with_layout(self, layout, pdf_path: Path) -> TableDetectionResult:
        """
        Same as ``detect`` but reuses an existing LayoutResult for the Docling
        detector, avoiding a redundant extraction pass.

        Args:
            layout:   LayoutResult already produced for ``pdf_path``.
            pdf_path: Path to the PDF (used by non-Docling detectors).

        Returns:
            TableDetectionResult with merged regions.
        """
        from nlp_histo.pipeline.stages.pdf_text_extraction.table_detectors.docling_detector import DoclingTableDetector

        results = []
        for detector in self._detectors:
            if isinstance(detector, DoclingTableDetector):
                results.append(detector.detect_from_layout(layout))
            else:
                results.append(detector.detect(pdf_path))

        return self._merge(results, pdf_path)

    # ── Merging logic ─────────────────────────────────────────────────────────

    def _merge(
        self, results: List[TableDetectionResult], pdf_path: Path
    ) -> TableDetectionResult:
        """Union per-page bounding boxes from all detectors using merge_rects."""
        from nlp_histo.parsers.layout_utils import merge_rects

        # Collect page_dims from the first result that has them
        page_dims: dict = {}
        for r in results:
            if r.page_dims:
                page_dims = r.page_dims
                break

        # Group all fitz.Rect objects by page, keeping track of source/score
        # For simplicity we only need the rects for merging; metadata is rebuilt.
        page_rects: dict = {}      # page -> List[fitz.Rect]
        page_region_meta: dict = {}  # page -> List[(score, source)]

        for result in results:
            for region in result.regions:
                page = region.bbox.page
                page_h = page_dims.get(page, {}).get("height", 792.0)
                rect = region.bbox.to_fitz_rect(page_h)
                page_rects.setdefault(page, []).append(rect)
                page_region_meta.setdefault(page, []).append(
                    (region.score, region.source)
                )

        merged_regions: List[DetectedRegion] = []
        for page, rects in page_rects.items():
            page_h = page_dims.get(page, {}).get("height", 792.0)
            merged = merge_rects(rects)
            for rect in merged:
                # Assign the highest score among original regions that overlap this rect
                best_score = max(
                    (score for r, (score, _) in zip(
                        page_rects[page], page_region_meta[page]
                    ) if rect.intersects(r)),
                    default=1.0,
                )
                bbox = BoundingBox.from_fitz_rect(rect, page_h, page)
                merged_regions.append(
                    DetectedRegion(
                        bbox=bbox,
                        score=round(best_score, 3),
                        source="hybrid",
                        label="table",
                    )
                )

        sources = "+".join(sorted({r.source for res in results for r in res.regions}))
        logger.info(
            "HybridDetector: merged %d regions from [%s] → %d regions",
            sum(len(r.regions) for r in results),
            sources,
            len(merged_regions),
        )

        return TableDetectionResult(
            regions=merged_regions,
            source="hybrid",
            pdf_path=pdf_path,
            page_dims=page_dims,
        )
