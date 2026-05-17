"""
DetectionVisualizer

Draws colour-coded bounding boxes directly onto a copy of the PDF using
PyMuPDF (fitz) and saves it as an annotated PDF.

Approach matches scripts/visualize_docling_full.py:
  - No matplotlib / numpy required
  - Outputs one annotated PDF per document (not per-page PNGs)
  - Layout elements are colour-coded by type with a legend on page 1
  - Detected table regions are drawn in red with source + score labels
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pipeline.stages.pdf_text_extraction.config import VisualizationConfig
from pipeline.stages.pdf_text_extraction.models.dto import LayoutResult, TableDetectionResult

logger = logging.getLogger(__name__)

# Colour scheme matching visualize_docling_full.py (R, G, B in 0-1 range)
_ELEMENT_COLORS = {
    "TABLE":               (0,    0.8,  0),
    "RECONSTRUCTED_TABLE": (0,    1,    0.5),
    "PICTURE":             (0.5,  0,    1),
    "FIGURE":              (1,    0,    0.5),
    "PARAGRAPH":           (0.7,  0.7,  0.7),
    "TITLE":               (1,    0.5,  0),
    "SECTION_HEADER":      (1,    0.5,  0),
    "LIST":                (0,    0.5,  1),
    "LIST_ITEM":           (0.3,  0.7,  1),
    "CAPTION":             (1,    0.8,  0),
    "PAGE_HEADER":         (0.5,  0.5,  0.5),
    "PAGE_FOOTER":         (0.5,  0.5,  0.5),
    "FOOTNOTE":            (0.8,  0.4,  0.2),
    "TEXT":                (0.6,  0.6,  0.9),
    "UNKNOWN":             (0.5,  0.5,  0.5),
}
_DETECTION_COLORS = {
    "docling": (1,    0,    0   ),   # red
    "tatr":    (0,    0.6,  1   ),   # blue
    "hybrid":  (1,    0.5,  0   ),   # orange
    "vlm":     (0.6,  0,    0.8 ),   # purple
}
_DETECTION_COLOR_DEFAULT = (0.5, 0.5, 0.5)   # grey fallback


class DetectionVisualizer:
    """
    Saves annotated PDF visualizations using PyMuPDF only.

    Parameters
    ----------
    config:
        VisualizationConfig controlling which outputs to generate.
    output_dir:
        Directory where annotated PDFs are written.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._config = config or VisualizationConfig()
        self._output_dir = output_dir or Path("out/visualization")

    def visualize_layout(self, layout: LayoutResult, pmcid: str) -> None:
        """Save an annotated PDF with colour-coded layout element boxes."""
        if not self._config.enabled or not self._config.save_combined_visualization:
            return
        self._render(layout.pdf_path, layout=layout, detection=None, pmcid=pmcid)

    def visualize_detections(
        self,
        detection: TableDetectionResult,
        layout: Optional[LayoutResult],
        pmcid: str,
    ) -> None:
        """Save an annotated PDF with table detection boxes (red)."""
        if not self._config.enabled or not self._config.save_tatr_visualization:
            return
        self._render(detection.pdf_path, layout=layout, detection=detection, pmcid=pmcid)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _render(
        self,
        pdf_path: Path,
        layout: Optional[LayoutResult],
        detection: Optional[TableDetectionResult],
        pmcid: str,
    ) -> None:
        try:
            import fitz  # type: ignore
        except ImportError:
            logger.warning("Visualization requires PyMuPDF (fitz) — skipping.")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(pdf_path))
        max_pages = self._config.max_pages or len(doc)
        type_counts: dict = {}

        # ── Pre-compute detection rects per page for overlap checks ──────────
        detection_rects_by_page: dict[int, list] = {}
        if detection:
            for region in detection.regions:
                pg = region.bbox.page
                if 1 <= pg <= len(doc):
                    detection_rects_by_page.setdefault(pg, []).append(
                        region.bbox.to_fitz_rect(doc[pg - 1].rect.height)
                    )

        # ── Draw layout elements ──────────────────────────────────────────────
        if layout:
            for el in layout.elements:
                if el.page < 1 or el.page > len(doc):
                    continue
                if el.page > max_pages:
                    continue
                page = doc[el.page - 1]
                rect = el.bbox.to_fitz_rect(page.rect.height)
                if rect.is_empty:
                    continue

                # Check if this element overlaps any detection region
                will_be_masked = any(
                    rect.intersects(dr)
                    for dr in detection_rects_by_page.get(el.page, [])
                )

                if will_be_masked:
                    # Yellow filled highlight for elements that will be masked
                    page.draw_rect(rect, color=(1, 0.8, 0), fill=(1, 1, 0.6), width=1.5)
                    page.insert_text(
                        (rect.x0 + 1, rect.y0 + 8),
                        f"MASK:{el.type[:3]}",
                        fontsize=6,
                        color=(0.6, 0.4, 0),
                    )
                else:
                    color = _ELEMENT_COLORS.get(el.type, _ELEMENT_COLORS["UNKNOWN"])
                    page.draw_rect(rect, color=color, width=1, dashes="[2] 0")
                    page.insert_text(
                        (rect.x0 + 1, rect.y0 + 8),
                        el.type[:3].upper(),
                        fontsize=6,
                        color=color,
                    )
                type_counts[el.type] = type_counts.get(el.type, 0) + 1

        # ── Draw detection regions ────────────────────────────────────────────
        if detection:
            for region in detection.regions:
                if region.bbox.page < 1 or region.bbox.page > len(doc):
                    continue
                if region.bbox.page > max_pages:
                    continue
                page = doc[region.bbox.page - 1]
                rect = region.bbox.to_fitz_rect(page.rect.height)
                if rect.is_empty:
                    continue
                color = _DETECTION_COLORS.get(region.source, _DETECTION_COLOR_DEFAULT)
                page.draw_rect(rect, color=color, width=2)
                page.insert_text(
                    (rect.x0 + 1, rect.y0 + 8),
                    f"{region.source}:{region.score:.2f}",
                    fontsize=6,
                    color=color,
                )

        # ── Legend on page 1 ─────────────────────────────────────────────────
        if len(doc) > 0 and type_counts:
            first_page = doc[0]
            lx, ly = 20, 20
            legend_h = 35 + len(type_counts) * 10
            first_page.draw_rect(
                fitz.Rect(lx - 5, ly - 5, lx + 150, ly + legend_h),
                color=(0, 0, 0), fill=(1, 1, 1), width=0.5,
            )
            first_page.insert_text((lx, ly + 10), "Layout elements:",
                                   fontsize=9, color=(0, 0, 0))
            for i, (etype, count) in enumerate(
                sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            ):
                y = ly + 25 + i * 10
                color = _ELEMENT_COLORS.get(etype, _ELEMENT_COLORS["UNKNOWN"])
                first_page.draw_line((lx, y - 3), (lx + 12, y - 3),
                                     color=color, width=1, dashes="[2] 0")
                first_page.insert_text((lx + 15, y), f"{etype} ({count})",
                                       fontsize=7, color=(0, 0, 0))

        suffix = "detections" if detection and not layout else "layout"
        out_path = self._output_dir / f"{pmcid}_{suffix}_vis.pdf"
        doc.save(str(out_path))
        doc.close()
        logger.info("Visualizer: saved %s", out_path.name)
