"""
PyMuPDFRegionMasker — paint opaque white rectangles on a PDF copy via PyMuPDF.

Used in two distinct spots:

* `collect_regions()` + `mask()` — Step 3 of the standard flow.  Collects
  table regions (from `TableDetectionResult`), figure regions (from
  `LayoutResult`), and optionally header/footer/sidebar regions (when
  `MaskingConfig.mask_header_footer_sidebar=True`), then writes a masked
  `<stem>_masked.pdf` for Step 4 re-extraction.
* `mask_figures_only()` — Stage-2 detector-input helper used by
  `PipelineRunner._detection_pdf_path()` when
  `MaskingConfig.mask_figures_before_table_detection=True`.  Paints only
  PICTURE / FIGURE bboxes from `LayoutResult` to give pixel-based table
  detectors (TATR / Hybrid) a figure-free view of the PDF.  Writes
  `<stem>_fig_masked.pdf`; returns `None` (and falls back to the original
  PDF) when the layout contains no figures.  Downstream Step-7 cropping
  always uses the original PDF — this helper only affects detector input.

Header / footer / sidebar detection runs locally in
`_detect_header_footer_sidebars` using the heuristics defined alongside
the class (`_SIDEBAR_MAX_W`, `_COLUMN_GAP_MIN`, and the scispaCy
`nlp_is_meaningful` helper from `parsers.layout_utils`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from nlp_histo.pipeline.stages.pdf_text_extraction.config import MaskingConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, LayoutResult, TableDetectionResult
from nlp_histo.parsers.layout_utils import MIN_ANCHOR_H, nlp_is_meaningful

logger = logging.getLogger(__name__)

_SIDEBAR_MAX_W  = 150  # pts — narrowness threshold for annotation columns
_COLUMN_GAP_MIN = 50   # pts — minimum x1 gap to identify a column boundary


class PyMuPDFRegionMasker:
    """
    Masks a list of bounding boxes in a PDF using PyMuPDF (fitz).

    Parameters
    ----------
    config:
        MaskingConfig controlling whether to expand boxes, merge overlaps, etc.
    output_dir:
        Directory where masked PDFs are written.  Defaults to the same
        directory as the input PDF if not provided.
    """

    def __init__(
        self,
        config: Optional[MaskingConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._config = config or MaskingConfig()
        self._output_dir = output_dir

    _FIGURE_TYPES = frozenset({"PICTURE", "FIGURE"})

    def collect_regions(
        self,
        detection: TableDetectionResult,
        layout: LayoutResult,
        nlp=None,
    ) -> List[BoundingBox]:
        """
        Build the list of bounding boxes to mask, driven by config flags.

        Args:
            detection: Table detection result (used when mask_tables=True).
            layout:    Full layout result (used when mask_figures=True and
                       mask_header_footer_sidebar=True).
            nlp:       Optional scispaCy model for NER-based fallback in
                       header/footer detection (Category 3).  Skipped if None.

        Returns:
            Combined list of BoundingBoxes to paint white.
        """
        regions: List[BoundingBox] = []
        if self._config.mask_tables:
            regions.extend(r.bbox for r in detection.regions)
        if self._config.mask_figures:
            regions.extend(
                el.bbox for el in layout.elements if el.type in self._FIGURE_TYPES
            )
        if self._config.mask_header_footer_sidebar:
            regions.extend(self._detect_header_footer_sidebars(layout, nlp=nlp))
        return regions

    def _detect_header_footer_sidebars(
        self,
        layout: LayoutResult,
        nlp=None,
    ) -> List[BoundingBox]:
        """
        Detect sidebar, header, and footer regions for masking.

        Ported directly from merged_pipeline._detect_header_footer_elements:
          - Category 1 — Sidebar: narrow left/right annotation columns,
            detected by x1 gap analysis.
          - Category 2 — Header/footer: full-width strips outside anchor
            block y-bounds per page.
          - Category 3 — NER fallback: single-line TEXT elements on pages
            with no anchor blocks (only when nlp is provided).
        """
        docling_elements = layout.to_element_dicts()
        page_dims        = layout.page_dims

        anchor_x1s = sorted(
            el['bbox']['x1'] for el in docling_elements
            if abs(el['bbox'].get('y1', 0) - el['bbox'].get('y2', 0)) >= MIN_ANCHOR_H
        )
        sig_gaps = []
        for i in range(len(anchor_x1s) - 1):
            gap = anchor_x1s[i + 1] - anchor_x1s[i]
            if gap > _COLUMN_GAP_MIN:
                sig_gaps.append((gap, anchor_x1s[i], anchor_x1s[i + 1]))

        x_left_bound_main   = sig_gaps[0][2]  if sig_gaps           else None
        x_right_bound_start = sig_gaps[-1][1] if len(sig_gaps) >= 2 else None

        def _is_sidebar(el):
            b = el.get('bbox', {})
            x1, x2 = b.get('x1', 0), b.get('x2', 0)
            if (x2 - x1) >= _SIDEBAR_MAX_W:
                return False
            if x_left_bound_main  is not None and x2 < x_left_bound_main:
                return True
            if x_right_bound_start is not None and x1 > x_right_bound_start:
                return True
            return False

        # Pass 1: per-page anchor y-bounds (excluding sidebar elements)
        page_bounds: dict = {}
        for el in docling_elements:
            if _is_sidebar(el):
                continue
            page = el.get('page')
            if page is None:
                continue
            b = el.get('bbox', {})
            h = abs(b.get('y1', 0) - b.get('y2', 0))
            if h >= MIN_ANCHOR_H:
                y1, y2 = b['y1'], b['y2']
                if page in page_bounds:
                    top, bot = page_bounds[page]
                    page_bounds[page] = (max(top, y1), min(bot, y2))
                else:
                    page_bounds[page] = (y1, y2)

        # Pass 2: build mask list
        mask_bboxes: List[BoundingBox] = []

        def _dims(page):
            d = page_dims.get(page) or page_dims.get(str(page)) or {}
            return d.get('width', 595.0), d.get('height', 842.0)

        # Category 1: sidebar elements (mask by individual bbox)
        n_sidebar = 0
        for el in docling_elements:
            if _is_sidebar(el) and (el.get('text') or '').strip():
                b = el['bbox']
                mask_bboxes.append(BoundingBox(
                    x1=b['x1'], y1=b['y1'], x2=b['x2'], y2=b['y2'],
                    page=el['page'],
                ))
                n_sidebar += 1

        # Category 2: header/footer strips (full-width, per page)
        n_strips = 0
        for page, (top_bound, bot_bound) in page_bounds.items():
            pw, ph = _dims(page)
            if top_bound < ph:
                mask_bboxes.append(BoundingBox(x1=0, y1=ph, x2=pw, y2=top_bound, page=page))
                n_strips += 1
            if bot_bound > 0:
                mask_bboxes.append(BoundingBox(x1=0, y1=bot_bound, x2=pw, y2=0, page=page))
                n_strips += 1

        # Category 3: NER fallback for pages with no anchor blocks
        n_ner = 0
        if nlp is not None:
            pages_with_anchors = set(page_bounds.keys())
            for el in docling_elements:
                if el.get('type') != 'TEXT' or _is_sidebar(el):
                    continue
                page = el.get('page')
                if page in pages_with_anchors:
                    continue
                text = (el.get('text') or '').strip()
                b = el.get('bbox', {})
                h = abs(b.get('y1', 0) - b.get('y2', 0))
                if h < MIN_ANCHOR_H and not nlp_is_meaningful(text, nlp):
                    mask_bboxes.append(BoundingBox(
                        x1=b['x1'], y1=b['y1'], x2=b['x2'], y2=b['y2'],
                        page=page,
                    ))
                    n_ner += 1

        logger.info(
            "Header/footer/sidebar: %d sidebar, %d strip region(s), %d NER-filtered",
            n_sidebar, n_strips, n_ner,
        )
        return mask_bboxes

    def mask(self, pdf_path: Path, regions: List[BoundingBox]) -> Path:
        """
        Write a new PDF with all ``regions`` painted white.

        Args:
            pdf_path: Source PDF path.
            regions:  Regions to mask (Docling PDF coordinates).

        Returns:
            Path to the masked PDF (written next to the source or into output_dir).
        """
        import fitz  # type: ignore
        from nlp_histo.parsers.layout_utils import merge_rects

        out_dir = self._output_dir or pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdf_path.stem}_masked.pdf"

        doc = fitz.open(str(pdf_path))

        # Group regions by page
        by_page: dict = {}
        for bbox in regions:
            by_page.setdefault(bbox.page, []).append(bbox)

        exp = self._config.expand_box_px

        for page_num in range(len(doc)):
            page_no = page_num + 1
            if page_no not in by_page:
                continue

            page = doc[page_num]
            page_h = page.rect.height

            rects = []
            for b in by_page[page_no]:
                r = b.to_fitz_rect(page_h)
                if exp:
                    r = fitz.Rect(r.x0 - exp, r.y0 - exp, r.x1 + exp, r.y1 + exp)
                rects.append(r)

            if self._config.merge_overlapping_boxes:
                rects = merge_rects(rects)

            for rect in rects:
                if rect.is_empty:
                    continue
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

        doc.save(str(out_path))
        doc.close()
        logger.info("Masked PDF written to %s (%d pages affected)",
                    out_path, len(by_page))
        return out_path

    def mask_figures_only(
        self,
        pdf_path: Path,
        layout: LayoutResult,
    ) -> Optional[Path]:
        """Write a PDF where only PICTURE/FIGURE bboxes from ``layout`` are
        painted white.  Used to give the Step-2 table detector a clean
        figure-free view of the PDF without affecting downstream cropping.

        Returns the masked PDF path, or ``None`` if the layout contains no
        figures (in which case the caller should fall back to the original
        PDF).  Independent of ``MaskingConfig.mask_figures`` — that flag
        gates the standard Step-3 mask, not this detector-input mask.

        The output file is named ``<stem>_fig_masked.pdf`` to avoid
        collision with the standard ``<stem>_masked.pdf`` produced by
        :meth:`mask`.
        """
        import fitz  # type: ignore
        from nlp_histo.parsers.layout_utils import merge_rects

        fig_regions = [
            el.bbox for el in layout.elements if el.type in self._FIGURE_TYPES
        ]
        if not fig_regions:
            return None

        out_dir = self._output_dir or pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdf_path.stem}_fig_masked.pdf"

        doc = fitz.open(str(pdf_path))

        by_page: dict = {}
        for bbox in fig_regions:
            by_page.setdefault(bbox.page, []).append(bbox)

        exp = self._config.expand_box_px

        for page_num in range(len(doc)):
            page_no = page_num + 1
            if page_no not in by_page:
                continue
            page = doc[page_num]
            page_h = page.rect.height

            rects = []
            for b in by_page[page_no]:
                r = b.to_fitz_rect(page_h)
                if exp:
                    r = fitz.Rect(r.x0 - exp, r.y0 - exp, r.x1 + exp, r.y1 + exp)
                rects.append(r)

            if self._config.merge_overlapping_boxes:
                rects = merge_rects(rects)

            for rect in rects:
                if rect.is_empty:
                    continue
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

        doc.save(str(out_path))
        doc.close()
        logger.info(
            "Figure-only masked PDF (for table detection) written to %s "
            "(%d figure(s), %d page(s) affected)",
            out_path, len(fig_regions), len(by_page),
        )
        return out_path
