"""
TwoPassTextExtractor

Orchestrates the full two-pass PDF text-cleanup pipeline.

Pass 1 — extract, score, mask
------------------------------
  Step 1. Run DoclingLayoutExtractor on the original PDF.
  Step 2. Open PDF with PyMuPDF; gather word-overlap + pixel-brightness
          evidence for every Docling element.
  Step 3. Score each element: KEEP real text, REJECT invisible/ghost text.
  Step 4. For each page, find the top-most accepted TEXT/LIST_ITEM/PARAGRAPH
          element — the "body anchor" where real body text begins.
  Step 5. Compute per-page header mask regions (from page top to just above
          the body anchor).
  Step 6. Draw opaque white rectangles over all header-zone regions using
          PyMuPDF, writing a masked PDF to disk.

Pass 2 — re-extract, separate
------------------------------
  Step 7. Run DoclingLayoutExtractor on the masked PDF.
  Step 8. Separate elements by role: TEXT family / FIGURE / TABLE.
  Step 9. Return TwoPassResult.

Canonical vs. support role
---------------------------
  Docling is the canonical layout layer.  Every LayoutElement in the
  final output comes from Docling — never from fitz.
  PyMuPDF is used only for:
    • evidence gathering in Pass 1 (word lists, rendered pixmaps)
    • writing the masked PDF in Step 6
  No fitz-derived text nodes are ever merged into Docling's element list.

FIGURE and TABLE separation
----------------------------
  PICTURE/FIGURE elements are kept as ``figure_elements`` — the inner text
  they contain (axis labels, callouts) is intentionally excluded from body
  text assembly.
  TABLE/RECONSTRUCTED_TABLE elements are kept as ``table_elements`` — cell
  content must not be flattened into paragraph text.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nlp_histo.pipeline.stages.pdf_text_extraction.components.evidence_gatherer import (
    PyMuPDFEvidenceGatherer,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.components.layout_extractor import (
    DoclingLayoutExtractor,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.components.node_scorer import NodeScorer
from nlp_histo.pipeline.stages.pdf_text_extraction.config import (
    DoclingConfig,
    MaskingConfig,
    TwoPassConfig,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import (
    BoundingBox,
    LayoutElement,
    LayoutResult,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.models.scored_node import (
    HeaderAnchor,
    ScoredNode,
    TextNodeEvidence,
    TwoPassResult,
)

logger = logging.getLogger(__name__)

# Element type sets

# Elements eligible to act as the body anchor (top-most real text on a page)
_ANCHOR_ELIGIBLE = frozenset({"TEXT", "LIST_ITEM", "PARAGRAPH", "SECTION_HEADER"})

# Elements that may be masked by category-4 (rejected-node) redaction.
# Intentionally excludes SECTION_HEADER — section titles must never be
# auto-masked even if the scorer somehow rejects them.
_MASKABLE_TYPES = frozenset({"TEXT", "LIST_ITEM", "PARAGRAPH"})

# Pass-2 role classification
_FIGURE_TYPES = frozenset({"PICTURE", "FIGURE"})
_TABLE_TYPES  = frozenset({"TABLE", "RECONSTRUCTED_TABLE"})

# Elements always masked regardless of scoring (structural chrome, not body text).
# PAGE_HEADER / PAGE_FOOTER: running titles and page numbers — never body text.
# CAPTION: figure/table captions — already in SKIP_TYPES for the text assembler,
#   but masking them prevents Pass-2 Docling from reclassifying them as TEXT
#   once their parent figure/table has been redacted.
_ALWAYS_MASK = frozenset({"PAGE_HEADER", "PAGE_FOOTER", "CAPTION"})

# MaskingConfig preset: disable built-in table/figure/header masking so the
# two-pass extractor can apply its own targeted header mask only.
_NO_BUILTIN_MASK = MaskingConfig(
    mask_tables=False,
    mask_figures=False,
    mask_header_footer_sidebar=False,
    merge_overlapping_boxes=True,
    expand_box_px=0,
)

# Output filename suffix for the two-pass masked PDF
_MASKED_SUFFIX = "_twopass_masked.pdf"

# Sidebar detection thresholds (mirror PyMuPDFRegionMasker constants)
_SIDEBAR_MAX_W   = 150   # pts — element narrower than this is a sidebar candidate
_COLUMN_GAP_MIN  =  50   # pts — x1 gap larger than this marks a column boundary


class TwoPassTextExtractor:
    """
    Two-pass PDF text cleanup pipeline.

    Instantiate once and call ``process()`` for each PDF.  The two Docling
    extractors are created lazily and reused across calls to avoid repeated
    model loading.

    Parameters
    ----------
    config:
        TwoPassConfig — scoring thresholds and rendering DPI.
        Defaults to TwoPassConfig() if not provided.
    docling_config:
        DoclingConfig for Pass 1 (full layout extraction).
        Defaults to DoclingConfig().
    docling_text_config:
        DoclingConfig for Pass 2 (re-extraction from masked PDF).
        If None, reuses ``docling_config``.  Provide a separate instance
        to enable OCR on masked pages while keeping Pass 1 fast.
    cache_dir:
        Directory for Docling JSON layout caches.  If None, no caching.
    masked_pdf_dir:
        Output directory for header-masked PDFs.  Defaults to the input
        PDF's parent directory when not provided.
    """

    def __init__(
        self,
        config: Optional[TwoPassConfig] = None,
        docling_config: Optional[DoclingConfig] = None,
        docling_text_config: Optional[DoclingConfig] = None,
        cache_dir: Optional[Path] = None,
        masked_pdf_dir: Optional[Path] = None,
    ) -> None:
        self._cfg = config or TwoPassConfig()
        self._docling_cfg = docling_config or DoclingConfig()
        self._docling_text_cfg = docling_text_config or self._docling_cfg
        self._cache_dir = cache_dir
        self._masked_pdf_dir = masked_pdf_dir

        # Lazy-initialised; re-used across process() calls
        self._extractor1: Optional[DoclingLayoutExtractor] = None
        self._extractor2: Optional[DoclingLayoutExtractor] = None

    # Public API

    def process(self, pdf_path: Path) -> TwoPassResult:
        """
        Run the full two-pass pipeline on one PDF.

        Args:
            pdf_path: Path to the source PDF file.

        Returns:
            TwoPassResult containing scored Pass-1 nodes, header anchors,
            the masked PDF path, and the final separated Pass-2 elements.

        Raises:
            FileNotFoundError: if ``pdf_path`` does not exist.
            RuntimeError:      on unrecoverable Docling or masking failures.
        """
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("=== TwoPassTextExtractor: %s ===", pdf_path.name)

        # Pass 1
        pass1_layout   = self._pass1_layout(pdf_path)
        scored_nodes   = self._score_nodes(pdf_path, pass1_layout)
        body_extents   = self._find_body_extents(scored_nodes, pass1_layout.page_dims)
        header_anchors = {
            page: HeaderAnchor(page=page, element=ext["top_el"], top_y_fitz=ext["top_y"])
            for page, ext in body_extents.items()
            if ext["top_el"] is not None
        }
        masked_path    = self._apply_masks(
            pdf_path, pass1_layout.page_dims, body_extents, scored_nodes
        )

        # Pass 2
        # Re-extract from the masked PDF; fall back to original if masking was
        # skipped (no accepted anchors found, or masking produced zero regions).
        extraction_source = masked_path if masked_path is not None else pdf_path
        pass2_layout  = self._pass2_layout(extraction_source)
        text_els, fig_els, tbl_els = _separate_by_role(pass2_layout.elements)

        n_kept    = sum(1 for n in scored_nodes if n.keep)
        n_rejected = len(scored_nodes) - n_kept
        logger.info(
            "Pass-1 scoring: %d kept, %d rejected | "
            "Pass-2: %d text, %d figs, %d tables",
            n_kept, n_rejected, len(text_els), len(fig_els), len(tbl_els),
        )

        return TwoPassResult(
            pdf_path=pdf_path,
            page_dims=pass1_layout.page_dims,
            pass1_layout=pass1_layout.elements,
            scored_nodes=scored_nodes,
            header_anchors=header_anchors,
            masked_pdf_path=masked_path,
            pass2_layout=pass2_layout.elements,
            text_elements=text_els,
            figure_elements=fig_els,
            table_elements=tbl_els,
        )

    # Pass 1: layout

    def _pass1_layout(self, pdf_path: Path) -> LayoutResult:
        if self._extractor1 is None:
            self._extractor1 = DoclingLayoutExtractor(
                config=self._docling_cfg,
                cache_dir=self._cache_dir,
                max_caption_chars_per_pt=self._cfg.max_chars_per_bbox_pt,
            )
        logger.info("Pass 1 — Docling layout extraction")
        return self._extractor1.extract(pdf_path)

    # Pass 1: evidence + scoring

    def _score_nodes(
        self,
        pdf_path: Path,
        layout: LayoutResult,
    ) -> List[ScoredNode]:
        """
        Gather PyMuPDF evidence for every Pass-1 element, then apply scoring
        rules.  Returns a list parallel to ``layout.elements``.
        """
        elements = layout.elements
        logger.info(
            "Pass 1 — gathering PyMuPDF evidence for %d elements", len(elements)
        )

        evidences: List[TextNodeEvidence] = []
        with PyMuPDFEvidenceGatherer(pdf_path, self._cfg) as gatherer:
            for el in elements:
                page_h = layout.page_dims.get(el.page, {}).get("height", 842.0)
                ev = gatherer.gather(el, page_h)
                evidences.append(ev)

        scorer = NodeScorer(self._cfg, layout.page_dims)
        scored = scorer.score_all(elements, evidences)

        n_rejected = sum(1 for s in scored if not s.keep)
        logger.info("  Scored %d nodes → %d rejected", len(scored), n_rejected)
        for s in scored:
            if not s.keep:
                logger.debug(
                    "  REJECTED  page=%d  type=%-20s  text=%.50r  reason=%s",
                    s.element.page, s.element.type,
                    (s.element.text or "")[:50], s.rejection_reason,
                )
        return scored

    # Pass 1: body-extent detection

    def _find_body_extents(
        self,
        scored_nodes: List[ScoredNode],
        page_dims: Dict[int, Dict[str, float]],
    ) -> Dict[int, dict]:
        """
        For each page, find the bounding extent of all accepted body-text
        elements in fitz screen coordinates.

        The returned dict maps page_no to:
            top_y    — fitz y0 of the top-most accepted node
            bottom_y — fitz y1 of the bottom-most accepted node
            top_el   — the LayoutElement at top_y (used to build HeaderAnchor)
            x0s      — sorted list of fitz x0 values (left edges)
            x1s      — sorted list of fitz x1 values (right edges)

        Only nodes with at least ``min_anchor_word_count`` words are included
        so that short single-line running headers that passed scoring do not
        pollute the extent measurement.
        """
        extents: Dict[int, dict] = {}

        for node in scored_nodes:
            if not node.keep:
                continue
            el = node.element
            if el.type not in _ANCHOR_ELIGIBLE:
                continue
            if (
                el.type != "SECTION_HEADER"
                and len((el.text or "").split()) < self._cfg.min_anchor_word_count
            ):
                continue

            page_h    = page_dims.get(el.page, {}).get("height", 842.0)
            fitz_rect = el.bbox.to_fitz_rect(page_h)

            if el.page not in extents:
                extents[el.page] = {
                    "top_y":    fitz_rect.y0,
                    "bottom_y": fitz_rect.y1,
                    "top_el":   el,
                    "x0s":      [fitz_rect.x0],
                    "x1s":      [fitz_rect.x1],
                }
            else:
                ext = extents[el.page]
                if fitz_rect.y0 < ext["top_y"]:
                    ext["top_y"]  = fitz_rect.y0
                    ext["top_el"] = el
                if fitz_rect.y1 > ext["bottom_y"]:
                    ext["bottom_y"] = fitz_rect.y1
                ext["x0s"].append(fitz_rect.x0)
                ext["x1s"].append(fitz_rect.x1)

        for page_no, ext in extents.items():
            ext["x0s"].sort()
            ext["x1s"].sort()
            logger.debug(
                "  Body extent  page=%d  top_y=%.1f  bottom_y=%.1f  "
                "x=[%.1f, %.1f]  n=%d",
                page_no, ext["top_y"], ext["bottom_y"],
                ext["x0s"][0], ext["x1s"][-1], len(ext["x0s"]),
            )

        return extents

    # Pass 1: masking

    def _apply_masks(
        self,
        pdf_path: Path,
        page_dims: Dict[int, Dict[str, float]],
        body_extents: Dict[int, dict],
        scored_nodes: List[ScoredNode],
    ) -> Optional[Path]:
        """
        Build all mask regions and write a redacted PDF.

        Four categories of regions are masked:

        1. Header strip  — full-width horizontal band from the top of the page
                           down to just above the topmost accepted body node.
        2. Footer strip  — full-width horizontal band from just below the
                           bottommost accepted body node to the bottom of the page.
        3. Sidebar strips — narrow vertical bands at the left and/or right page
                           edges, detected by finding significant gaps in the
                           x-distribution of accepted body nodes (same x-gap
                           analysis used by PyMuPDFRegionMasker).
        4. Rejected nodes — individual bboxes of every TEXT/LIST_ITEM/PARAGRAPH
                           element that the scorer rejected (invisible-layer ghost
                           text anywhere on the page, not just in the margin zones).

        Categories 1–3 catch real-ink content (page numbers, running titles,
        journal/DOI footers, annotation side-columns) that passes the scorer
        because it has genuine ink but is not body text.
        Category 4 catches ghost text regardless of position.
        """
        mask_bboxes: List[BoundingBox] = []
        margin = self._cfg.header_mask_margin_pt
        n_header = n_footer = n_sidebar = n_nodes = 0

        for page_no, ext in body_extents.items():
            page_info = page_dims.get(page_no, {})
            page_h    = page_info.get("height", 842.0)
            page_w    = page_info.get("width",  595.0)

            # 1. Header strip
            fitz_header_bottom = max(0.0, ext["top_y"] - margin)
            # Docling coords: y1=page_h (top of page), y2=page_h-fitz_header_bottom
            docling_y1 = page_h
            docling_y2 = page_h - fitz_header_bottom
            if docling_y1 > docling_y2 and fitz_header_bottom > 0:
                mask_bboxes.append(BoundingBox(
                    x1=0.0, y1=docling_y1, x2=page_w, y2=docling_y2, page=page_no,
                ))
                n_header += 1
                logger.debug(
                    "  Header strip  page=%d  fitz(0 → %.1f)", page_no, fitz_header_bottom
                )

            # 2. Footer strip
            fitz_footer_top = min(page_h, ext["bottom_y"] + margin)
            # Docling coords: y1=page_h-fitz_footer_top (top of footer), y2=0
            docling_y1_f = page_h - fitz_footer_top
            docling_y2_f = 0.0
            if docling_y1_f > docling_y2_f and fitz_footer_top < page_h:
                mask_bboxes.append(BoundingBox(
                    x1=0.0, y1=docling_y1_f, x2=page_w, y2=docling_y2_f, page=page_no,
                ))
                n_footer += 1
                logger.debug(
                    "  Footer strip  page=%d  fitz(%.1f → %.1f)",
                    page_no, fitz_footer_top, page_h,
                )

            # 3. Sidebar elements
            # Identical logic to region_masker.py Category 1, but the x-gap
            # analysis runs on accepted body nodes rather than all elements, so
            # ghost text cannot corrupt the column-boundary estimate.
            #
            # An element is a sidebar candidate when:
            #   (a) it is narrow  — width < _SIDEBAR_MAX_W (same guard as region_masker)
            #   (b) it lies outside the main body column detected by x-gap analysis
            #
            # We mask individual element bboxes (not full-height strips) so that
            # wide body paragraphs whose left edge happens to be near a sidebar
            # boundary are never clipped.  This matches region_masker.py exactly.
            x0s = ext["x0s"]
            left_bound: Optional[float]  = None
            right_bound: Optional[float] = None
            if len(x0s) >= 2:
                gaps = [
                    (x0s[i + 1] - x0s[i], x0s[i], x0s[i + 1])
                    for i in range(len(x0s) - 1)
                    if x0s[i + 1] - x0s[i] > _COLUMN_GAP_MIN
                ]
                if gaps:
                    left_bound  = gaps[0][2]    # right edge of first gap = body start
                    right_bound = gaps[-1][1]   # left edge of last gap  = body end
            # Store on extent so the cross-page sidebar pass (3b) can read them
            ext["left_bound"]  = left_bound
            ext["right_bound"] = right_bound

        # 3b. Sidebar elements (element-level, cross-page)
        # Build a per-page sidebar predicate from the bounds computed above,
        # then walk all scored nodes (accepted AND rejected) and mask those
        # whose bbox is narrow and outside the body column.
        # Rejected nodes are already caught by category 4; we include them
        # here too but _draw_masks merges overlapping rects so there is no cost.
        #
        # SECTION_HEADER and TITLE are always excluded: section titles can be
        # narrow (e.g. "4. Conclusions" = 67 pt wide) yet belong firmly in the
        # body column.  Masking them would destroy the document's structure.
        _SIDEBAR_PROTECTED = frozenset({"SECTION_HEADER", "TITLE"})
        for node in scored_nodes:
            el = node.element
            if el.type in _SIDEBAR_PROTECTED:
                continue
            if el.page not in body_extents:
                continue
            ext = body_extents[el.page]
            lb  = ext.get("left_bound")
            rb  = ext.get("right_bound")
            if lb is None and rb is None:
                continue
            page_h    = page_dims.get(el.page, {}).get("height", 842.0)
            fitz_rect = el.bbox.to_fitz_rect(page_h)
            w = fitz_rect.x1 - fitz_rect.x0
            if w >= _SIDEBAR_MAX_W:
                continue  # wide element — cannot be a sidebar
            is_left_sidebar  = lb is not None and fitz_rect.x1 < lb
            is_right_sidebar = rb is not None and fitz_rect.x0 > rb
            if is_left_sidebar or is_right_sidebar:
                mask_bboxes.append(el.bbox)
                n_sidebar += 1

        # 4. Individual rejected node bboxes
        for node in scored_nodes:
            if node.keep or node.element.type not in _MASKABLE_TYPES:
                continue
            mask_bboxes.append(node.element.bbox)
            n_nodes += 1

        # 5. Figure, table, and structural-chrome regions
        # Mirrors Step 3 of the standard pipeline: mask figure/table bboxes so
        # that Pass-2 Docling does not return their interior text as body TEXT
        # elements.  Controlled by TwoPassConfig.mask_figures / mask_tables.
        # PAGE_HEADER, PAGE_FOOTER, and CAPTION are always masked — they are
        # structural chrome that must never appear in body text, and masking
        # them here prevents Pass-2 Docling from reclassifying them as TEXT
        # once surrounding content has been redacted.
        n_figures = n_tables = n_chrome = 0
        for node in scored_nodes:
            el = node.element
            if el.type in _ALWAYS_MASK:
                mask_bboxes.append(el.bbox)
                n_chrome += 1
            elif self._cfg.mask_figures and el.type in _FIGURE_TYPES:
                mask_bboxes.append(el.bbox)
                n_figures += 1
            elif self._cfg.mask_tables and el.type in _TABLE_TYPES:
                mask_bboxes.append(el.bbox)
                n_tables += 1

        logger.info(
            "Mask regions: %d header, %d footer, %d sidebar, "
            "%d rejected node(s), %d figure(s), %d table(s), %d chrome",
            n_header, n_footer, n_sidebar, n_nodes, n_figures, n_tables, n_chrome,
        )

        if not mask_bboxes:
            logger.info("No mask regions computed — masking skipped")
            return None

        return self._draw_masks(pdf_path, mask_bboxes)

    def _draw_masks(self, pdf_path: Path, bboxes: List[BoundingBox]) -> Optional[Path]:
        """
        Redact regions using PyMuPDF's redaction API so that the underlying
        text is removed from the PDF content stream, not merely painted over.

        draw_rect() only adds a white rectangle on the graphics layer — the
        text remains selectable and extractable by Docling/fitz in Pass 2.
        add_redact_annot() + apply_redactions() physically strips the text
        from the page before saving.

        The output filename is ``{stem}_header_masked.pdf`` to avoid collision
        with the main pipeline's ``{stem}_masked.pdf``.
        """
        try:
            import fitz  # type: ignore
            from nlp_histo.parsers.layout_utils import merge_rects

            out_dir = self._masked_pdf_dir or self._cache_dir or pdf_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{pdf_path.stem}{_MASKED_SUFFIX}"

            doc = fitz.open(str(pdf_path))

            # Group by page
            by_page: Dict[int, List[BoundingBox]] = {}
            for bbox in bboxes:
                by_page.setdefault(bbox.page, []).append(bbox)

            pages_affected = 0
            for page_num in range(len(doc)):
                page_no = page_num + 1
                if page_no not in by_page:
                    continue
                page   = doc[page_num]
                page_h = page.rect.height

                rects = [b.to_fitz_rect(page_h) for b in by_page[page_no]]
                rects = merge_rects([r for r in rects if not r.is_empty])

                for rect in rects:
                    # add_redact_annot marks the region; fill=(1,1,1) draws
                    # a white rectangle in place of the removed content.
                    page.add_redact_annot(rect, fill=(1.0, 1.0, 1.0))

                # apply_redactions() physically removes text (and optionally
                # images) from the content stream within every marked region.
                # images=fitz.PDF_REDACT_IMAGE_NONE preserves image pixels so
                # figure crops from the original PDF are unaffected.
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                pages_affected += 1

            doc.save(str(out_path))
            doc.close()

            logger.info(
                "Header-masked PDF → %s  (%d page(s) masked)",
                out_path.name, pages_affected,
            )
            return out_path

        except ImportError:
            logger.error(
                "PyMuPDF (fitz) not available — cannot write masked PDF; "
                "Pass 2 will use the original (unmasked) PDF"
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to write masked PDF for %s: %s",
                pdf_path.name, exc, exc_info=True,
            )
            return None

    # Pass 2: re-extraction

    def _pass2_layout(self, pdf_path: Path) -> LayoutResult:
        if self._extractor2 is None:
            self._extractor2 = DoclingLayoutExtractor(
                config=self._docling_text_cfg,
                cache_dir=self._cache_dir,
                max_caption_chars_per_pt=self._cfg.max_chars_per_bbox_pt,
            )
        logger.info("Pass 2 — Docling re-extraction from %s", pdf_path.name)
        return self._extractor2.extract(pdf_path)


# Module-level helper

def _separate_by_role(
    elements: List[LayoutElement],
) -> Tuple[List[LayoutElement], List[LayoutElement], List[LayoutElement]]:
    """
    Split Pass-2 elements into (text_elements, figure_elements, table_elements).

    Contract:
      • figure_elements: PICTURE / FIGURE only.  Inner text (e.g., axis labels)
        is intentionally excluded from body-text assembly.
      • table_elements: TABLE / RECONSTRUCTED_TABLE.  Cell text must not be
        flattened into paragraphs.
      • text_elements: everything else (TEXT, LIST_ITEM, SECTION_HEADER,
        CAPTION, FOOTNOTE, …).  These flow into hierarchical text assembly.
    """
    text_els:   List[LayoutElement] = []
    figure_els: List[LayoutElement] = []
    table_els:  List[LayoutElement] = []

    for el in elements:
        if el.type in _FIGURE_TYPES:
            figure_els.append(el)
        elif el.type in _TABLE_TYPES:
            table_els.append(el)
        else:
            text_els.append(el)

    return text_els, figure_els, table_els
