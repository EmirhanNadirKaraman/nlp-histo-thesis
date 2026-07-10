"""
NodeScorer

Applies keep/drop rules to Docling LayoutElements using pre-gathered
TextNodeEvidence.  This module contains only *decision logic* — no I/O,
no rendering, no fitz calls.

Decision table
--------------
Element type                Rule        Outcome
────────────────────────    ────────    ──────────────────────────────────────
FIGURE / PICTURE            always      KEEP as figure media
TABLE / RECONSTRUCTED_TABLE always      KEEP as table media
SECTION_HEADER / TITLE      always      KEEP (section structure)
CAPTION / FOOTNOTE          always      KEEP (label context)
LIST                        always      KEEP (outer container; items scored)

TEXT / LIST_ITEM / PARAGRAPH:
  R0      text is empty after strip         → REJECT "empty text"
  R-color invisible_char_fraction >
          max_white_char_fraction           → REJECT "white-text ghost layer"
          Catches white-on-white text (common publisher accessibility trick).
          Applied early because color is a direct signal — more reliable than
          pixel brightness when ghost overlaps real content.
  R1      visually_blank=True (pixel check) → REJECT "invisible text layer (blank pixels)"
  R2      fitz_word_count=0 AND char_coverage
          below threshold AND len(text)≥N
          AND render_skipped=True           → REJECT "no fitz words (no pixel data)"
          (R2 is a fallback when rendering is unavailable; prefer R1 when pixel
           data exists because fitz word lists also include invisible-layer words)
  R3      len(text) / bbox_height >
          max_chars_per_bbox_pt (when > 0) → REJECT "hidden text layer (bbox too short
                                              for text length)"
          Applied before pixel checks so it catches layers where the pixel crop
          happens to contain nearby dark pixels (e.g. the element sits inside a
          figure region).
  R4      in_header_zone AND (R-color OR R1 OR R2 OR R3) → rejection_reason amended with location hint
  Otherwise                                → KEEP

Why R2 is gated on render_skipped
----------------------------------
fitz.get_text("words") reads the PDF text stream — exactly the same source
that creates the fake words in invisible-layer PDFs.  If we relied solely
on "fitz_word_count == 0" we would never catch invisible-layer nodes because
fitz and Docling both find the same fake words.  The pixel-brightness check
(R1) is the primary discriminator; R2 is only for environments where numpy
or rendering is unavailable.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from pipeline.stages.pdf_text_extraction.config import TwoPassConfig
from pipeline.stages.pdf_text_extraction.models.dto import LayoutElement
from pipeline.stages.pdf_text_extraction.models.scored_node import ScoredNode, TextNodeEvidence

logger = logging.getLogger(__name__)

# ── Type classification ───────────────────────────────────────────────────────

_ALWAYS_KEEP = frozenset({
    "FIGURE", "PICTURE",
    "TABLE", "RECONSTRUCTED_TABLE",
    "SECTION_HEADER", "TITLE",
    "CAPTION", "FOOTNOTE",
    "LIST",                     # outer list container; items are scored individually
    "PAGE_HEADER", "PAGE_FOOTER",  # already structural — masker handles them
})

_SCORE_TYPES = frozenset({
    "TEXT",
    "LIST_ITEM",
    "PARAGRAPH",
})


class NodeScorer:
    """
    Apply keep/drop rules to (LayoutElement, TextNodeEvidence) pairs.

    Parameters
    ----------
    config:
        TwoPassConfig controlling scoring thresholds.
    page_dims:
        {page_no: {'width': …, 'height': …}} from LayoutResult.
        Used to determine whether an element is in the header zone.
    """

    def __init__(self, config: TwoPassConfig, page_dims: Dict[int, Dict[str, float]]) -> None:
        self._cfg = config
        self._page_dims = page_dims

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        element: LayoutElement,
        evidence: TextNodeEvidence,
    ) -> ScoredNode:
        """
        Score one element.

        Args:
            element:  Docling LayoutElement.
            evidence: Evidence gathered by PyMuPDFEvidenceGatherer.

        Returns:
            ScoredNode with keep=True/False and rejection_reason set.
        """
        etype = element.type

        if etype in _ALWAYS_KEEP:
            return ScoredNode(element=element, evidence=evidence, keep=True)

        if etype in _SCORE_TYPES:
            return self._score_text_node(element, evidence)

        # Unknown types: keep with a debug note so new Docling label strings
        # don't silently cause data loss.
        logger.debug(
            "NodeScorer: unknown element type '%s' (page=%d) — keeping by default",
            etype, element.page,
        )
        return ScoredNode(element=element, evidence=evidence, keep=True)

    def score_all(
        self,
        elements: List[LayoutElement],
        evidences: List[TextNodeEvidence],
    ) -> List[ScoredNode]:
        """
        Score a parallel list of elements and evidences.

        Args:
            elements:  Docling LayoutElements (same order as layout.elements).
            evidences: Corresponding TextNodeEvidence from the gatherer.

        Returns:
            List of ScoredNode in the same order as the inputs.

        Raises:
            ValueError: if the two lists have different lengths.
        """
        if len(elements) != len(evidences):
            raise ValueError(
                f"elements length ({len(elements)}) != "
                f"evidences length ({len(evidences)})"
            )
        return [self.score(el, ev) for el, ev in zip(elements, evidences)]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_text_node(
        self,
        element: LayoutElement,
        evidence: TextNodeEvidence,
    ) -> ScoredNode:
        """Apply R0–R3 to a TEXT / LIST_ITEM / PARAGRAPH element."""
        text = (element.text or "").strip()

        # R0: reject empty text regardless of evidence
        if not text:
            return ScoredNode(
                element=element, evidence=evidence,
                keep=False, rejection_reason="empty text",
            )

        in_header_zone = self._in_header_zone(element)
        location_hint  = " in header zone" if in_header_zone else ""

        # R-color: white-text ghost layer — direct color signal.
        #   invisible_char_fraction > 0 only when _gather_span_colors found white
        #   spans; it stays at 0.0 when no spans were found or the check failed
        #   (safe default: do not reject).
        if (
            self._cfg.max_white_char_fraction < 1.0
            and evidence.invisible_char_fraction > self._cfg.max_white_char_fraction
        ):
            reason = (
                f"white-text ghost layer — "
                f"{evidence.invisible_char_fraction:.0%} of characters are near-white"
                f"{location_hint}"
            )
            logger.debug(
                "  REJECT[R-color] page=%d type=%s text=%.50r  "
                "white_frac=%.3f",
                element.page, element.type, text,
                evidence.invisible_char_fraction,
            )
            return ScoredNode(
                element=element, evidence=evidence,
                keep=False, rejection_reason=reason,
            )

        # R3: dense-text heuristic — hidden text layers pack many characters
        #     into a suspiciously short bbox (e.g. 180 chars in 10pt height).
        #     Applied before pixel checks because the element may sit inside a
        #     figure region where the pixel crop captures dark figure pixels,
        #     causing R1 to miss it.
        if self._cfg.max_chars_per_bbox_pt > 0:
            bbox_height = abs(element.bbox.y1 - element.bbox.y2)
            if bbox_height > 0 and len(text) / bbox_height > self._cfg.max_chars_per_bbox_pt:
                reason = (
                    f"hidden text layer — {len(text)} chars in "
                    f"{bbox_height:.1f}pt bbox "
                    f"({len(text)/bbox_height:.1f} chars/pt){location_hint}"
                )
                logger.debug(
                    "  REJECT[R3] page=%d type=%s text=%.50r  "
                    "chars=%d  bbox_h=%.1f  ratio=%.1f",
                    element.page, element.type, text,
                    len(text), bbox_height, len(text) / bbox_height,
                )
                return ScoredNode(
                    element=element, evidence=evidence,
                    keep=False, rejection_reason=reason,
                )

        # R1: pixel-based check — blank region with non-empty Docling text
        #     Only applied when rendering succeeded (render_skipped=False).
        if not evidence.render_skipped and evidence.visually_blank:
            reason = f"visually blank region — invisible text layer{location_hint}"
            logger.debug(
                "  REJECT[R1] page=%d type=%s text=%.50r  "
                "brightness=%.1f  dark_frac=%.4f",
                element.page, element.type, text,
                evidence.pixel_brightness_mean, evidence.dark_pixel_fraction,
            )
            return ScoredNode(
                element=element, evidence=evidence,
                keep=False, rejection_reason=reason,
            )

        # R2: fitz word-count fallback — only used when rendering was unavailable.
        # (R3 dense-text check runs above; R4 location hint is applied inside R1/R2/R3 messages)
        #     We do NOT apply this when render_skipped=False because fitz also
        #     reads invisible-layer words; absence of fitz words with pixel data
        #     available would be contradictory and likely a bbox mismatch.
        if (
            evidence.render_skipped
            and evidence.fitz_word_count == 0
            and evidence.char_coverage < self._cfg.min_char_coverage_threshold
            and len(text) >= self._cfg.min_text_chars_for_word_check
        ):
            reason = (
                f"no fitz words in bbox (no pixel data available){location_hint}"
            )
            logger.debug(
                "  REJECT[R2] page=%d type=%s text=%.50r  "
                "fitz_words=%d  char_coverage=%.3f",
                element.page, element.type, text,
                evidence.fitz_word_count, evidence.char_coverage,
            )
            return ScoredNode(
                element=element, evidence=evidence,
                keep=False, rejection_reason=reason,
            )

        return ScoredNode(element=element, evidence=evidence, keep=True)

    def _in_header_zone(self, element: LayoutElement) -> bool:
        """
        Return True when the element's top edge is within the uppermost
        ``max_top_fraction_header`` fraction of the page.

        Docling coordinates: y = 0 at bottom, y1 > y2.
        "Top edge" = max(y1, y2), which maps to fitz y0 = page_h - top_edge.
        fraction_from_top = (page_h - top_edge) / page_h = fitz_y0 / page_h.
        """
        page_h = self._page_dims.get(element.page, {}).get("height", 842.0)
        if page_h <= 0:
            return False
        top_edge_docling = max(element.bbox.y1, element.bbox.y2)
        fraction_from_top = (page_h - top_edge_docling) / page_h
        return fraction_from_top <= self._cfg.max_top_fraction_header
