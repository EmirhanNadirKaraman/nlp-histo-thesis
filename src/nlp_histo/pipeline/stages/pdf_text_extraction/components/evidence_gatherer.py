"""
PyMuPDFEvidenceGatherer

Attaches supporting evidence from PyMuPDF (fitz) to Docling LayoutElements.
PyMuPDF is a support role here — it validates Docling nodes but never
contributes layout nodes of its own to the final output.

Three categories of evidence are produced for every element:

  1. Word / block overlap
     page.get_text("words") → fitz word tuples whose centroids land inside
     the Docling element's bounding box.  High overlap = real text.

  2. Pixel brightness
     The PDF page is rendered at ``render_dpi`` DPI, clipped to the element
     bbox, and the mean/std/dark-fraction of pixel luminance is computed.
     A visually blank region (high brightness, very few dark pixels) indicates
     an invisible / selectable-but-not-rendered text layer.

  3. Span color
     page.get_text("dict") → text spans overlapping the bbox are inspected
     for near-white RGB color, yielding ``invisible_char_fraction`` (used by
     NodeScorer's white-text ghost-layer rule).

All fitz word/block lists are cached per page for the lifetime of the gatherer
so that a single page is only parsed once even when many elements live on it.

Usage
-----
    gatherer = PyMuPDFEvidenceGatherer(pdf_path, config)
    with gatherer:
        for el in layout.elements:
            page_h = layout.page_dims[el.page]["height"]
            ev = gatherer.gather(el, page_h)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from nlp_histo.pipeline.stages.pdf_text_extraction.config import TwoPassConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import LayoutElement
from nlp_histo.pipeline.stages.pdf_text_extraction.models.scored_node import TextNodeEvidence

logger = logging.getLogger(__name__)

# fitz word / block tuple field indices
# page.get_text("words") → (x0, y0, x1, y1, word, block_no, line_no, word_no)
_W_X0, _W_Y0, _W_X1, _W_Y1, _W_TEXT = 0, 1, 2, 3, 4

# page.get_text("blocks") → (x0, y0, x1, y1, text, block_no, block_type)
_B_X0, _B_Y0, _B_X1, _B_Y1, _B_TYPE = 0, 1, 2, 3, 6
_BLOCK_TYPE_TEXT = 0  # block_type == 0 means text (not image)

# Luminance threshold below which a pixel is counted as "ink"
_INK_LUMINANCE_THRESHOLD = 200  # 0-255

# Per-channel threshold for "near-white" text color (packed RGB int).
# Characters whose R, G, and B are all >= this value are counted as
# white-text ghost layer candidates.
_WHITE_COLOR_THRESHOLD = 240  # per channel, 0-255


def _is_near_white(color: int) -> bool:
    """Return True when the packed-RGB color integer is near-white."""
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF
    return r >= _WHITE_COLOR_THRESHOLD and g >= _WHITE_COLOR_THRESHOLD and b >= _WHITE_COLOR_THRESHOLD


def _rects_overlap(ax0: float, ay0: float, ax1: float, ay1: float,
                   bx0: float, by0: float, bx1: float, by1: float) -> bool:
    """Return True when two axis-aligned rectangles overlap (open boundary)."""
    return not (ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1)


class PyMuPDFEvidenceGatherer:
    """
    Per-PDF evidence gatherer.

    Parameters
    ----------
    pdf_path:
        Path to the PDF.  The fitz Document is opened lazily on the first
        ``gather()`` call and held open until ``close()`` is called.
    config:
        TwoPassConfig holding render DPI and brightness thresholds.
    """

    def __init__(self, pdf_path: Path, config: TwoPassConfig) -> None:
        self._pdf_path = pdf_path
        self._cfg = config
        self._doc: Optional[Any] = None          # fitz.Document
        self._word_cache: Dict[int, list] = {}   # page_no (1-indexed) → list of word tuples
        self._block_cache: Dict[int, list] = {}  # page_no → list of block tuples
        self._dict_cache: Dict[int, dict] = {}   # page_no → get_text("dict") result

    # Context manager

    def __enter__(self) -> "PyMuPDFEvidenceGatherer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying fitz.Document and clear page caches."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None
        self._word_cache.clear()
        self._block_cache.clear()
        self._dict_cache.clear()

    # Public API

    def gather(self, element: LayoutElement, page_height: float) -> TextNodeEvidence:
        """
        Gather word-overlap and pixel-brightness evidence for one element.

        Args:
            element:     Docling LayoutElement with bbox in Docling PDF coords.
            page_height: Height of the page in pts (from LayoutResult.page_dims).

        Returns:
            Populated TextNodeEvidence.  On any non-fatal error the affected
            sub-fields remain at their default sentinel values and
            ``render_skipped`` is set to True.
        """
        ev = TextNodeEvidence()

        # Convert Docling bbox → fitz screen rect (y = 0 at top)
        fitz_rect = element.bbox.to_fitz_rect(page_height)
        if fitz_rect.is_empty or fitz_rect.width <= 0 or fitz_rect.height <= 0:
            ev.render_skipped = True
            return ev

        doc = self._open_doc()
        if doc is None:
            ev.render_skipped = True
            return ev

        page_no = element.page  # 1-indexed (Docling convention)
        if page_no < 1 or page_no > len(doc):
            logger.warning(
                "Evidence gatherer: page %d out of range (doc has %d pages) in %s",
                page_no, len(doc), self._pdf_path.name,
            )
            ev.render_skipped = True
            return ev

        fitz_page = doc[page_no - 1]  # fitz is 0-indexed

        # Gather in independent passes; errors in one don't abort the others
        self._gather_word_overlap(ev, element, fitz_page, fitz_rect, page_no)
        self._gather_pixel_brightness(ev, fitz_page, fitz_rect)
        self._gather_span_colors(ev, fitz_page, fitz_rect, page_no)

        return ev

    # Internal: fitz document

    def _open_doc(self) -> Optional[Any]:
        """Lazily open fitz.Document; return None on import / open failure."""
        if self._doc is not None:
            return self._doc
        try:
            import fitz  # type: ignore
            self._doc = fitz.open(str(self._pdf_path))
            return self._doc
        except ImportError:
            logger.warning("PyMuPDF (fitz) not available — evidence gathering disabled")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to open PDF with fitz (%s): %s", self._pdf_path.name, exc)
            return None

    # Internal: word / block overlap

    def _words_for_page(self, fitz_page: Any, page_no: int) -> list:
        if page_no not in self._word_cache:
            self._word_cache[page_no] = fitz_page.get_text("words")
        return self._word_cache[page_no]

    def _blocks_for_page(self, fitz_page: Any, page_no: int) -> list:
        if page_no not in self._block_cache:
            self._block_cache[page_no] = fitz_page.get_text("blocks")
        return self._block_cache[page_no]

    def _dict_for_page(self, fitz_page: Any, page_no: int) -> dict:
        if page_no not in self._dict_cache:
            self._dict_cache[page_no] = fitz_page.get_text("dict")
        return self._dict_cache[page_no]

    def _gather_word_overlap(
        self,
        ev: TextNodeEvidence,
        element: LayoutElement,
        fitz_page: Any,
        fitz_rect: Any,
        page_no: int,
    ) -> None:
        """
        Fill fitz_word_count, fitz_word_chars, fitz_block_present, char_coverage.

        Word membership test: the word's centroid must lie inside ``fitz_rect``.
        Centroid testing is more robust than bbox intersection for words at the
        boundary of an element.
        """
        try:
            fx0, fy0, fx1, fy1 = fitz_rect.x0, fitz_rect.y0, fitz_rect.x1, fitz_rect.y1

            # Words
            words = self._words_for_page(fitz_page, page_no)
            word_count = 0
            total_chars = 0
            for w in words:
                wx0, wy0, wx1, wy1 = w[_W_X0], w[_W_Y0], w[_W_X1], w[_W_Y1]
                wtext: str = w[_W_TEXT]
                # Centroid check
                wcx = (wx0 + wx1) / 2.0
                wcy = (wy0 + wy1) / 2.0
                if fx0 <= wcx <= fx1 and fy0 <= wcy <= fy1:
                    word_count += 1
                    total_chars += len(wtext)

            ev.fitz_word_count = word_count
            ev.fitz_word_chars = total_chars

            docling_text = (element.text or "").strip()
            if docling_text:
                ev.char_coverage = min(total_chars / len(docling_text), 1.0)

            # Blocks
            blocks = self._blocks_for_page(fitz_page, page_no)
            for blk in blocks:
                if blk[_B_TYPE] != _BLOCK_TYPE_TEXT:
                    continue
                bx0, by0, bx1, by1 = blk[_B_X0], blk[_B_Y0], blk[_B_X1], blk[_B_Y1]
                if _rects_overlap(fx0, fy0, fx1, fy1, bx0, by0, bx1, by1):
                    ev.fitz_block_present = True
                    break  # one is enough

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Word overlap check failed for element on page %d: %s",
                element.page, exc,
            )

    # Internal: span color (white-text ghost layer)

    def _gather_span_colors(
        self,
        ev: TextNodeEvidence,
        fitz_page: Any,
        fitz_rect: Any,
        page_no: int,
    ) -> None:
        """
        Fill ``invisible_char_fraction`` by inspecting text colors in overlapping
        spans via ``page.get_text("dict")``.

        Characters whose span color is near-white (all RGB channels >= 240) are
        counted as invisible.  This catches "white text on white background"
        ghost layers — a common publisher technique for embedding accessible /
        searchable text that is not meant to be visible.

        Note: rendering-mode-3 (PDF Tr=3) text is not detectable through
        PyMuPDF's ``get_text`` API regardless of color; it requires content-stream
        parsing.  This check covers only the color-based variant.
        """
        try:
            page_dict = self._dict_for_page(fitz_page, page_no)
            fx0, fy0, fx1, fy1 = fitz_rect.x0, fitz_rect.y0, fitz_rect.x1, fitz_rect.y1

            total_chars = 0
            white_chars = 0

            for block in page_dict.get("blocks", []):
                if block.get("type") != _BLOCK_TYPE_TEXT:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sb = span.get("bbox")
                        if sb is None:
                            continue
                        if not _rects_overlap(fx0, fy0, fx1, fy1, sb[0], sb[1], sb[2], sb[3]):
                            continue
                        span_text = span.get("text", "")
                        n = len(span_text)
                        if n == 0:
                            continue
                        total_chars += n
                        color = span.get("color", -1)
                        if color >= 0 and _is_near_white(color):
                            white_chars += n

            if total_chars > 0:
                ev.invisible_char_fraction = white_chars / total_chars

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Span color check failed for element on page %d: %s",
                page_no, exc,
            )

    # Internal: pixel brightness

    def _gather_pixel_brightness(
        self,
        ev: TextNodeEvidence,
        fitz_page: Any,
        fitz_rect: Any,
    ) -> None:
        """
        Fill pixel_brightness_mean, pixel_brightness_std, dark_pixel_fraction,
        and visually_blank by rendering the clipped region.

        Requires numpy.  Sets render_skipped=True if numpy or the render fails.
        """
        try:
            import numpy as np  # type: ignore
            import fitz as _fitz  # type: ignore  # already imported above; re-import is free

            scale = self._cfg.render_dpi / 72.0
            mat = _fitz.Matrix(scale, scale)
            pix = fitz_page.get_pixmap(matrix=mat, clip=fitz_rect, colorspace=_fitz.csRGB)

            if pix.width == 0 or pix.height == 0:
                ev.render_skipped = True
                return

            # Reshape raw bytes into (H, W, 3) uint8 array
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            # Perceived luminance: ITU-R BT.601 coefficients
            gray = (
                0.299 * arr[:, :, 0].astype(np.float32)
                + 0.587 * arr[:, :, 1].astype(np.float32)
                + 0.114 * arr[:, :, 2].astype(np.float32)
            )

            ev.pixel_brightness_mean  = float(gray.mean())
            ev.pixel_brightness_std   = float(gray.std())
            ev.dark_pixel_fraction    = float((gray < _INK_LUMINANCE_THRESHOLD).mean())

            ev.visually_blank = (
                ev.pixel_brightness_mean >= self._cfg.blank_brightness_threshold
                and ev.dark_pixel_fraction <= self._cfg.blank_dark_pixel_max_fraction
            )

        except ImportError:
            # numpy not installed — pixel check unavailable
            ev.render_skipped = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pixel brightness check failed: %s", exc)
            ev.render_skipped = True
