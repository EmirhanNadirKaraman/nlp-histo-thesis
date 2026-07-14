"""
Data Transfer Objects for the two-pass text-cleanup pipeline.

Each Docling LayoutElement extracted in Pass 1 is wrapped in a ScoredNode
after evidence from PyMuPDF (word overlap, pixel brightness) is gathered and
a keep/drop decision is applied.

The final TwoPassResult bundles both pass outputs plus the separated
text/figure/table element lists that downstream consumers should use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import LayoutElement


# ── Evidence ──────────────────────────────────────────────────────────────────

@dataclass
class TextNodeEvidence:
    """
    Evidence gathered from PyMuPDF for a single Docling LayoutElement.

    All numeric fields default to "no evidence / not measured" sentinels so
    that callers can distinguish "gatherer returned zero" from "gatherer never
    ran".  The ``render_skipped`` flag signals when pixel data is unavailable.

    Coordinate note: all measurements are taken in fitz screen space (y = 0
    at the top of the page) after converting from Docling PDF coordinates.
    """

    # ── PyMuPDF word / block overlap ──────────────────────────────────────────
    fitz_word_count: int = 0
    """Number of fitz words whose centroids fall inside the element's bbox."""

    fitz_word_chars: int = 0
    """Total character count of those overlapping fitz words."""

    fitz_block_present: bool = False
    """True when at least one fitz text block's bbox overlaps this element."""

    char_coverage: float = 0.0
    """
    fitz_word_chars / len(docling_text), capped at 1.0.

    Near 0 → fitz found very little text in the same area → suspicious.
    Near 1 → fitz and Docling agree → real text.
    """

    # ── Pixel-based checks (rendered image) ───────────────────────────────────
    pixel_brightness_mean: float = 255.0
    """
    Mean pixel luminance (0 = black, 255 = pure white) in the rendered region.
    High value on a PDF with supposedly non-empty text → invisible text layer.
    """

    pixel_brightness_std: float = 0.0
    """
    Std-dev of pixel luminance.  Low std on a bright image confirms uniform
    whiteness (blank paper), not just an accidentally bright text region.
    """

    dark_pixel_fraction: float = 0.0
    """
    Fraction of pixels below a darkness threshold (default 200/255).
    High → ink is present → real text.  Near 0 → blank → suspicious.
    """

    visually_blank: bool = False
    """
    Composite flag: True when pixel_brightness_mean exceeds
    ``blank_brightness_threshold`` AND dark_pixel_fraction is below
    ``blank_dark_pixel_max_fraction``.  Computed by the evidence gatherer.
    """

    # ── Span-color check (white-text ghost layer) ─────────────────────────────
    invisible_char_fraction: float = 0.0
    """
    Fraction of characters in overlapping spans whose text color is near-white
    (all RGB channels >= 240), indicating a white-text ghost / accessibility
    layer.  0.0 = all chars have visible (dark) color.  1.0 = all chars are
    white / invisible.  Stays at 0.0 when the check could not run (no spans
    found in bbox, or extraction failed).
    """

    # ── Metadata ──────────────────────────────────────────────────────────────
    render_skipped: bool = False
    """
    True when pixel rendering was skipped (zero-area bbox, missing numpy,
    or a rendering error).  When True, ``visually_blank`` is unreliable —
    the scorer falls back to fitz word-overlap rules only.
    """


# ── Scored node ───────────────────────────────────────────────────────────────

@dataclass
class ScoredNode:
    """
    A Docling LayoutElement paired with gathered evidence and a keep/drop
    decision produced by NodeScorer.

    Attributes
    ----------
    element:
        The original Docling element (type, bbox, text, page, level).
    evidence:
        PyMuPDF-derived evidence.  Empty (all defaults) for element types
        that are always kept (TABLE, FIGURE, SECTION_HEADER, …).
    keep:
        True → include this node in downstream processing.
        False → reject; the node likely comes from an invisible/fake text layer.
    rejection_reason:
        Human-readable explanation when ``keep`` is False.  Empty string
        when ``keep`` is True.
    """

    element: LayoutElement
    evidence: TextNodeEvidence = field(default_factory=TextNodeEvidence)
    keep: bool = True
    rejection_reason: str = ""


# ── Header anchor ─────────────────────────────────────────────────────────────

@dataclass
class HeaderAnchor:
    """
    The top-most *accepted* body-text element on a page; used to determine
    where the header zone ends.

    ``top_y_fitz`` is the y-coordinate of the element's top edge in fitz
    screen space (y = 0 at the top of the page, increasing downward).
    """

    page: int
    element: LayoutElement
    top_y_fitz: float  # fitz screen coord — smaller = closer to top of page


# ── Two-pass result ───────────────────────────────────────────────────────────

@dataclass
class TwoPassResult:
    """
    Full output of the two-pass cleanup pipeline for one PDF.

    Typical consumers should use ``text_elements``, ``figure_elements``, and
    ``table_elements`` (all from Pass 2).  The raw per-pass fields are kept for
    debugging and offline analysis.

    Separation contract
    -------------------
    - ``text_elements``   : TEXT, LIST_ITEM, SECTION_HEADER, CAPTION, FOOTNOTE, …
                            These flow into hierarchical text assembly.
    - ``figure_elements`` : PICTURE / FIGURE only.  Inner text (axis labels, etc.)
                            must NOT be merged into body text.
    - ``table_elements``  : TABLE / RECONSTRUCTED_TABLE.  Cell text must NOT be
                            flattened into paragraph text.
    """

    pdf_path: Path
    page_dims: Dict[int, Dict[str, float]]  # {page_no: {'width': …, 'height': …}}

    # ── Pass 1 outputs ────────────────────────────────────────────────────────
    pass1_layout: List[LayoutElement]
    """All elements returned by the first (unmasked) Docling extraction."""

    scored_nodes: List[ScoredNode]
    """Pass-1 elements after evidence gathering and scoring.  Same length as pass1_layout."""

    header_anchors: Dict[int, HeaderAnchor]
    """Top-most accepted body anchor per page (page_no → HeaderAnchor)."""

    masked_pdf_path: Optional[Path]
    """Path to the header-masked PDF used for Pass 2.  None if masking was skipped."""

    # ── Pass 2 outputs ────────────────────────────────────────────────────────
    pass2_layout: List[LayoutElement]
    """All elements from the second (header-masked) Docling extraction — final layout."""

    # ── Role-separated (from pass2_layout) ───────────────────────────────────
    text_elements: List[LayoutElement]
    figure_elements: List[LayoutElement]
    table_elements: List[LayoutElement]

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def n_rejected(self) -> int:
        """Number of Pass-1 nodes rejected by the scorer."""
        return sum(1 for n in self.scored_nodes if not n.keep)

    @property
    def rejected_nodes(self) -> List[ScoredNode]:
        """Rejected nodes only — useful for inspection."""
        return [n for n in self.scored_nodes if not n.keep]
