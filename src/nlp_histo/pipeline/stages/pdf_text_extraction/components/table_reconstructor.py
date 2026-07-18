"""
TableReconstructor

Two utilities for recovering table content that Docling misses:

reconstruct_tables_from_lists
  Injects RECONSTRUCTED_TABLE elements into a LayoutResult by grouping TEXT /
  LIST_ITEM elements that follow a table CAPTION into a single synthetic element.
  Handles PDFs where Docling emits a table as a sequence of list/text rows.

  Algorithm (ported from scripts/visualize_docling_full.py):
    1. Scan elements for a CAPTION whose text contains "table".
    2. Consume following TEXT / LIST_ITEM elements while the vertical gap
       between consecutive elements stays within the adaptive threshold.
    3. The threshold is initialised at 20 pts; after the first row is captured
       it is refined to ``first_inter_row_gap * threshold_multiplier``.
    4. Union the bounding boxes into a single RECONSTRUCTED_TABLE element.

expand_tables_with_footnotes
  Expands already-detected table bboxes downward to absorb TEXT / LIST_ITEM
  elements whose top edge falls within a proximity threshold of the table's
  bottom edge.  Consumed elements are tracked globally so each element is
  absorbed by at most one table.  Expansion repeats until no new elements are
  absorbed (greedy).
"""
from __future__ import annotations

import logging
from typing import List

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, LayoutElement, LayoutResult

logger = logging.getLogger(__name__)


def reconstruct_tables_from_lists(
    layout: LayoutResult,
    threshold_multiplier: float = 1.2,
) -> LayoutResult:
    """
    Return a new LayoutResult with RECONSTRUCTED_TABLE elements injected.

    Args:
        layout:               Input LayoutResult (not modified in place).
        threshold_multiplier: Multiplier applied to the first inter-row gap to
                              set the maximum allowed vertical gap between rows.

    Returns:
        New LayoutResult with RECONSTRUCTED_TABLE elements spliced in after
        each qualifying table caption.
    """
    elements = layout.elements
    new_elements: List[LayoutElement] = []
    i = 0

    while i < len(elements):
        el = elements[i]

        if el.type == "CAPTION" and "table" in (el.text or "").lower():
            new_elements.append(el)

            sub_elements: List[LayoutElement] = []
            max_allowed_gap = 20.0
            # Docling coords: y2 is the bottom edge (smaller value)
            last_y2 = el.bbox.y2
            i += 1

            while i < len(elements):
                next_el = elements[i]
                if next_el.type not in ("TEXT", "LIST_ITEM"):
                    break

                # In Docling coords y1 > y2; gap between bottom of last element
                # and top of next element = next.y1 - last.y2
                vertical_gap = abs(next_el.bbox.y1 - last_y2)

                # After the first row, refine threshold from actual row spacing
                if len(sub_elements) == 1:
                    true_gutter = abs(next_el.bbox.y1 - sub_elements[0].bbox.y2)
                    max_allowed_gap = true_gutter * threshold_multiplier

                if vertical_gap < max_allowed_gap:
                    sub_elements.append(next_el)
                    last_y2 = next_el.bbox.y2
                    i += 1
                else:
                    break

            if sub_elements:
                page = sub_elements[0].bbox.page
                x1 = min(e.bbox.x1 for e in sub_elements)
                y1 = max(e.bbox.y1 for e in sub_elements)  # topmost (largest y)
                x2 = max(e.bbox.x2 for e in sub_elements)
                y2 = min(e.bbox.y2 for e in sub_elements)  # bottommost (smallest y)
                new_elements.append(LayoutElement(
                    type="RECONSTRUCTED_TABLE",
                    page=page,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, page=page),
                    text=el.text,
                    level=0,
                ))
                logger.debug(
                    "Reconstructed table from %d sub-elements on page %d (caption: %s)",
                    len(sub_elements), page, el.text,
                )
        else:
            new_elements.append(el)
            i += 1

    n_recon = sum(1 for e in new_elements if e.type == "RECONSTRUCTED_TABLE")
    if n_recon:
        logger.info("Table reconstruction: injected %d RECONSTRUCTED_TABLE element(s)", n_recon)

    return LayoutResult(
        elements=new_elements,
        page_dims=layout.page_dims,
        pdf_path=layout.pdf_path,
        source=layout.source,
    )


def _y_overlaps(a: dict, b: dict) -> bool:
    """True if two Docling-coord bboxes have any vertical overlap."""
    a_top = max(a["y1"], a["y2"])
    a_bot = min(a["y1"], a["y2"])
    b_top = max(b["y1"], b["y2"])
    b_bot = min(b["y1"], b["y2"])
    return a_bot < b_top and b_bot < a_top


def _expand_rotated_table(
    bbox: dict,
    page: int,
    candidates: list,
    consumed: set,
    proximity_pts: float,
    threshold_multiplier: float,
    text_proximity_pts: float = 8.0,
) -> int:
    """
    Expand a rotated table bbox laterally (x-axis) to absorb footnotes.

    For a 90°-rotated table the footnotes sit to the left or right of the
    bbox in PDF coordinates.  We scan both sides independently so that the
    correct direction is found regardless of whether the rotation is CW or CCW.
    A vertical-overlap guard ensures only elements aligned with the table are
    considered, preventing absorption of unrelated body text columns.
    """
    total = 0
    for direction in ("left", "right"):
        if direction == "left":
            nearby = sorted(
                [
                    (i, el) for i, el in candidates
                    if el.get("page") == page
                    and el["bbox"]["x2"] <= bbox["x1"]
                    and _y_overlaps(el["bbox"], bbox)
                ],
                key=lambda x: x[1]["bbox"]["x2"],
                reverse=True,  # highest x2 = closest to table's left edge
            )
            frontier = bbox["x1"]
        else:
            nearby = sorted(
                [
                    (i, el) for i, el in candidates
                    if el.get("page") == page
                    and el["bbox"]["x1"] >= bbox["x2"]
                    and _y_overlaps(el["bbox"], bbox)
                ],
                key=lambda x: x[1]["bbox"]["x1"],
            )
            frontier = bbox["x2"]

        max_gap   = proximity_pts
        first_gap = None

        for i, el in nearby:
            if i in consumed:
                continue
            eb      = el["bbox"]
            gap     = (frontier - eb["x2"]) if direction == "left" else (eb["x1"] - frontier)
            limit   = text_proximity_pts if el.get("type") == "TEXT" else max_gap
            if gap < 0 or gap > limit:
                break
            bbox["y1"] = max(bbox["y1"], eb["y1"])
            bbox["y2"] = min(bbox["y2"], eb["y2"])
            if direction == "left":
                bbox["x1"] = min(bbox["x1"], eb["x1"])
                frontier   = bbox["x1"]
            else:
                bbox["x2"] = max(bbox["x2"], eb["x2"])
                frontier   = bbox["x2"]
            consumed.add(i)
            total += 1
            logger.debug(
                "Absorbed rotated-table footnote (%s, gap=%.1f pts, max=%.1f): %s",
                direction, gap, max_gap, (el.get("text") or "")[:60],
            )
            if first_gap is None and el.get("type") != "TEXT":
                first_gap = gap
                max_gap   = first_gap * threshold_multiplier

    return total


def expand_tables_with_footnotes(
    merged_tables: dict,
    element_dicts: list,
    proximity_pts: float = 20.0,
    threshold_multiplier: float = 1.2,
    text_proximity_pts: float = 8.0,
) -> dict:
    """
    Expand table bboxes to absorb nearby TEXT/LIST_ITEM/FOOTNOTE elements.

    Upright tables expand downward (y-axis); rotated tables (TATR label
    ``"table rotated"``) expand laterally (x-axis, both left and right).

    Elements are processed in proximity order (closest edge first).
    The first absorbed gap sets an adaptive threshold (first_gap * multiplier)
    for subsequent absorptions — preventing cascade into body text paragraphs.
    Once an element is consumed it cannot be absorbed by another table.

    Args:
        merged_tables:        Dict of table entries as built by PyMuPDFMediaCropper.
        element_dicts:        Layout elements as dicts (LayoutResult.to_element_dicts()).
        proximity_pts:        Maximum initial gap (pts) between table edge and element.
        threshold_multiplier: After the first absorption, max_gap = first_gap * multiplier.

    Returns:
        The same ``merged_tables`` dict with expanded bboxes (mutated in place).
    """
    consumed: set = set()

    candidates = [
        (i, el) for i, el in enumerate(element_dicts)
        if el.get("type") in ("TEXT", "LIST_ITEM", "FOOTNOTE")
    ]

    total_absorbed = 0
    for table in merged_tables.values():
        page = table["page"]
        bbox = table["bbox"]

        if table.get("rotated"):
            total_absorbed += _expand_rotated_table(
                bbox, page, candidates, consumed, proximity_pts, threshold_multiplier,
                text_proximity_pts=text_proximity_pts,
            )
            continue

        # Upright table: expand downward (y-axis)
        # Collect elements below the table on the same page, sorted closest first
        below = sorted(
            [(i, el) for i, el in candidates
             if el.get("page") == page and el["bbox"]["y1"] < bbox["y2"]],
            key=lambda x: x[1]["bbox"]["y1"],
            reverse=True,  # highest y1 = closest to table bottom
        )

        frontier  = bbox["y2"]
        max_gap   = proximity_pts
        first_gap = None

        for i, el in below:
            if i in consumed:
                continue
            eb    = el["bbox"]
            gap   = frontier - eb["y1"]
            limit = text_proximity_pts if el.get("type") == "TEXT" else max_gap
            if gap < 0 or gap > limit:
                break  # gap too large — stop
            bbox["x1"] = min(bbox["x1"], eb["x1"])
            bbox["x2"] = max(bbox["x2"], eb["x2"])
            bbox["y2"] = min(bbox["y2"], eb["y2"])
            frontier   = bbox["y2"]
            consumed.add(i)
            total_absorbed += 1
            logger.debug(
                "Absorbed footnote into table (gap=%.1f pts, max=%.1f): %s",
                gap, limit, (el.get("text") or "")[:60],
            )
            if first_gap is None and el.get("type") != "TEXT":
                first_gap = gap
                max_gap   = first_gap * threshold_multiplier

    if total_absorbed:
        logger.info(
            "Footnote expansion: absorbed %d element(s) into %d table(s)",
            total_absorbed, len(merged_tables),
        )

    return merged_tables
