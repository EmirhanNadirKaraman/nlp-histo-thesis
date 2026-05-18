"""
PyMuPDFMediaCropper

Crops figure and table regions from a PDF using PyMuPDF, saves each crop as a
PNG image, and returns CroppedMedia metadata for both categories.

Figures: sourced from PICTURE/FIGURE Docling elements, merged by caption number.
Tables:  sourced from detection regions (TATR/hybrid) as primary, plus
         TABLE/RECONSTRUCTED_TABLE Docling elements as supplementary,
         merged by caption number.

Merging logic ported from merged_pipeline._crop_and_save:
  - Elements sharing the same caption number are unioned via union_bbox().
  - The longer caption string wins.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline.stages.pdf_text_extraction.config import CroppingConfig
from pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, CroppedMedia, LayoutResult, TableDetectionResult
from parsers.layout_utils import (
    FIG_NUM_RE,
    TAB_NUM_RE,
    nearest_caption,
    parse_caption_num,
    union_bbox,
)

logger = logging.getLogger(__name__)


class PyMuPDFMediaCropper:
    """
    Crops figure and table regions from PDFs using PyMuPDF.

    Parameters
    ----------
    config:
        CroppingConfig controlling DPI, output format, etc.
    figures_dir:
        Directory to save figure crops.
    tables_dir:
        Directory to save table crops.
    """

    def __init__(
        self,
        config: Optional[CroppingConfig] = None,
        figures_dir: Optional[Path] = None,
        tables_dir:  Optional[Path] = None,
    ) -> None:
        self._config      = config or CroppingConfig()
        self._figures_dir = figures_dir or Path("out/figures")
        self._tables_dir  = tables_dir  or Path("out/tables")

    def crop(
        self,
        pdf_path: Path,
        layout: LayoutResult,
        detection: Optional[TableDetectionResult] = None,
        docling_table_types: Tuple[str, ...] = ("TABLE", "RECONSTRUCTED_TABLE"),
    ) -> Tuple[List[CroppedMedia], List[CroppedMedia]]:
        """
        Crop and save figure and table regions.

        Args:
            pdf_path:  Path to the original (unmasked) PDF.
            layout:    Layout result containing element positions and page dims.
            detection: Table detection result (TATR/hybrid) used as primary
                       source for table crops.  Docling TABLE elements are
                       used as a supplementary source regardless.

        Returns:
            ``(figures, tables)`` — two lists of CroppedMedia.
        """
        import fitz  # type: ignore

        element_dicts = layout.to_element_dicts()
        all_captions  = [e for e in element_dicts if e.get("type") == "CAPTION"]

        doc   = fitz.open(str(pdf_path))
        scale = self._config.dpi / 72.0
        mat   = fitz.Matrix(scale, scale)

        figures: List[CroppedMedia] = []
        tables:  List[CroppedMedia] = []

        # ── Figures ───────────────────────────────────────────────────────────
        if self._config.save_figure_crops:
            merged_figures: dict = {}
            claimed_captions: dict = {}   # caption_key → num of figure that claimed it

            def _caption_key(cap_el):
                b = cap_el.get("bbox", {})
                return (b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2"))

            def _bbox_edge_gap(a, b):
                """Minimum edge-to-edge distance between two bboxes (Docling coords)."""
                a_top, a_bot = max(a["y1"], a["y2"]), min(a["y1"], a["y2"])
                b_top, b_bot = max(b["y1"], b["y2"]), min(b["y1"], b["y2"])
                vert = max(0, b_bot - a_top) if b_bot > a_top else max(0, a_bot - b_top)
                horiz = max(0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
                return max(vert, horiz)

            available_captions = list(all_captions)

            for el in element_dicts:
                if el.get("type") not in ("PICTURE", "FIGURE"):
                    continue
                b = el.get("bbox") or {}
                w = b.get("x2", 0) - b.get("x1", 0)
                h = abs(b.get("y1", 0) - b.get("y2", 0))
                if w < self._config.min_figure_pts or h < self._config.min_figure_pts:
                    logger.debug("Skipping small figure (%.0f×%.0f pts)", w, h)
                    continue

                # Find nearest available (unclaimed) caption
                cap_el  = nearest_caption(el, available_captions)
                caption = cap_el.get("text", "") if cap_el else ""
                parsed_num = parse_caption_num(caption, FIG_NUM_RE)

                # Check if this caption is already claimed by another figure
                cap_key = _caption_key(cap_el) if cap_el else None
                if cap_key and cap_key in claimed_captions:
                    # Caption already taken — check proximity to the figure that owns it
                    owner_num = claimed_captions[cap_key]
                    owner_bbox = merged_figures[owner_num]["bbox"]
                    gap = _bbox_edge_gap(b, owner_bbox) if owner_bbox else float("inf")
                    if gap <= self._config.subfigure_proximity_pts:
                        # Close enough — treat as subfigure panel, merge into owner
                        existing = merged_figures[owner_num]
                        existing["bbox"] = union_bbox(existing["bbox"], b)
                        logger.debug("Merged subfigure panel into Figure %s (gap=%.0f pts)", owner_num, gap)
                        continue
                    else:
                        # Far away — different figure, output without caption
                        cap_el  = None
                        caption = ""
                        parsed_num = None

                if self._config.merge_figures_by_caption and parsed_num is not None:
                    num = str(parsed_num)
                else:
                    num = str(len(merged_figures) + 1)

                while num in merged_figures:
                    num = f"{num}b"

                merged_figures[num] = {
                    "figure_id": parsed_num or num,
                    "caption":   caption or None,
                    "page":      el.get("page"),
                    "bbox":      el.get("bbox"),
                }
                if cap_key is not None:
                    claimed_captions[cap_key] = num
                    available_captions = [c for c in available_captions
                                          if _caption_key(c) != cap_key]

            for fig in merged_figures.values():
                page_no = fig["page"]
                b       = fig["bbox"]
                if page_no is None or b is None:
                    continue
                bbox    = BoundingBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"], page=page_no)
                num_int = int(fig["figure_id"]) if str(fig["figure_id"]).isdigit() else None
                label   = f"Figure {fig['figure_id']}"
                media   = self._crop_element(
                    doc, bbox, layout.page_dims, mat, label,
                    num_int, fig["caption"] or None, "figure", pdf_path.stem, self._figures_dir,
                )
                if media:
                    figures.append(media)

        # ── Tables ────────────────────────────────────────────────────────────
        if self._config.save_table_crops:
            merged_tables: dict = {}

            def _overlap_ratio(a: dict, b: dict) -> float:
                """Intersection / min-area for two Docling bboxes (same page assumed)."""
                ix1 = max(a["x1"], b["x1"])
                ix2 = min(a["x2"], b["x2"])
                iy1 = min(a["y1"], b["y1"])  # Docling: y1 > y2
                iy2 = max(a["y2"], b["y2"])
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy1 - iy2)
                inter = iw * ih
                if inter == 0:
                    return 0.0
                area_a = (a["x2"] - a["x1"]) * abs(a["y1"] - a["y2"])
                area_b = (b["x2"] - b["x1"]) * abs(b["y1"] - b["y2"])
                denom = min(area_a, area_b)
                return inter / denom if denom > 0 else 0.0

            # Primary source: detection regions (TATR / hybrid)
            if detection:
                tatr_source = detection.source  # 'tatr' | 'hybrid' | …
                for region in detection.regions:
                    page_no = region.bbox.page
                    b       = region.bbox.to_dict()
                    pseudo_el = {"page": page_no, "bbox": b}
                    cap_el  = nearest_caption(pseudo_el, all_captions)
                    caption = cap_el.get("text", "") if cap_el else ""
                    parsed_num = parse_caption_num(caption, TAB_NUM_RE)
                    if self._config.merge_tables_by_caption and parsed_num is not None:
                        num = f"{parsed_num}_p{page_no}"
                    else:
                        num = str(len(merged_tables) + 1)
                    is_rotated = region.label.lower() == "table rotated"
                    if num not in merged_tables:
                        merged_tables[num] = {
                            "table_id": parsed_num or num,
                            "caption":  caption or f"Table {num}",
                            "page":     page_no,
                            "bbox":     b,
                            "source":   tatr_source,
                            "rotated":  is_rotated,
                        }
                    elif self._config.merge_tables_by_caption:
                        existing = merged_tables[num]
                        existing["bbox"] = union_bbox(existing["bbox"], b)
                        if len(caption) > len(existing["caption"] or ""):
                            existing["caption"] = caption
                        if is_rotated:
                            existing["rotated"] = True
                        logger.debug("Merged duplicate TATR Table %s", num)

            # Supplementary source: Docling TABLE / RECONSTRUCTED_TABLE elements
            for el in element_dicts:
                el_type = el.get("type")
                if el_type not in docling_table_types:
                    continue
                page_no = el.get("page")
                b       = el.get("bbox") or {}
                if page_no is None or not b:
                    continue
                docling_source = "docling_reconstructed" if el_type == "RECONSTRUCTED_TABLE" else "docling"
                # If this element substantially overlaps an existing detection, merge source and skip
                overlapping = next(
                    (k for k, t in merged_tables.items()
                     if t["page"] == page_no and _overlap_ratio(b, t["bbox"]) > 0.5),
                    None,
                )
                if overlapping is not None:
                    existing = merged_tables[overlapping]
                    existing["source"] = f"{existing['source']}+{docling_source}"
                    logger.debug("Merged overlapping %s into existing entry %s", el_type, overlapping)
                    continue
                cap_el  = nearest_caption(el, all_captions)
                caption = cap_el.get("text", "") if cap_el else el.get("caption") or ""
                parsed_num = parse_caption_num(caption, TAB_NUM_RE)
                if self._config.merge_tables_by_caption and parsed_num is not None:
                    num = f"{parsed_num}_p{page_no}"
                else:
                    num = str(len(merged_tables) + 1)
                if num not in merged_tables:
                    merged_tables[num] = {
                        "table_id": parsed_num or num,
                        "caption":  caption or f"Table {num}",
                        "page":     page_no,
                        "bbox":     b,
                        "source":   docling_source,
                    }
                elif self._config.merge_tables_by_caption:
                    existing = merged_tables[num]
                    existing["bbox"] = union_bbox(existing["bbox"], b)
                    if len(caption) > len(existing["caption"] or ""):
                        existing["caption"] = caption
                    logger.debug("Merged Docling Table %s into existing entry", num)

            if self._config.expand_tables_with_footnotes:
                from pipeline.stages.pdf_text_extraction.components.table_reconstructor import expand_tables_with_footnotes
                expand_tables_with_footnotes(
                    merged_tables, element_dicts,
                    proximity_pts=self._config.footnote_proximity_pts,
                    threshold_multiplier=self._config.footnote_threshold_multiplier,
                    text_proximity_pts=self._config.text_footnote_proximity_pts,
                )

            for tbl in merged_tables.values():
                page_no = tbl["page"]
                b       = tbl["bbox"]
                if page_no is None or not b:
                    continue
                bbox    = BoundingBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"], page=page_no)
                num_int = int(tbl["table_id"]) if str(tbl["table_id"]).isdigit() else None
                label   = f"Table {tbl['table_id']}"
                media   = self._crop_element(
                    doc, bbox, layout.page_dims, mat, label,
                    num_int, tbl["caption"] or None, "table", pdf_path.stem, self._tables_dir,
                    source=tbl.get("source", "unknown"),
                )
                if media:
                    tables.append(media)

        doc.close()
        logger.info(
            "MediaCropper: %d figures, %d tables cropped from %s",
            len(figures), len(tables), pdf_path.name,
        )
        return figures, tables

    # ── Internal ──────────────────────────────────────────────────────────────

    def _crop_element(
        self,
        doc,
        bbox: BoundingBox,
        page_dims: dict,
        mat,
        label: str,
        number: Optional[int],
        caption: Optional[str],
        media_type: str,
        stem: str,
        out_dir: Path,
        source: str = "unknown",
    ) -> Optional[CroppedMedia]:
        page_no = bbox.page
        page_h  = page_dims.get(page_no, {}).get("height", 792.0)
        rect    = bbox.to_fitz_rect(page_h)
        if rect.is_empty:
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "-")
        base       = f"{stem}_{safe_label}_p{page_no}"
        filename   = f"{base}.{self._config.image_format}"
        out_path   = out_dir / filename
        if out_path.exists():
            idx = 2
            while (out_dir / f"{base}_{idx}.{self._config.image_format}").exists():
                idx += 1
            filename = f"{base}_{idx}.{self._config.image_format}"
            out_path = out_dir / filename

        page = doc[page_no - 1]
        pix  = page.get_pixmap(matrix=mat, clip=rect)
        pix.save(str(out_path))

        return CroppedMedia(
            media_type=media_type,
            label=label,
            number=number,
            caption=caption,
            image_path=out_path,
            bbox=bbox,
            page=page_no,
            source=source,
        )
