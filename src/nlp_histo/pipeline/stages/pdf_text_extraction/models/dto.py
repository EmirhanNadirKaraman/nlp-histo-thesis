"""
Data Transfer Objects (DTOs) for the pipeline.

All inter-stage data is passed as typed dataclasses defined here.

Coordinate convention (Docling PDF space):
  x1, y1 = top-left;  x2, y2 = bottom-right.
  The y-axis origin is at the BOTTOM of the page (PDF convention), so
  y1 > y2 for a normal element.  Use BoundingBox.to_fitz_rect() to
  convert to screen-space fitz.Rect (y = 0 at top).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Geometry ──────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Bounding box in Docling PDF coordinates (points)."""
    x1: float
    y1: float   # top  — larger value in Docling coords
    x2: float
    y2: float   # bottom — smaller value in Docling coords
    page: int

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, d: dict, page: int) -> "BoundingBox":
        return cls(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"], page=page)

    def to_fitz_rect(self, page_height: float) -> Any:
        """Convert to fitz.Rect (screen coordinates, y = 0 at top)."""
        import fitz  # type: ignore
        top    = page_height - max(self.y1, self.y2)
        bottom = page_height - min(self.y1, self.y2)
        return fitz.Rect(self.x1, top, self.x2, bottom)

    @classmethod
    def from_fitz_rect(cls, rect: Any, page_height: float, page: int) -> "BoundingBox":
        """Build from a fitz.Rect (screen coordinates) back to Docling PDF coords."""
        return cls(
            x1=rect.x0,
            y1=page_height - rect.y0,
            x2=rect.x1,
            y2=page_height - rect.y1,
            page=page,
        )


# ── Detection ─────────────────────────────────────────────────────────────────

@dataclass
class DetectedRegion:
    """A detected region (table or figure) within a PDF page."""
    bbox: BoundingBox
    score: float
    source: str           # 'tatr' | 'docling' | 'hybrid' | 'vlm'
    label: str = "table"  # 'table' | 'figure' | ...


@dataclass
class TableDetectionResult:
    """Aggregated output from a TableDetector implementation."""
    regions: List[DetectedRegion]
    source: str
    pdf_path: Path
    page_dims: Dict[int, Dict[str, float]] = field(default_factory=dict)


# ── Layout ────────────────────────────────────────────────────────────────────

@dataclass
class LayoutElement:
    """One element extracted from a PDF layout pass."""
    type: str    # 'TEXT' | 'TABLE' | 'PICTURE' | 'SECTION_HEADER' | 'CAPTION' | …
    page: int
    bbox: BoundingBox
    text: Optional[str]
    level: int = 0

    def to_dict(self) -> dict:
        """Return the dict format used throughout the existing parser utilities."""
        return {
            "type":  self.type,
            "page":  self.page,
            "bbox":  self.bbox.to_dict(),
            "text":  self.text,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutElement":
        page = d.get("page", 0)
        return cls(
            type=d.get("type", "TEXT"),
            page=page,
            bbox=BoundingBox.from_dict(d["bbox"], page),
            text=d.get("text"),
            level=d.get("level", 0),
        )


@dataclass
class LayoutResult:
    """Output of a layout-extraction stage."""
    elements: List[LayoutElement]
    page_dims: Dict[int, Dict[str, float]]  # {page_no: {'width': ..., 'height': ...}}
    pdf_path: Path
    source: str = "docling"

    def to_element_dicts(self) -> List[dict]:
        """Convert to the legacy dict format used by parsers/layout_utils.py helpers."""
        return [el.to_dict() for el in self.elements]


# ── Text assembly ─────────────────────────────────────────────────────────────

@dataclass
class HierarchicalRow:
    """A single paragraph with its section-hierarchy path."""
    path_string: str          # e.g. "Methods > 2.1 Staining"
    path_list: List[str]      # e.g. ["Methods", "2.1 Staining"]
    depth: int
    text: str
    # Pre-stitch chunks merged into `text`.  len > 1 means stitching occurred;
    # the rule-extraction pipeline can reference each chunk individually.
    source_chunks: List[str] = field(default_factory=list)


# ── Media ─────────────────────────────────────────────────────────────────────

@dataclass
class CroppedMedia:
    """A cropped figure or table image extracted from the PDF."""
    media_type: str            # 'figure' | 'table'
    label: str                 # e.g. 'Figure 1', 'Table 2'
    number: Optional[int]
    caption: Optional[str]
    image_path: Optional[Path]
    bbox: BoundingBox
    page: int
    source: str = "unknown"    # 'tatr' | 'docling' | 'docling_reconstructed' | 'tatr+docling' | etc.
