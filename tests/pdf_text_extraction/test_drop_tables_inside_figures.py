"""
Tests for the "drop tables inside figures" filter in
``pipeline/stages/pdf_text_extraction/runner.py``.

Covers:
* ``_bbox_intersect_area`` math (orientation-agnostic for Docling vs fitz coords).
* ``_drop_tables_inside_figures``:
  - drops when ≥ threshold of the table's area lies inside any same-page figure
  - keeps when the overlap is below threshold
  - keeps when no figures are present
  - keeps detections on a different page than the figure
* CLI flag ``--drop-tables-inside-figures`` / ``--no-...``.
* ``main()`` wires the flag into ``cfg.masking.drop_tables_inside_figures``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from pipeline.stages.pdf_text_extraction import runner as runner_mod
from pipeline.stages.pdf_text_extraction.models.dto import (
    BoundingBox,
    DetectedRegion,
    LayoutElement,
    TableDetectionResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _bb(x1: float, y1: float, x2: float, y2: float, page: int = 1) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, page=page)


def _figure(bbox: BoundingBox, etype: str = "FIGURE") -> LayoutElement:
    return LayoutElement(type=etype, page=bbox.page, bbox=bbox, text=None)


def _table_region(bbox: BoundingBox, score: float = 0.95,
                   source: str = "tatr") -> DetectedRegion:
    return DetectedRegion(bbox=bbox, score=score, source=source, label="table")


@dataclass
class _FakeLayout:
    elements: List[LayoutElement] = field(default_factory=list)


def _detection(regions: List[DetectedRegion]) -> TableDetectionResult:
    return TableDetectionResult(
        regions=regions,
        source="hybrid",
        pdf_path=Path("fake.pdf"),
        page_dims={},
    )


# ── _bbox_intersect_area ──────────────────────────────────────────────────────


def test_intersect_full_containment() -> None:
    # Table fully inside figure (Docling coord style: y1 > y2)
    table = _bb(10, 200, 20, 100)
    fig = _bb(0, 250, 30, 50)
    inter = runner_mod._bbox_intersect_area(table, fig)
    table_area = runner_mod._bbox_area_pts(table)
    assert inter == pytest.approx(table_area)


def test_intersect_no_overlap() -> None:
    a = _bb(0, 100, 10, 0)
    b = _bb(20, 100, 30, 0)
    assert runner_mod._bbox_intersect_area(a, b) == 0


def test_intersect_partial_overlap() -> None:
    # 10×10 overlap on a 20×20 vs 20×20 box → 100 sq pts
    a = _bb(0, 20, 20, 0)
    b = _bb(10, 30, 30, 10)
    inter = runner_mod._bbox_intersect_area(a, b)
    assert inter == pytest.approx(100)


def test_intersect_different_pages_returns_zero() -> None:
    a = _bb(0, 100, 100, 0, page=1)
    b = _bb(0, 100, 100, 0, page=2)
    assert runner_mod._bbox_intersect_area(a, b) == 0


def test_intersect_handles_fitz_orientation() -> None:
    """fitz uses y₀ < y₁ (top-origin).  Helper is orientation-agnostic."""
    a = _bb(0, 0, 20, 20)   # fitz-style
    b = _bb(10, 10, 30, 30)
    inter = runner_mod._bbox_intersect_area(a, b)
    assert inter == pytest.approx(100)


# ── _drop_tables_inside_figures ──────────────────────────────────────────────


def test_drop_keeps_when_no_figures_present() -> None:
    det = _detection([_table_region(_bb(0, 50, 100, 0))])
    layout = _FakeLayout(elements=[])
    out, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    assert dropped == 0
    assert len(out.regions) == 1


def test_drop_eliminates_table_fully_inside_figure() -> None:
    table_bbox = _bb(20, 80, 40, 60)
    fig_bbox = _bb(10, 100, 60, 50)  # contains table_bbox
    det = _detection([_table_region(table_bbox)])
    layout = _FakeLayout(elements=[_figure(fig_bbox)])
    out, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    assert dropped == 1
    assert out.regions == []


def test_drop_keeps_table_with_low_overlap() -> None:
    table_bbox = _bb(0, 100, 100, 0)        # 100×100 = 10000 area
    fig_bbox = _bb(0, 100, 10, 50)          # 10×50 = 500 → 5 % of table
    det = _detection([_table_region(table_bbox)])
    layout = _FakeLayout(elements=[_figure(fig_bbox)])
    out, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    assert dropped == 0
    assert len(out.regions) == 1


def test_drop_keeps_table_just_below_threshold() -> None:
    """A table at 79 % overlap is kept (threshold defaults to 0.8)."""
    table_bbox = _bb(0, 100, 100, 0)       # 10 000 area
    fig_bbox = _bb(0, 100, 79, 100)        # Wait this is degenerate
    # Use a clearer construction: fig covers 79 % of table
    table_bbox = _bb(0, 100, 100, 0)       # 100×100 = 10000
    fig_bbox = _bb(0, 100, 79, 0)          # 79×100 = 7900, 79 % overlap
    det = _detection([_table_region(table_bbox)])
    layout = _FakeLayout(elements=[_figure(fig_bbox)])
    out, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    assert dropped == 0
    assert len(out.regions) == 1


def test_drop_uses_custom_threshold() -> None:
    table_bbox = _bb(0, 100, 100, 0)
    fig_bbox = _bb(0, 100, 50, 0)  # 50 %
    det = _detection([_table_region(table_bbox)])
    layout = _FakeLayout(elements=[_figure(fig_bbox)])
    # threshold 0.4 → drop, threshold 0.8 → keep
    _, dropped_strict = runner_mod._drop_tables_inside_figures(det, layout, threshold=0.8)
    _, dropped_loose  = runner_mod._drop_tables_inside_figures(det, layout, threshold=0.4)
    assert dropped_strict == 0
    assert dropped_loose == 1


def test_drop_respects_page_boundary() -> None:
    """A figure on page 2 must NOT drop a table on page 1."""
    table_bbox = _bb(0, 100, 100, 0, page=1)
    fig_bbox = _bb(0, 100, 100, 0, page=2)
    det = _detection([_table_region(table_bbox)])
    layout = _FakeLayout(elements=[_figure(fig_bbox)])
    out, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    assert dropped == 0
    assert len(out.regions) == 1


def test_drop_accepts_picture_element_type() -> None:
    """PICTURE counts as a figure for the filter (Docling tags both)."""
    table_bbox = _bb(0, 50, 10, 0)
    pic_bbox = _bb(-10, 60, 20, -10)
    det = _detection([_table_region(table_bbox)])
    layout = _FakeLayout(elements=[_figure(pic_bbox, etype="PICTURE")])
    _, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    assert dropped == 1


def test_drop_ignores_non_figure_elements() -> None:
    """TEXT / SECTION_HEADER must not influence the filter."""
    table_bbox = _bb(0, 50, 10, 0)
    bg_bbox = _bb(-10, 60, 20, -10)
    det = _detection([_table_region(table_bbox)])
    layout = _FakeLayout(elements=[_figure(bg_bbox, etype="SECTION_HEADER")])
    out, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    assert dropped == 0
    assert len(out.regions) == 1


def test_drop_returns_new_detection_does_not_mutate_input() -> None:
    table_bbox = _bb(20, 80, 40, 60)
    fig_bbox = _bb(10, 100, 60, 50)
    det = _detection([_table_region(table_bbox), _table_region(_bb(100, 200, 200, 100))])
    layout = _FakeLayout(elements=[_figure(fig_bbox)])
    original_regions = list(det.regions)
    out, dropped = runner_mod._drop_tables_inside_figures(det, layout)
    # Original unchanged
    assert det.regions == original_regions
    # New result has 1 fewer
    assert len(out.regions) == 1
    assert dropped == 1


# ── CLI flag wiring ──────────────────────────────────────────────────────────


def test_parse_args_default_is_none() -> None:
    args, _ = runner_mod._parse_args([])
    assert args.drop_tables_inside_figures is None


def test_parse_args_positive_flag() -> None:
    args, _ = runner_mod._parse_args(["--drop-tables-inside-figures"])
    assert args.drop_tables_inside_figures is True


def test_parse_args_negative_flag() -> None:
    args, _ = runner_mod._parse_args(["--no-drop-tables-inside-figures"])
    assert args.drop_tables_inside_figures is False


# ── main() honors the flag ───────────────────────────────────────────────────


class _CapturedConfig:
    def __init__(self) -> None:
        self.cfg = None

    def __call__(self, cfg, max_workers=None):
        self.cfg = cfg
        return self

    def run(self, *args, **kwargs):
        return {"processed": 0, "failed": 0, "skipped": 0}


def _run_main_and_capture(argv) -> _CapturedConfig:
    captured = _CapturedConfig()
    with patch("pipeline.stages.pdf_text_extraction.batch.ParallelBatchRunner",
               captured):
        runner_mod.main(argv)
    return captured


def test_main_default_is_filter_on() -> None:
    """Default (no flag) → filter ON (flipped 2026-05-18)."""
    captured = _run_main_and_capture(["--pdf-dir", "/tmp/does_not_exist"])
    assert captured.cfg.masking.drop_tables_inside_figures is True


def test_main_positive_flag_redundant_but_still_works() -> None:
    """--drop-tables-inside-figures is now a no-op alongside the default,
    but explicit override should still resolve to True."""
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--drop-tables-inside-figures",
    ])
    assert captured.cfg.masking.drop_tables_inside_figures is True


def test_main_negative_flag_disables_filter() -> None:
    """--no-drop-tables-inside-figures opts back into the historical
    (pre-2026-05-18) behaviour."""
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--no-drop-tables-inside-figures",
    ])
    assert captured.cfg.masking.drop_tables_inside_figures is False
