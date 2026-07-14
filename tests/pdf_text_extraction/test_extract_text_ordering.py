"""Regression tests for B-040 — extract_text contiguous-runs emission.

Pre-fix: extract_text accumulated paragraphs into a defaultdict keyed by
path_string and emitted rows by iterating the dict (path-first-appearance
order). When a sub-section returned to its parent, the parent's later
paragraphs were teleported to the parent's first-emit position.

Post-fix: arrival-order traversal with contiguous-run grouping. Rows emit
in document order; only same-path runs that are adjacent get stitched.
"""
from __future__ import annotations

from nlp_histo.parsers.layout_utils import extract_text


def _header(text: str, level: int) -> dict:
    return {"type": "SECTION_HEADER", "level": level, "text": text}


def _para(text: str) -> dict:
    return {"type": "TEXT", "text": text}


def test_subsection_return_preserves_document_order():
    """Parent → sub-section → back to parent emits rows in arrival order."""
    elements = [
        _header("Methods", level=1),
        _para("alpha alpha alpha alpha"),
        _header("Sub1", level=2),
        _para("beta beta beta beta"),
        _header("Methods", level=1),  # back to parent; clears level-2
        _para("gamma gamma gamma gamma"),
    ]
    rows, _ = extract_text(elements, pre_filter=False)
    paths = [r[0] for r in rows]
    texts = [r[3] for r in rows]
    assert paths == ["Methods", "Methods > Sub1", "Methods"]
    assert texts[0].startswith("alpha")
    assert texts[1].startswith("beta")
    assert texts[2].startswith("gamma")


def test_simple_sequential_sections_ordered():
    """Two sequential top-level sections emit in arrival order."""
    elements = [
        _header("Methods", level=1),
        _para("methods body methods body"),
        _header("Results", level=1),
        _para("results body results body"),
    ]
    rows, _ = extract_text(elements, pre_filter=False)
    paths = [r[0] for r in rows]
    assert paths == ["Methods", "Results"]


def test_dedup_drops_duplicate_text_within_path():
    """Identical text in the same path is deduped (existing path_seen behavior)."""
    elements = [
        _header("Methods", level=1),
        _para("alpha alpha alpha alpha"),
        _para("alpha alpha alpha alpha"),  # exact dup → dropped
        _para("beta beta beta beta"),
    ]
    rows, n_skipped = extract_text(elements, pre_filter=False)
    texts = [r[3] for r in rows]
    assert any(t.startswith("alpha") for t in texts)
    assert any(t.startswith("beta") for t in texts)
    # one alpha + one beta (stitched output may merge consecutive paragraphs)
    assert n_skipped >= 1
