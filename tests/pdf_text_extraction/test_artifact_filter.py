"""Regression tests for B-010 — ArtifactFilter identity rebuild."""
from __future__ import annotations

from unittest.mock import patch

from nlp_histo.pipeline.stages.pdf_text_extraction.components.artifact_filter import (
    ArtifactFilter,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.config import FilteringConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, LayoutElement


def _el(text: str, *, page: int = 1, etype: str = "TEXT", y1: float = 700.0,
        y2: float = 600.0) -> LayoutElement:
    return LayoutElement(
        type=etype,
        page=page,
        bbox=BoundingBox(x1=50.0, y1=y1, x2=500.0, y2=y2, page=page),
        text=text,
    )


def test_filter_drops_empty_text():
    """Sanity: an empty-text element is dropped, anchors survive."""
    elements = [
        _el("This is a real paragraph with several alphabetic words.", y1=720, y2=400),
        _el("   "),  # empty after strip → filter_artifacts drops it
    ]
    out = ArtifactFilter(FilteringConfig(apply_ner_filtering=False)).filter_elements(elements)
    assert len(out) == 1
    assert out[0] is elements[0]  # original instance preserved


def test_identity_preserved_under_dict_mutation():
    """B-010 regression — if filter_artifacts mutates a kept dict in place, the
    rebuild must still return the corresponding LayoutElement.

    The pre-fix rebuild (`element_dicts[i] in filtered_dicts`) compared dicts by
    value, so any in-place mutation of a kept dict would break equality and
    silently drop the element.  The id()-based rebuild survives mutation.
    """
    elements = [
        _el("This is a real anchor paragraph with several alphabetic words.",
            y1=720, y2=400),
        _el("Second real anchor paragraph also with alphabetic content here.",
            y1=720, y2=400, page=2),
    ]

    def mutating_filter(dicts, nlp=None):  # noqa: ARG001
        # Simulate a future ligature normalisation pass that rewrites text in place.
        for d in dicts:
            d["text"] = (d.get("text") or "").replace("a", "A")
        return list(dicts)

    with patch(
        "nlp_histo.pipeline.stages.pdf_text_extraction.components.artifact_filter.filter_artifacts",
        side_effect=mutating_filter,
    ):
        out = ArtifactFilter(FilteringConfig(apply_ner_filtering=False)).filter_elements(elements)

    assert len(out) == 2
    assert out[0] is elements[0]
    assert out[1] is elements[1]
