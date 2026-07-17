"""
HierarchicalTextAssembler

Walks a LayoutResult in document order, tracks section headers to build a
hierarchical path, filters skippable element types, and returns stitched,
citation-free HierarchicalRow objects.

Delegates to the existing utilities in parsers/layout_utils.py and
parsers/text_processing.py so that all text-processing logic stays in one
place.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from nlp_histo.pipeline.stages.pdf_text_extraction.config import TextAssemblyConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import HierarchicalRow, LayoutResult
from nlp_histo.parsers.layout_utils import extract_text
from nlp_histo.parsers.text_processing import is_reference_entry

logger = logging.getLogger(__name__)


class HierarchicalTextAssembler:
    """
    Assembles hierarchical text rows from a LayoutResult.

    Parameters
    ----------
    config:
        TextAssemblyConfig controlling stitching, citation removal, etc.
    skip_references_section:
        Drop any section whose top-level header matches "References" or
        "Bibliography".
    nlp:
        Optional scispaCy model for relevance filtering (passed through to
        ``extract_text``).
    """

    _REF_HEADERS = frozenset({"references", "bibliography", "reference list"})

    def __init__(
        self,
        config: Optional[TextAssemblyConfig] = None,
        skip_references_section: bool = True,
        nlp=None,
    ) -> None:
        self._config = config or TextAssemblyConfig()
        self._skip_refs = skip_references_section
        self._nlp = nlp

    def assemble(self, layout: LayoutResult) -> List[HierarchicalRow]:
        """
        Assemble text from ``layout`` into hierarchical rows.

        Args:
            layout: LayoutResult (typically from the masked PDF).

        Returns:
            List of HierarchicalRow in document order.
        """
        element_dicts = layout.to_element_dicts()

        # For unmasked mode, also pass table bboxes so they can be skipped
        table_bboxes = None  # already masked; no extra filtering needed

        rows_raw, n_skipped = extract_text(
            element_dicts,
            nlp=self._nlp,
            table_bboxes=table_bboxes,
            pre_filter=self._config.pre_filter_relevance,
            with_sources=True,
        )
        logger.debug("TextAssembler: %d rows assembled, %d skipped", len(rows_raw), n_skipped)

        rows: List[HierarchicalRow] = []
        for path_str, path_list, depth, text, source_chunks in rows_raw:
            if self._skip_refs:
                if path_list and path_list[0].strip().lower() in self._REF_HEADERS:
                    continue
                if is_reference_entry(text):
                    continue
            rows.append(HierarchicalRow(
                path_string=path_str,
                path_list=path_list,
                depth=depth,
                text=text,
                source_chunks=source_chunks,
            ))

        return rows
