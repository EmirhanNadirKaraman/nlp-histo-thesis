"""
TextFileWriter

Persists assembled pipeline results to plain-text files using the format
defined in parsers/layout_utils.save_text.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from pipeline.stages.pdf_text_extraction.models.dto import CroppedMedia, HierarchicalRow
from nlp_histo.parsers.layout_utils import save_text

logger = logging.getLogger(__name__)


class TextFileWriter:
    """
    Writes hierarchical text rows to a .txt file per document.

    Parameters
    ----------
    output_dir:
        Directory where text files are written.
    label:
        Short descriptor appended to the file header
        (e.g. ``'masked | hierarchical'``).
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        label: str = "pipeline",
    ) -> None:
        self._output_dir = output_dir or Path("out/text")
        self._label = label

    def write(
        self,
        pmcid: str,
        rows: List[HierarchicalRow],
        _figures: List[CroppedMedia],
        _tables: List[CroppedMedia],
        pdf_path=None,  # noqa: ARG002
    ) -> None:
        """
        Write rows to ``{output_dir}/{pmcid}_text.txt``.

        figures and tables are accepted for API compatibility but not written
        to the text file (they are stored as images by the MediaCropper).
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{pmcid}_text.txt"

        # Convert HierarchicalRow objects to the tuple format expected by save_text
        row_tuples = [
            (row.path_string, row.path_list, row.depth, row.text)
            for row in rows
        ]

        save_text(row_tuples, out_path, pmcid=pmcid, label=self._label)
        logger.info("TextFileWriter: wrote %d rows to %s", len(rows), out_path)
