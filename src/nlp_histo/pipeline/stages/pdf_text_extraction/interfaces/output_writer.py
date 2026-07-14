from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import CroppedMedia, HierarchicalRow


@runtime_checkable
class OutputWriter(Protocol):
    """
    Contract for pipeline output-writing components.

    An implementation persists the assembled document data (text rows, figures,
    tables) to whatever backing store it targets (file system, database, …).
    """

    def write(
        self,
        pmcid: str,
        rows: List[HierarchicalRow],
        figures: List[CroppedMedia],
        tables: List[CroppedMedia],
        pdf_path: Path | None = None,
    ) -> None:
        """Persist results for one document. ``pdf_path`` is used only by some
        writers (e.g. the DB ingester); writers ignore the inputs they don't need."""
        ...
