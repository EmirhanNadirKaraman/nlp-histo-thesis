"""
MediaJsonWriter

Writes figure and table caption-matching results to a JSON file per document.

Output format (``out/json/{pmcid}_media.json``):
::

    {
      "pmcid": "PMC123",
      "figures": [
        {
          "label":      "Figure 1",
          "number":     1,
          "caption":    "Haematoxylin and eosin staining...",
          "image_path": "out/figures/PMC123_Figure_1.png",
          "page":       3,
          "source":     "docling",
          "bbox":       {"x1": 56.0, "y1": 612.0, "x2": 540.0, "y2": 342.0}
        },
        ...
      ],
      "tables": [ ... ]
    }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from pipeline.stages.pdf_text_extraction.models.dto import CroppedMedia, HierarchicalRow

logger = logging.getLogger(__name__)


class MediaJsonWriter:
    """
    Writes per-document JSON files containing figure and table metadata,
    including the nearest matched caption for each crop.

    Parameters
    ----------
    output_dir:
        Directory where JSON files are written.  Defaults to ``out/json``.
    """

    def __init__(self, output_dir: Optional[Path] = None, suffix: str = "") -> None:
        self._output_dir = output_dir or Path("out/json")
        self._suffix = suffix

    def write(
        self,
        pmcid: str,
        _rows: List[HierarchicalRow],
        figures: List[CroppedMedia],
        tables: List[CroppedMedia],
        pdf_path=None,  # noqa: ARG002
    ) -> None:
        """
        Write ``{output_dir}/{pmcid}_media.json``.

        ``rows`` is accepted for API compatibility but not used here.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{pmcid}_media{self._suffix}.json"

        payload = {
            "pmcid": pmcid,
            "figures": [self._serialise(m) for m in figures],
            "tables":  [self._serialise(m) for m in tables],
        }

        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info(
            "MediaJsonWriter: wrote %d figures, %d tables → %s",
            len(figures), len(tables), out_path.name,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _serialise(media: CroppedMedia) -> dict:
        return {
            "label":      media.label,
            "number":     media.number,
            "caption":    media.caption,
            "image_path": str(media.image_path) if media.image_path else None,
            "page":       media.page,
            "source":     media.source,
            "bbox":       media.bbox.to_dict(),
        }
