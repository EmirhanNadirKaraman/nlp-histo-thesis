"""
PostgresDatabaseIngester

Persists assembled document data (text elements, figures, tables, and their
cross-references) to the PostgreSQL database using the existing SQLAlchemy
models in database/.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import List

from pipeline.stages.pdf_text_extraction.models.dto import CroppedMedia, HierarchicalRow

logger = logging.getLogger(__name__)

# Patterns for detecting inline figure/table references in text
_FIG_REF_RE = re.compile(r'\bfig(?:ure)?\.?\s*(\d+)', re.IGNORECASE)
_TAB_REF_RE = re.compile(r'\btable\.?\s*(\d+)', re.IGNORECASE)


class PostgresDatabaseIngester:
    """
    Ingests pipeline results into the PostgreSQL database.

    Parameters
    ----------
    db:
        A DatabaseConnection instance (from database.db_connection).
        If None, the connection is created lazily on first write.
    """

    def __init__(self, db=None, db_url: str | None = None) -> None:
        self._db = db
        self._db_url = db_url

    # ── Lazy DB connection ────────────────────────────────────────────────────

    def _get_db(self):
        if self._db is None:
            from database import get_db_connection  # type: ignore
            self._db = get_db_connection(database_url=self._db_url)
        return self._db

    # ── Public API ────────────────────────────────────────────────────────────

    def write(
        self,
        pmcid: str,
        rows: List[HierarchicalRow],
        figures: List[CroppedMedia],
        tables: List[CroppedMedia],
        pdf_path=None,
    ) -> None:
        """
        Ingest all data for ``pmcid`` into the database.

        Skips the document if it already exists (idempotent by pmcid).

        Args:
            pmcid:    PubMed Central document identifier.
            rows:     Hierarchical text rows.
            figures:  Cropped figure metadata.
            tables:   Cropped table metadata.
            pdf_path: Path to the source PDF (stored in documents.file_path).
        """
        from database import Document, Figure, Table, TextElement  # type: ignore
        from database.models import (  # type: ignore
            TextElementFigureReference,
            TextElementTableReference,
        )

        db = self._get_db()

        with db.session_scope() as session:
            # Skip if already ingested
            existing = session.query(Document).filter_by(pmcid=pmcid).first()
            if existing:
                logger.info("DB: %s already ingested — skipping.", pmcid)
                return

            doc = Document(
                pmcid=pmcid,
                filename=f"{pmcid}.pdf",
                file_path=str(pdf_path) if pdf_path else f"{pmcid}.pdf",
                text_source="pdf",
            )
            session.add(doc)
            session.flush()  # get doc.id

            # ── Figures ───────────────────────────────────────────────────────
            fig_objs = {}
            for i, fig in enumerate(figures, start=1):
                obj = Figure(
                    document_id=doc.id,
                    figure_id=f"{pmcid}_fig_{i}",
                    figure_label=fig.label,
                    figure_number=fig.number,
                    caption_text=fig.caption,
                    image_filename=fig.image_path.name if fig.image_path else None,
                    image_path=str(fig.image_path) if fig.image_path else None,
                    # B-075 (migration 0014): persist the cropper's page+bbox (Docling coords).
                    page_number=fig.page,
                    bbox_x1=fig.bbox.x1, bbox_y1=fig.bbox.y1,
                    bbox_x2=fig.bbox.x2, bbox_y2=fig.bbox.y2,
                )
                session.add(obj)
                session.flush()
                fig_objs[fig.number or i] = obj

            # ── Tables ────────────────────────────────────────────────────────
            tab_objs = {}
            for i, tbl in enumerate(tables, start=1):
                obj = Table(
                    document_id=doc.id,
                    table_id=f"{pmcid}_tbl_{i}",
                    table_label=tbl.label,
                    table_number=tbl.number,
                    caption_text=tbl.caption,
                    image_filename=tbl.image_path.name if tbl.image_path else None,
                    image_path=str(tbl.image_path) if tbl.image_path else None,
                    # B-075: persist the page+bbox the cropper already computed on
                    # CroppedMedia (Docling coords, y=0 at page bottom) — previously dropped.
                    page_number=tbl.page,
                    bbox_x1=tbl.bbox.x1, bbox_y1=tbl.bbox.y1,
                    bbox_x2=tbl.bbox.x2, bbox_y2=tbl.bbox.y2,
                )
                session.add(obj)
                session.flush()
                tab_objs[tbl.number or i] = obj

            # ── Text elements ─────────────────────────────────────────────────
            position_counter: dict = defaultdict(int)
            for row in rows:
                pos = position_counter[row.path_string]
                position_counter[row.path_string] += 1
                unique_path = f"{pmcid}/{row.path_string}/{pos}"

                fig_refs = list(dict.fromkeys(int(m) for m in _FIG_REF_RE.findall(row.text)))
                tab_refs = list(dict.fromkeys(int(m) for m in _TAB_REF_RE.findall(row.text)))

                te = TextElement(
                    document_id=doc.id,
                    unique_path=unique_path,
                    path_list=row.path_list,
                    path_string=row.path_string,
                    depth=row.depth,
                    position_in_section=pos,
                    text_content=row.text,
                    references={"figures": fig_refs, "tables": tab_refs},
                )
                session.add(te)
                session.flush()

                # Junction-table references
                for fig_num in fig_refs:
                    if fig_num in fig_objs:
                        session.add(TextElementFigureReference(
                            text_element_id=te.id,
                            figure_id=fig_objs[fig_num].id,
                        ))
                for tab_num in tab_refs:
                    if tab_num in tab_objs:
                        session.add(TextElementTableReference(
                            text_element_id=te.id,
                            table_id=tab_objs[tab_num].id,
                        ))

        logger.info(
            "DB: ingested %s — %d rows, %d figures, %d tables",
            pmcid, len(rows), len(figures), len(tables),
        )
