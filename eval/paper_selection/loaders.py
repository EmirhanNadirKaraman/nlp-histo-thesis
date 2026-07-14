"""Loaders that produce a ``RawPaper`` per paper.

Two backends:

  - :class:`DBLoader`     — reads from the ingestion Postgres DB.
  - :class:`JSONLLoader`  — fallback that reads pre-exported JSONL.

Both yield the same in-memory shape so :mod:`fingerprints` is backend-agnostic.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_PROGRESS_EVERY = 50


# ── Selection-YAML helper ────────────────────────────────────────────────────

def load_pmcids_from_selection(path: str | Path) -> list[str]:
    """Flatten a ``related``/``diverse``/``hard`` selection YAML to its PMCID list.

    Accepts both the wrapped form (a single top-level version key, e.g.
    ``related15:``) and a bare ``{related: [...], ...}`` mapping.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    block = data
    if not any(k in block for k in ("related", "diverse", "hard")) and data:
        block = next(iter(data.values())) or {}
    return [
        *(block.get("related") or []),
        *(block.get("diverse") or []),
        *(block.get("hard") or []),
    ]


@dataclass
class RawEntity:
    """One entity row, in either backend."""
    text: str
    label: str
    umls_cui: str | None = None
    canonical_name: str | None = None
    semantic_types: list[str] = field(default_factory=list)
    umls_score: float | None = None


@dataclass
class RawPaper:
    """Backend-agnostic paper payload that ``build_fingerprint`` consumes."""
    pmcid: str
    title: str | None
    text_elements: list[str]                # paragraph text, in order
    entities: list[RawEntity]
    n_figures: int
    n_tables: int
    n_captions: int


# ── DB loader ────────────────────────────────────────────────────────────────

class DBLoader:
    """Read papers + entities + figures/tables straight from Postgres.

    Uses the project's ``database`` package; instantiation is cheap, but each
    ``load_paper`` call opens a session via ``session_scope``.
    """

    def __init__(self, db=None, min_umls_score: float = 0.0) -> None:
        self._db = db
        self._min_umls_score = min_umls_score

    def _get_db(self):
        if self._db is not None:
            return self._db
        from nlp_histo.database import get_db_connection
        self._db = get_db_connection()
        return self._db

    def list_pmcids(self) -> list[str]:
        from nlp_histo.database import Document
        db = self._get_db()
        t0 = time.perf_counter()
        with db.session_scope() as session:
            pmcids = [row.pmcid for row in session.query(Document.pmcid).all()]
        logger.info("DBLoader.list_pmcids: %d pmcids in %.2fs",
                    len(pmcids), time.perf_counter() - t0)
        return pmcids

    def load_paper(self, pmcid: str) -> RawPaper | None:
        from nlp_histo.database import Document, Entity, Figure, Table, TextElement
        db = self._get_db()
        with db.session_scope() as session:
            doc = session.query(Document).filter_by(pmcid=pmcid).one_or_none()
            if doc is None:
                return None

            te_rows = (
                session.query(TextElement.text_content)
                .filter(TextElement.document_id == doc.id)
                .order_by(TextElement.id)
                .all()
            )
            text_elements = [t for (t,) in te_rows if t]

            ent_rows = (
                session.query(
                    Entity.entity_text,
                    Entity.entity_label,
                    Entity.umls_cui,
                    Entity.canonical_name,
                    Entity.semantic_types,
                    Entity.umls_score,
                )
                .join(TextElement, Entity.text_element_id == TextElement.id)
                .filter(TextElement.document_id == doc.id)
                .all()
            )
            entities: list[RawEntity] = []
            for text, label, cui, canonical, semtypes, score in ent_rows:
                if score is not None and score < self._min_umls_score:
                    continue
                entities.append(RawEntity(
                    text=text, label=label,
                    umls_cui=cui, canonical_name=canonical,
                    semantic_types=list(semtypes or []),
                    umls_score=score,
                ))

            n_figures = session.query(Figure).filter_by(document_id=doc.id).count()
            n_tables  = session.query(Table).filter_by(document_id=doc.id).count()
            n_captions = (
                session.query(Figure.caption_text)
                .filter(Figure.document_id == doc.id, Figure.caption_text.isnot(None))
                .count()
            ) + (
                session.query(Table.caption_text)
                .filter(Table.document_id == doc.id, Table.caption_text.isnot(None))
                .count()
            )

            return RawPaper(
                pmcid=doc.pmcid,
                title=doc.title,
                text_elements=text_elements,
                entities=entities,
                n_figures=n_figures,
                n_tables=n_tables,
                n_captions=n_captions,
            )

    def iter_papers(self, pmcids: Iterable[str] | None = None) -> Iterable[RawPaper]:
        targets = list(pmcids) if pmcids is not None else self.list_pmcids()
        logger.info("DBLoader.iter_papers: loading %d papers from DB", len(targets))
        t0 = time.perf_counter()
        n_yielded = 0
        n_missing = 0
        for i, pmcid in enumerate(targets, start=1):
            paper = self.load_paper(pmcid)
            if paper is None:
                logger.warning("DBLoader: pmcid %s not found", pmcid)
                n_missing += 1
                continue
            n_yielded += 1
            yield paper
            if i % _PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(targets) - i) / rate if rate > 0 else 0.0
                logger.info("DBLoader: %d/%d loaded (%.1f papers/s, ETA %.0fs)",
                            i, len(targets), rate, eta)
        logger.info("DBLoader: done — %d loaded, %d missing in %.1fs",
                    n_yielded, n_missing, time.perf_counter() - t0)


# ── JSONL loader ─────────────────────────────────────────────────────────────

class JSONLLoader:
    """Fallback loader for environments where the DB is not reachable.

    Expected shape (one JSON object per line)::

        {
          "pmcid": "PMC1",
          "title": "...",
          "text_elements": ["paragraph 1", "paragraph 2", ...],
          "entities": [
            {"text": "CD30", "label": "ENTITY", "umls_cui": "C0...",
             "canonical_name": "CD30", "semantic_types": ["T116"],
             "umls_score": 0.95},
            ...
          ],
          "n_figures": 4,
          "n_tables": 2,
          "n_captions": 6
        }
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def list_pmcids(self) -> list[str]:
        return [p.pmcid for p in self.iter_papers()]

    def iter_papers(self, pmcids: Iterable[str] | None = None) -> Iterable[RawPaper]:
        wanted = set(pmcids) if pmcids is not None else None
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if wanted is not None and obj.get("pmcid") not in wanted:
                    continue
                yield RawPaper(
                    pmcid=obj["pmcid"],
                    title=obj.get("title"),
                    text_elements=list(obj.get("text_elements") or []),
                    entities=[
                        RawEntity(
                            text=e["text"],
                            label=e.get("label", "ENTITY"),
                            umls_cui=e.get("umls_cui"),
                            canonical_name=e.get("canonical_name"),
                            semantic_types=list(e.get("semantic_types") or []),
                            umls_score=e.get("umls_score"),
                        )
                        for e in (obj.get("entities") or [])
                    ],
                    n_figures=int(obj.get("n_figures", 0) or 0),
                    n_tables=int(obj.get("n_tables", 0) or 0),
                    n_captions=int(obj.get("n_captions", 0) or 0),
                )
