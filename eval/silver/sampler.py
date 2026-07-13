"""Sample source cases from the TextElement table."""
from __future__ import annotations

import logging
import random
from typing import List

logger = logging.getLogger(__name__)

# Sections to exclude (references, acknowledgements, etc.)
_SKIP_SECTIONS = {
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "funding", "conflicts of interest", "conflict of interest",
    "author contributions", "supplementary", "appendix",
}

# Minimum paragraph length to be useful
_MIN_WORDS = 40


def _is_useful(te) -> bool:
    """Filter out boilerplate text elements."""
    if te.word_count and te.word_count < _MIN_WORDS:
        return False
    path_lower = (te.path_string or "").lower()
    for skip in _SKIP_SECTIONS:
        if skip in path_lower:
            return False
    return True


def sample_source_cases(
    n: int,
    seed: int = 42,
    pmcids: List[str] | None = None,
) -> List[dict]:
    """
    Sample n TextElement rows from the database.

    Returns list of dicts matching SourceCase schema.
    No pipeline imports — only database access.

    ``seed`` is frozen at 42 so the silver source-case set is reproducible.
    """
    from database import get_db_connection
    from database.models import TextElement, Document

    db = get_db_connection()
    with db.session_scope() as session:
        q = (
            session.query(TextElement, Document.pmcid)
            .join(Document, TextElement.document_id == Document.id)
        )
        if pmcids:
            q = q.filter(Document.pmcid.in_(pmcids))

        candidates = []
        for te, pmcid in q.all():
            text = te.text_content or ""
            wc = te.word_count or len(text.split())
            path = te.path_string or ""
            if wc < _MIN_WORDS:
                continue
            if any(skip in path.lower() for skip in _SKIP_SECTIONS):
                continue
            candidates.append({
                "te_id": te.id,
                "pmcid": pmcid,
                "text": text,
                "path_string": path,
                "word_count": wc,
            })

    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n, len(candidates)))

    cases = []
    for c in chosen:
        case_id = f"{c['pmcid']}|{c['te_id']}"
        cases.append({
            "case_id": case_id,
            "pmcid": c["pmcid"],
            "te_id": c["te_id"],
            "text": c["text"],
            "path_string": c["path_string"],
        })

    logger.info("Sampled %d / %d candidate paragraphs", len(cases), len(candidates))
    return cases


def sample_papers_source_cases(
    n_papers: int,
    seed: int = 42,
    pmcids: list[str] | None = None,
) -> list[dict]:
    """
    Randomly select n_papers papers and return ALL useful text elements from them.

    Unlike sample_source_cases (which samples individual TEs), this returns the
    complete set of paragraphs for each selected paper — giving the pipeline
    per-paper context for RELATE/RESOLVE calibration.
    """
    from database import get_db_connection
    from database.models import Document

    db = get_db_connection()
    with db.session_scope() as session:
        q = session.query(Document.pmcid).order_by(Document.pmcid)
        if pmcids:
            q = q.filter(Document.pmcid.in_(pmcids))
        all_pmcids = [row.pmcid for row in q.all()]

    if not all_pmcids:
        logger.warning("No documents found in database.")
        return []

    rng = random.Random(seed)
    chosen_pmcids = rng.sample(all_pmcids, min(n_papers, len(all_pmcids)))
    logger.info("Selected %d papers (seed=%d): %s", len(chosen_pmcids), seed, ", ".join(chosen_pmcids))

    return sample_source_cases(n=10_000_000, seed=seed, pmcids=chosen_pmcids)
