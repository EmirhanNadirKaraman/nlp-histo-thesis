"""
eval.llm_judge.sampling — Deterministic paper and paragraph sampling.

Paper selection: latest successful run per PMCID, word-count filtered.
Paragraph sampling: stratified into with-extraction and zero-extraction
strata, with section/keyword filtering for zero-extraction paragraphs.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, text

logger = logging.getLogger(__name__)

# Sections unlikely to contain extractable medical findings.
# Used to filter zero-extraction paragraphs so we don't waste judge calls
# on acknowledgements, author metadata, etc.
_SKIP_SECTION_PATTERNS = re.compile(
    r"(?i)\b(acknowledg|funding|conflict.of.interest|author.contrib|"
    r"data.availability|supplementary|abbreviation|ethical.approval|"
    r"competing.interest|consent.for.publication|reference)\b"
)

# Very short paragraphs are typically headers, captions, or labels.
_MIN_WORDS_DEFAULT = 25


@dataclass
class PaperSample:
    pmcid: str
    run_id: int
    word_count: int


@dataclass
class ParagraphSample:
    text_element_id: int
    text_content: str
    word_count: int
    stratum: str           # "with_extractions" | "zero_extractions"
    extracted_claims: list[str]   # claim texts for this paragraph
    section_path: str      # hierarchical path for context


# ── Deterministic seeding ────────────────────────────────────────────────────

def stable_seed(global_seed: int, pmcid: str, run_id: str | int, task_name: str) -> int:
    """Per-task, per-paper deterministic seed.

    Adding a new paper does not change the sample for existing papers.
    """
    blob = f"{global_seed}:{pmcid}:{run_id}:{task_name}"
    return int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)


# ── Paper selection ──────────────────────────────────────────────────────────

def select_papers(
    session: Any,
    n: int,
    seed: int,
    min_words: int,
    max_words: int,
) -> list[PaperSample]:
    """Select n papers: latest successful run per PMCID, word-count filtered."""
    from nlp_histo.database import Document, TextElement  # noqa: PLC0415

    rows = session.execute(text(
        "SELECT DISTINCT ON (pmcid) pmcid, id AS run_id "
        "FROM pipeline_runs WHERE status = 'success' "
        "ORDER BY pmcid, started_at DESC"
    )).fetchall()

    candidates: list[PaperSample] = []
    for pmcid, run_id in rows:
        doc = session.query(Document).filter_by(pmcid=pmcid).first()
        if doc is None:
            continue
        wc = (
            session.query(func.sum(TextElement.word_count))
            .filter_by(document_id=doc.id)
            .scalar()
        ) or 0
        if min_words <= wc <= max_words:
            candidates.append(PaperSample(pmcid=pmcid, run_id=run_id, word_count=wc))

    logger.info(
        "%d papers pass word-count filter (%d–%d words)",
        len(candidates), min_words, max_words,
    )
    if not candidates:
        return []

    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n, len(candidates)))
    logger.info("Sampled %d papers (seed=%d)", len(chosen), seed)
    for p in sorted(chosen, key=lambda x: x.pmcid):
        logger.info("  %-20s  run_id=%-6d  words=%d", p.pmcid, p.run_id, p.word_count)
    return chosen


# ── Paragraph sampling ───────────────────────────────────────────────────────

def _parse_te_ids(evidence_refs: list[str] | None) -> set[int]:
    """Parse text_element_ids from evidence_refs format 'S1|PMC123456|789'."""
    ids: set[int] = set()
    for ref in (evidence_refs or []):
        parts = ref.split("|")
        if len(parts) >= 3:
            try:
                ids.add(int(parts[2]))
            except ValueError:
                pass
    return ids


def _is_boilerplate_section(path_string: str) -> bool:
    """Check if a section path looks like non-extractable boilerplate."""
    return bool(_SKIP_SECTION_PATTERNS.search(path_string))


def sample_paragraphs(
    session: Any,
    pmcid: str,
    run_id: int,
    n_with_extraction: int,
    n_zero_extraction: int,
    seed: int,
    min_words: int = _MIN_WORDS_DEFAULT,
) -> list[ParagraphSample]:
    """Sample paragraphs from both strata for Q3/Q5.

    With-extraction stratum: paragraphs where MAP extracted at least one finding.
    Zero-extraction stratum: paragraphs with no matched findings, filtered to
    exclude boilerplate sections (acknowledgements, funding, etc.) and very
    short paragraphs.
    """
    from nlp_histo.database import SumMapFinding, TextElement, Document  # noqa: PLC0415

    doc = session.query(Document).filter_by(pmcid=pmcid).first()
    if doc is None:
        logger.warning("[%s] No document found", pmcid)
        return []

    # Get all findings for this run
    findings = (
        session.query(SumMapFinding)
        .filter_by(pmcid=pmcid, pipeline_run_id=run_id)
        .all()
    )

    # Map text_element_id → list of claims
    te_claims: dict[int, list[str]] = {}
    for f in findings:
        for te_id in _parse_te_ids(f.evidence_refs):
            te_claims.setdefault(te_id, []).append(f.claim)

    # Get all text elements for the paper
    all_tes = (
        session.query(TextElement)
        .filter_by(document_id=doc.id)
        .all()
    )
    te_by_id: dict[int, TextElement] = {te.id: te for te in all_tes}

    rng = random.Random(seed)
    results: list[ParagraphSample] = []

    # ── With-extraction stratum ──────────────────────────────────────────
    with_ids = [
        te_id for te_id in te_claims
        if te_id in te_by_id
        and te_by_id[te_id].text_content
        and (te_by_id[te_id].word_count or 0) >= min_words
    ]
    chosen_with = rng.sample(with_ids, min(n_with_extraction, len(with_ids)))
    for te_id in chosen_with:
        te = te_by_id[te_id]
        results.append(ParagraphSample(
            text_element_id=te_id,
            text_content=te.text_content,
            word_count=te.word_count or 0,
            stratum="with_extractions",
            extracted_claims=te_claims[te_id],
            section_path=te.path_string or "",
        ))

    # ── Zero-extraction stratum ──────────────────────────────────────────
    te_ids_with_findings = set(te_claims.keys())
    zero_ids = [
        te.id for te in all_tes
        if te.id not in te_ids_with_findings
        and te.text_content
        and (te.word_count or 0) >= min_words
        and not _is_boilerplate_section(te.path_string or "")
    ]
    chosen_zero = rng.sample(zero_ids, min(n_zero_extraction, len(zero_ids)))
    for te_id in chosen_zero:
        te = te_by_id[te_id]
        results.append(ParagraphSample(
            text_element_id=te_id,
            text_content=te.text_content,
            word_count=te.word_count or 0,
            stratum="zero_extractions",
            extracted_claims=[],
            section_path=te.path_string or "",
        ))

    logger.info(
        "[%s] Paragraph sample: %d with_extractions, %d zero_extractions",
        pmcid, len(chosen_with), len(chosen_zero),
    )
    return results
