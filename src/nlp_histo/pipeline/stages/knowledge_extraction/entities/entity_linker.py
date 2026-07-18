"""
UMLS CUI enrichment for CanonicalRules.

Delegates model loading to ``pipeline.stages.knowledge_extraction.entities.umls_resources`` so
the scispaCy + UMLS linker is loaded at most once per process.  Loading the
~5 GB UMLS knowledge base twice was the most reliable way to OOM-kill a run
mid-pipeline; this module exists to keep that from happening again.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .umls_resources import get_nlp, get_linker, umls_disabled, _log_memory
from .umls_utils import best_cui as _best_cui

_SKIP_ENV = "NLP_HISTO_SKIP_UMLS_ENRICHMENT"

if TYPE_CHECKING:
    from ..models import CanonicalRule

logger = logging.getLogger(__name__)


def _best_cui_from_doc(doc, kb) -> str | None:
    """Extract the highest-scoring non-junk CUI from a spacy Doc."""
    spans = doc.ents if doc.ents else [doc]
    kb_ents_by_span = [getattr(span._, "kb_ents", None) or [] for span in spans]
    return _best_cui(kb_ents_by_span, kb, doc.text)


def enrich_rules_with_cuis(
    rules: list[CanonicalRule],
    *,
    batch_size: int = 32,
) -> None:
    """Fill in subject_cui / outcome_cui on rules in-place.

    Skips rules that already have both CUIs set (typically propagated from
    NORMALIZE).  Silently no-ops when the linker is unavailable or has been
    disabled via the env var / CLI flag.
    """
    if os.environ.get(_SKIP_ENV, "").lower() in {"1", "true", "yes"}:
        logger.info(
            "CUI enrichment skipped via $%s (rules pass through unmodified)",
            _SKIP_ENV,
        )
        return
    rules_to_enrich = [r for r in rules if not (r.subject_cui and r.outcome_cui)]
    if not rules_to_enrich:
        logger.info(
            "CUI enrichment skipped: all %d rules already enriched", len(rules),
        )
        return
    if umls_disabled():
        logger.info("CUI enrichment skipped: UMLS disabled (env / CLI flag)")
        return

    _log_memory("before UMLS enrichment")
    nlp = get_nlp()
    if nlp is None:
        return
    linker = get_linker()
    kb = linker.kb

    texts: list[str] = list({
        t
        for r in rules_to_enrich
        for t in (r.subject_entity, r.outcome_entity)
        if t
    })
    if not texts:
        return

    logger.info(
        "CUI enrichment: %d unique entity strings across %d rules (batch_size=%d)",
        len(texts), len(rules_to_enrich), batch_size,
    )

    cui_map: dict[str, str | None] = {}
    try:
        for doc in nlp.pipe(texts, batch_size=batch_size):
            cui_map[doc.text] = _best_cui_from_doc(doc, kb)
    except Exception as exc:
        logger.warning("CUI enrichment batch failed: %s", exc)
        return

    for rule in rules_to_enrich:
        if not rule.subject_cui:
            rule.subject_cui = cui_map.get(rule.subject_entity)
        if not rule.outcome_cui:
            rule.outcome_cui = cui_map.get(rule.outcome_entity)

    enriched = sum(1 for r in rules if r.subject_cui or r.outcome_cui)
    logger.info("CUI enrichment: %d/%d rules enriched", enriched, len(rules))
    _log_memory("after UMLS enrichment")
