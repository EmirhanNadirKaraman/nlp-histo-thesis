"""Finding-level citation/provenance filter (B-080).

The production counterpart to the *dormant* router-only ``ProvenanceValidator``
gate. The legacy L1→L2→L3 cascade (``enable_router=False``, the production path)
runs **no** citation validation — a MAP finding can cite a sentence position or
``text_element_id`` that does not exist in its chunk and still ship. This filter
closes that gap by dropping such findings from the SELECTED chunk summary, in
every cascade path and at every escalation level (L1/L2/L3).

It reuses the exact ``ProvenanceValidator`` logic so the check is identical to
what the router would apply, then keeps only the *structural* citation failures
as hard drops:

  INVALID_SENTENCE_ID        — citation not parseable as ``S{n}|PMCID|te_id``
  NONEXISTENT_SOURCE         — ``S{n}`` out of range for the chunk
  CROSS_DOCUMENT_SOURCE_ERROR — cited PMCID != document PMCID
  INVALID_TEXT_ELEMENT_ID    — cited ``te_id`` != the te_id at that sentence

These are *exact* integer/string comparisons against the chunk source index, so
they carry no false-positive risk. The fuzzy ``FABRICATED_VERBATIM_SUPPORT``
check (verbatim quote unrecognisable in the cited sentence) is opt-in via
``check_verbatim`` — off by default because it is a SequenceMatcher ratio, not an
exact match, and the grounding NLI filter already covers claim↔support quality.

ORDERING CONTRACT — must run BEFORE ``replace_verbatim_from_db``. The fabrication
check fuzzy-matches ``verbatim_support`` against the cited single *source
sentence*; after DB replacement ``verbatim_support`` is the full cited
*paragraph* and a sentence-vs-paragraph ratio would mis-fire. Both call sites
(``MapStage._cascade`` and the offline ``_replay``) validate pre-replacement.
"""
from __future__ import annotations

import logging

from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary, Finding
from nlp_histo.pipeline.stages.knowledge_extraction.validation.models import ReasonCode
from nlp_histo.pipeline.stages.knowledge_extraction.provenance.validator import (
    ProvenanceValidator,
    SourceIndex,
)

logger = logging.getLogger(__name__)

# Exact structural citation failures — always dropped when the filter is on.
STRUCTURAL_CITATION_CODES = frozenset({
    ReasonCode.INVALID_SENTENCE_ID,
    ReasonCode.NONEXISTENT_SOURCE,
    ReasonCode.CROSS_DOCUMENT_SOURCE_ERROR,
    ReasonCode.INVALID_TEXT_ELEMENT_ID,
})

# Fuzzy verbatim failure — dropped only when ``check_verbatim=True``.
_FABRICATED_CODE = ReasonCode.FABRICATED_VERBATIM_SUPPORT

DEFAULT_FABRICATED_THRESHOLD = 0.25


def citation_drop_indices(
    summary: AuditableSummary,
    chunk: list[dict],
    pmcid: str,
    *,
    check_verbatim: bool = False,
    fabricated_threshold: float = DEFAULT_FABRICATED_THRESHOLD,
) -> set[int]:
    """Return the set of ``summary.findings`` indices whose citation hard-fails.

    The single source of truth for the drop decision, shared by
    :func:`filter_summary_by_citation` (live runners, Finding-level) and the
    offline replay (index-filters the raw cached dicts). Empty set when there
    are no findings or ``chunk`` is empty/missing — a missing source index must
    never be read as "every citation is invalid".
    """
    if not summary.findings:
        return set()
    if not chunk:
        logger.warning(
            "citation filter: empty chunk for pmcid=%s — skipping (kept %d findings)",
            pmcid, len(summary.findings),
        )
        return set()

    drop_codes = set(STRUCTURAL_CITATION_CODES)
    if check_verbatim:
        drop_codes.add(_FABRICATED_CODE)
        fab_threshold = fabricated_threshold
    else:
        # 0.0 → the validator's ``best_ratio < fab_threshold`` is never true,
        # so FABRICATED is never emitted (and WEAK is suppressed below too).
        fab_threshold = 0.0

    validator = ProvenanceValidator(
        source_index=SourceIndex(chunk),
        document_pmcid=pmcid,
        fabricated_threshold=fab_threshold,
        weak_threshold=0.0,  # suppress soft WEAK_GROUNDING — only hard codes drop here
    )
    validations = validator.validate(summary)
    return {
        i for i, val in enumerate(validations)
        if any(code in drop_codes for code in val.reason_codes)
    }


def filter_summary_by_citation(
    summary: AuditableSummary,
    chunk: list[dict],
    pmcid: str,
    *,
    check_verbatim: bool = False,
    fabricated_threshold: float = DEFAULT_FABRICATED_THRESHOLD,
) -> tuple[AuditableSummary, list[Finding]]:
    """Drop findings with hard citation failures from ``summary`` **in place**.

    Returns ``(summary, dropped)``. ``summary.findings`` is replaced with the
    kept subset; ``dropped`` is the list of removed Findings (for logging /
    audit). No-op (keeps everything) when the summary has no findings or when
    ``chunk`` is empty/missing.
    """
    drop = citation_drop_indices(
        summary, chunk, pmcid,
        check_verbatim=check_verbatim,
        fabricated_threshold=fabricated_threshold,
    )
    if not drop:
        return summary, []
    kept: list[Finding] = []
    dropped: list[Finding] = []
    for i, finding in enumerate(summary.findings):
        (dropped if i in drop else kept).append(finding)
    summary.findings = kept
    return summary, dropped
