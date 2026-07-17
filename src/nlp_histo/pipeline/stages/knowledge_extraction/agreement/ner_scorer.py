"""NERScorer — pairwise entity Jaccard overlap using scispaCy.

Delegates model loading to ``pipeline.stages.knowledge_extraction.entities.umls_resources`` so
the scispaCy model is loaded at most once per process.  Previously this module
held its own copy of ``en_core_sci_lg``; combined with the NORMALIZE/UMLS
linker load that meant ~2 copies of the model resident at once, which OOM-killed
runs at the post-CANONICALIZE UMLS enrichment step.
"""
from __future__ import annotations

import logging
from itertools import combinations

from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext, ScoreBundle
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)


def _get_nlp():
    """Return the shared scispaCy nlp (or None when UMLS is unavailable)."""
    from nlp_histo.pipeline.stages.knowledge_extraction.entities.umls_resources import get_nlp  # noqa: PLC0415
    return get_nlp()


def _extract_entities(claims: list[str]) -> frozenset[str]:
    """Extract lowercased entity spans from a list of claim strings."""
    if not claims:
        return frozenset()
    nlp = _get_nlp()
    if nlp is None:
        return frozenset()
    text = " ".join(claims)
    # Skip the linker — NER scorer only needs ``doc.ents``, and running the
    # scispacy_linker on every voter pair during MAP would be wasted work.
    if "scispacy_linker" in nlp.pipe_names:
        with nlp.select_pipes(disable=["scispacy_linker"]):
            doc = nlp(text)
    else:
        doc = nlp(text)
    return frozenset(ent.text.lower() for ent in doc.ents)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NERScorer:
    """
    Measures voter agreement via pairwise biomedical entity Jaccard overlap.

    For each voter output, entities are extracted from the claim strings using
    scispaCy (en_core_sci_lg).  Mean pairwise Jaccard similarity is stored in
    ScoreBundle.entity_overlap.

    The scispaCy model is lazy-loaded on first use and shared across instances.
    """

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,  # noqa: ARG002
        context: AgreementContext | None = None,  # noqa: ARG002
    ) -> ScoreBundle:
        entity_sets = [
            _extract_entities([f.claim for f in o.findings]) for o in outputs
        ]

        if len(entity_sets) < 2:
            return ScoreBundle(entity_overlap=1.0)

        scores = [_jaccard(a, b) for a, b in combinations(entity_sets, 2)]
        mean_score = sum(scores) / len(scores)

        logger.debug("NERScorer entity_overlap=%.3f", mean_score)
        return ScoreBundle(entity_overlap=mean_score)
