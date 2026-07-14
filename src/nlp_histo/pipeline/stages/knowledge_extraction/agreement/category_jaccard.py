"""CategoryJaccardScorer — fast, API-free agreement on category sets."""
from __future__ import annotations

from itertools import combinations

from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext, ScoreBundle
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary


class CategoryJaccardScorer:
    """
    Mean pairwise Jaccard similarity on the category sets each voter extracted.

    Fast and deterministic — requires no API calls.  Less sensitive than
    EmbeddingScorer: voters can disagree on claim content while agreeing on
    categories, scoring 1.0 even when claims differ significantly.  Best
    used as a cheap pre-filter or in resource-constrained settings.

    Score is stored in ScoreBundle.embedding_agreement for compatibility
    with the AgreementChecker fallback threshold logic.
    """

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,  # noqa: ARG002
        context: AgreementContext | None = None,  # noqa: ARG002
    ) -> ScoreBundle:
        cat_sets = [_category_set(o) for o in outputs]
        scores = [_jaccard(a, b) for a, b in combinations(cat_sets, 2)]
        score = sum(scores) / len(scores) if scores else 1.0
        return ScoreBundle(embedding_agreement=score)


# ── Pure helpers ───────────────────────────────────────────────────────────────

def _category_set(output: AuditableSummary) -> frozenset[str]:
    return frozenset(f.category for f in output.findings)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0
