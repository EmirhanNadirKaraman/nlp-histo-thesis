"""
LexicalSimilarityStrategy — trivial word-overlap similarity.

IMPORTANT: This is a debugging baseline ONLY.
Not suitable for thesis evaluation or production use.
Use EmbeddingSimilarityStrategy or HybridStructuredSimilarity for real workloads.
"""
from __future__ import annotations

from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary


class LexicalSimilarityStrategy:
    """
    Trivial debugging baseline.  Not suitable for thesis evaluation or production use.

    Computes the Jaccard similarity of lowercased word sets extracted from
    all claims in each voter output.  Two voters that mention the same words
    in any order will score 1.0; entirely disjoint vocabularies score 0.0.
    """

    def similarity(self, a: AuditableSummary, b: AuditableSummary) -> float:
        words_a = frozenset(" ".join(f.claim for f in a.findings).lower().split())
        words_b = frozenset(" ".join(f.claim for f in b.findings).lower().split())
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
