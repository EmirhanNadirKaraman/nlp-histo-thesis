"""HybridScorer — combines EmbeddingScorer and NERScorer into a single bundle."""
from __future__ import annotations

import logging

from pipeline.stages.knowledge_extraction.agreement.embedding import EmbeddingScorer
from pipeline.stages.knowledge_extraction.agreement.ner_scorer import NERScorer
from pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext, ScoreBundle
from pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)


class HybridScorer:
    """
    Computes both embedding soft-alignment and entity Jaccard overlap.

    Populates ScoreBundle.embedding_agreement (from EmbeddingScorer) and
    ScoreBundle.entity_overlap (from NERScorer).  Does NOT set a decision —
    the caller (e.g. CascadedCompositeScorer) applies threshold logic.

    Parameters
    ----------
    embedding_scorer:
        Scorer for claim embedding similarity.  Defaults to EmbeddingScorer().
    ner_scorer:
        Scorer for entity Jaccard overlap.  Defaults to NERScorer().
    """

    def __init__(
        self,
        embedding_scorer: EmbeddingScorer | None = None,
        ner_scorer: NERScorer | None = None,
    ) -> None:
        self._embedding = embedding_scorer or EmbeddingScorer()
        self._ner = ner_scorer or NERScorer()

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
        context: AgreementContext | None = None,  # noqa: ARG002
    ) -> ScoreBundle:
        emb_bundle = self._embedding.compute(outputs, source_text)
        ner_bundle = self._ner.compute(outputs, source_text)

        bundle = ScoreBundle(
            embedding_agreement=emb_bundle.embedding_agreement,
            entity_overlap=ner_bundle.entity_overlap,
        )

        logger.debug(
            "HybridScorer emb=%.3f ner=%.3f",
            bundle.embedding_agreement or 0.0,
            bundle.entity_overlap or 0.0,
        )
        return bundle
