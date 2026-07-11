"""CascadedCompositeScorer — hybrid NER+embedding scorer with LP-optimized thresholds."""
from __future__ import annotations

import logging
from pathlib import Path

from pipeline.stages.knowledge_extraction.interfaces.agreement import MapOutputScorer
from pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext, ChunkDecision, ScoreBundle
from pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)


class CascadedCompositeScorer:
    """
    Applies LP-optimized thresholds to hybrid (embedding + NER) features.

    Decision rule
    -------------
    KEEP    : emb >= keep_emb_threshold  AND  ner >= keep_ner_threshold
    REJECT  : emb <= reject_threshold    (and not KEEP)
    ESCALATE: everything else

    Thresholds are typically obtained from ThresholdOptimizer.fit() and
    persisted as JSON.  Use the ``from_file`` class method to load them.

    Parameters
    ----------
    hybrid_scorer:
        Scorer that populates both ScoreBundle.embedding_agreement and
        ScoreBundle.entity_overlap.  HybridScorer is the recommended choice.
    keep_emb_threshold:
        Minimum embedding score required to accept a chunk.
    keep_ner_threshold:
        Minimum NER entity-overlap score required to accept a chunk.
    reject_threshold:
        Maximum embedding score below which a chunk is rejected outright.
    """

    def __init__(
        self,
        hybrid_scorer: MapOutputScorer,
        keep_emb_threshold: float = 0.80,
        keep_ner_threshold: float = 0.50,
        reject_threshold: float = 0.20,
    ) -> None:
        self._hybrid = hybrid_scorer
        self.keep_emb_threshold = keep_emb_threshold
        self.keep_ner_threshold = keep_ner_threshold
        self.reject_threshold = reject_threshold

    @classmethod
    def from_file(
        cls,
        hybrid_scorer: MapOutputScorer,
        thresholds_path: str | Path,
    ) -> "CascadedCompositeScorer":
        """
        Build a scorer with thresholds loaded from an OptimizedThresholds JSON file.

        Parameters
        ----------
        hybrid_scorer:
            HybridScorer (or any MapOutputScorer) for feature computation.
        thresholds_path:
            Path to JSON file produced by OptimizedThresholds.save().
        """
        # Import here to avoid circular dependency at module level
        from pipeline.stages.knowledge_extraction.agreement.calibration.threshold_optimizer import (
            OptimizedThresholds,
        )

        t = OptimizedThresholds.load(thresholds_path)
        logger.info(
            "Loaded thresholds from %s: keep_emb=%.3f keep_ner=%.3f reject_emb=%.3f",
            thresholds_path,
            t.keep_emb,
            t.keep_ner,
            t.reject_emb,
        )
        return cls(
            hybrid_scorer=hybrid_scorer,
            keep_emb_threshold=t.keep_emb,
            keep_ner_threshold=t.keep_ner,
            reject_threshold=t.reject_emb,
        )

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
        context: AgreementContext | None = None,  # noqa: ARG002
    ) -> ScoreBundle:
        bundle = self._hybrid.compute(outputs, source_text)
        emb = bundle.embedding_agreement or 0.0
        ner = bundle.entity_overlap or 0.0

        if emb >= self.keep_emb_threshold and ner >= self.keep_ner_threshold:
            bundle.confidence = (emb + ner) / 2.0
            bundle.decision = ChunkDecision.KEEP
            logger.debug(
                "Cascade KEEP (emb=%.3f >= %.3f, ner=%.3f >= %.3f)",
                emb, self.keep_emb_threshold, ner, self.keep_ner_threshold,
            )
            return bundle

        if emb <= self.reject_threshold:
            bundle.confidence = emb
            bundle.decision = ChunkDecision.REJECT
            logger.debug(
                "Cascade REJECT (emb=%.3f <= %.3f)",
                emb, self.reject_threshold,
            )
            return bundle

        bundle.confidence = (emb + ner) / 2.0
        bundle.decision = ChunkDecision.ESCALATE
        logger.debug(
            "Cascade ESCALATE (emb=%.3f ner=%.3f)",
            emb, ner,
        )
        return bundle
