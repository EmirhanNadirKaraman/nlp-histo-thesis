"""AgreementChecker — thin orchestrator: scorer + theta fallback + best()."""
from __future__ import annotations

import logging
from typing import Literal

from pipeline.stages.summarization.agreement.polarity_conflict import (
    detect_polarity_conflict,
)
from pipeline.stages.summarization.interfaces.agreement import MapOutputScorer
from pipeline.stages.summarization.interfaces.scoring import (
    AgreementContext,
    ChunkDecision,
    ScoreBundle,
)
from pipeline.stages.summarization.models import AuditableSummary

logger = logging.getLogger(__name__)


class AgreementChecker:
    """
    Wraps a MapOutputScorer with a theta fallback and the best() helper.

    Responsibilities
    ----------------
    - Calls scorer.compute() and ensures ScoreBundle.decision is always set.
    - When the scorer does not set decision (leaf scorers like EmbeddingScorer,
      CategoryJaccardScorer), applies the theta threshold against the primary
      score field (confidence → embedding_agreement → 0.0).
    - CascadedCompositeScorer sets decision internally; theta is ignored.

    Parameters
    ----------
    scorer:
        Any object satisfying the MapOutputScorer protocol.
    theta:
        Fallback threshold.  Only used when scorer does not set a decision.
        ``theta=0.7`` is calibrated for ``EmbeddingScorer`` (embedding cosine
        scores).  When using ``SemanticAgreementScorer`` with a different
        strategy, either pass a recalibrated ``theta`` here or set ``theta``
        on ``SemanticAgreementScorer`` directly so the scorer controls its
        own decision boundary.
    single_voter_policy:
        What to do when only one voter is available for peer comparison
        (e.g. the others' API calls failed). ``"keep"`` (default) preserves
        the prior implicit behaviour — the lone survivor is accepted with
        synthetic ``confidence=1.0`` and no peer comparison. ``"escalate"``
        treats N=1 as low-evidence and returns
        ``ChunkDecision.ESCALATE`` so the caller routes the chunk up the
        cascade. Mirror of ``MapOutputRouter.single_voter_policy`` for the
        legacy non-router path; configured at runner level via
        ``RoutingConfig.legacy_single_voter_policy``.
    """

    def __init__(
        self,
        scorer: MapOutputScorer,
        theta: float = 0.7,
        reject_theta: float = 0.2,
        single_voter_policy: Literal["keep", "escalate"] = "keep",
    ) -> None:
        self._scorer = scorer
        self.theta = theta
        self.reject_theta = reject_theta
        self._single_voter_policy = single_voter_policy

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
        context: AgreementContext | None = None,
    ) -> ScoreBundle:
        """Run the scorer and guarantee ScoreBundle.decision is populated."""
        if len(outputs) == 0:
            return ScoreBundle(confidence=0.0, decision=ChunkDecision.ESCALATE)
        if len(outputs) < 2:
            # Only one voter — no peer comparison is possible. Policy decides
            # whether to accept the lone survivor (legacy default) or escalate.
            # Note: when the router is active, N=1 is already short-circuited
            # by MapOutputRouter._chunk_decision_from_classifications before the
            # agreement gate is reached, so this branch only fires on the
            # legacy path (where N=1 means API survival, not validator pass).
            if self._single_voter_policy == "escalate":
                return ScoreBundle(confidence=0.0, decision=ChunkDecision.ESCALATE)
            return ScoreBundle(confidence=1.0, decision=ChunkDecision.KEEP)

        bundle = self._scorer.compute(outputs, source_text, context)

        # B-051: hard-fail veto for comparable opposite-polarity findings.
        # Runs AFTER the scorer so pairwise_upper / embedding_agreement remain
        # available for trace inspection, and BEFORE the theta fallback so the
        # decision cannot revert to KEEP regardless of similarity.
        conflict = detect_polarity_conflict(outputs)
        if conflict is not None:
            bundle.decision = ChunkDecision.ESCALATE
            details = bundle.score_details if bundle.score_details is not None else {}
            details = dict(details)
            details["hard_fail_reason"] = "polarity_conflict"
            details["polarity_conflict_details"] = conflict
            bundle.score_details = details
            logger.info(
                "AgreementChecker [%s] hard-fail → ESCALATE "
                "(polarity_conflict on %d pair(s); embedding_agreement=%s)",
                type(self._scorer).__name__,
                conflict["count"],
                _fmt(bundle.embedding_agreement),
            )
            return bundle

        if bundle.decision is None:
            primary = bundle.confidence or bundle.embedding_agreement or 0.0
            if primary >= self.theta:
                bundle.decision = ChunkDecision.KEEP
            elif primary <= self.reject_theta:
                bundle.decision = ChunkDecision.REJECT
            else:
                bundle.decision = ChunkDecision.ESCALATE

        logger.debug(
            "AgreementChecker [%s] emb=%s judge=%s conf=%s → %s",
            type(self._scorer).__name__,
            _fmt(bundle.embedding_agreement),
            _fmt(bundle.judge_agreement),
            _fmt(bundle.confidence),
            bundle.decision.value,
        )
        return bundle

    def best(
        self,
        outputs: list[AuditableSummary],
        bundle: ScoreBundle | None = None,
    ) -> AuditableSummary:
        """
        Return the best voter output.

        If ``bundle.best_index`` is set (e.g. from SemanticAgreementScorer),
        return that candidate directly.  Otherwise fall back to a quality key
        based on mean evidence chain length and finding count — both externally
        validated signals that are more reliable than self-reported confidence.
        """
        if bundle is not None and bundle.best_index is not None:
            if bundle.best_index < len(outputs):
                return outputs[bundle.best_index]
            logger.error(
                "best_index=%d out of bounds for outputs list of length %d — falling back",
                bundle.best_index,
                len(outputs),
            )
        return max(outputs, key=_quality_key)


def _quality_key(o: AuditableSummary) -> tuple:
    """Fallback quality signal for best() when no bundle.best_index is available."""
    mean_ev = (
        sum(len(f.evidence) for f in o.findings) / len(o.findings)
        if o.findings else 0.0
    )
    return (mean_ev, len(o.findings))


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"
