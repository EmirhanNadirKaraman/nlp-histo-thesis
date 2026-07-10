"""
SemanticAgreementScorer — paper-based max-consensus agreement scoring.

Implements the agreement model from
"Semantic Agreement Enables Efficient Open-Ended LLM Cascades":

  1. Build N×N pairwise similarity matrix over ELIGIBLE voter outputs only.
  2. Per-candidate average similarity = mean of off-diagonal row scores.
  3. Deferral score = max(avg_sim).
  4. Best candidate = argmax(avg_sim), tie-broken by grounding quality.

Empty voters (0 findings) receive avg_sim=0.0 and are ineligible for selection.

Grounding quality for tie-breaking is read from AgreementContext when provided
by the router, or derived directly from AuditableSummary as a fallback.
"""
from __future__ import annotations

import logging

import numpy as np

from pipeline.stages.knowledge_extraction.interfaces.scoring import (
    AgreementContext,
    ChunkDecision,
    ScoreBundle,
    VoterContext,
)
from pipeline.stages.knowledge_extraction.interfaces.similarity import SimilarityStrategy
from pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)


def _pass_frac(output: AuditableSummary) -> float:
    """
    **Fallback signal only** — used when no ``AgreementContext`` is available (legacy path).

    Returns the fraction of findings with a non-empty evidence list.  This is a
    structural proxy, not a validator result: a finding can have a non-empty evidence
    chain and still fail provenance checks.  For accurate grounding quality, use the
    router path which provides ``AgreementContext`` populated from
    ``ProvenanceValidator`` results.
    """
    if not output.findings:
        return 1.0
    return sum(1 for f in output.findings if f.evidence) / len(output.findings)


def _mean_ev(output: AuditableSummary) -> float:
    """Mean len(Finding.evidence) across findings (fallback signal)."""
    if not output.findings:
        return 0.0
    return sum(len(f.evidence) for f in output.findings) / len(output.findings)


class SemanticAgreementScorer:
    """
    Max-consensus agreement scorer.

    Parameters
    ----------
    strategy:
        SimilarityStrategy used to score voter pairs.
        Strategies that implement ``compute_matrix()`` are preferred; the
        scorer detects and uses that method to batch similarity computation.
    """

    def __init__(self, strategy: SimilarityStrategy, theta: float | None = None) -> None:
        self._strategy = strategy
        self._theta = theta  # scorer-level override; None = defer to AgreementChecker

    @classmethod
    def from_agreement_config(
        cls,
        agreement_cfg,
        embed_fn=None,
        theta: float | None = None,
    ) -> "SemanticAgreementScorer":
        """Build the scorer with the strategy selected by ``agreement_cfg.scorer_kind``.

        Replaces the hardcoded ``EmbeddingSimilarityStrategy.from_config(...)``
        callsites in ``KnowledgeExtractionRunner`` and ``BatchKnowledgeExtractionRunner``
        so the production strategy can be pinned from YAML. Both strategies
        consume the soft-alignment weights (``tau`` / ``count_alpha`` /
        ``reuse_weight`` / ``contradiction_weight``) from the same
        ``AgreementConfig``; the hybrid blend weights (``w_category`` etc.)
        are not yet config-driven — they stay at the ``HybridStructuredSimilarity``
        constructor defaults until a follow-up promotion lands.

        Raises ``ValueError`` for any ``scorer_kind`` outside
        ``{"embedding", "hybrid"}`` — defensive even though the dataclass
        default is ``"embedding"``: a YAML override or a programmatic
        ``AgreementConfig(scorer_kind="…")`` typo gets caught here, before
        the runner builds the cascade.
        """
        from .embedding_similarity import EmbeddingSimilarityStrategy
        from .hybrid_structured import HybridStructuredSimilarity

        kind = agreement_cfg.scorer_kind
        if kind == "embedding":
            strategy = EmbeddingSimilarityStrategy.from_config(
                agreement_cfg, embed_fn=embed_fn,
            )
        elif kind == "hybrid":
            strategy = HybridStructuredSimilarity.from_config(
                agreement_cfg, embed_fn=embed_fn,
            )
        else:
            raise ValueError(
                f"Invalid AgreementConfig.scorer_kind={kind!r}; "
                "expected one of {'embedding', 'hybrid'}"
            )
        return cls(strategy=strategy, theta=theta)

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,  # noqa: ARG002
        context: AgreementContext | None = None,
    ) -> ScoreBundle:
        """
        Compute deferral score and best candidate index.

        Parameters
        ----------
        outputs:
            ELIGIBLE voter AuditableSummary objects.
        source_text:
            Unused; kept for interface symmetry with MapOutputScorer.
        context:
            Optional grounding quality metadata from the router.
            When provided, ``len(context.voter_contexts)`` must equal
            ``len(outputs)``; a ``ValueError`` is raised otherwise.
            When absent, quality signals are derived from ``outputs`` directly.

        Returns
        -------
        ScoreBundle
            ``confidence`` = deferral score (max avg_sim).
            ``best_index`` = index into ``outputs`` of the selected candidate.
            ``decision`` is not set; the router / AgreementChecker applies theta.
        """
        n = len(outputs)

        if n == 0:
            return ScoreBundle(confidence=0.0, decision=ChunkDecision.REJECT)

        if n == 1:
            return ScoreBundle(
                confidence=1.0,
                embedding_agreement=1.0,
                decision=ChunkDecision.KEEP,
                best_index=0,
            )

        # Resolve grounding quality: context (validated) or AuditableSummary fallback.
        if context is not None:
            if len(context.voter_contexts) != n:
                raise ValueError(
                    f"AgreementContext has {len(context.voter_contexts)} voter_context(s) "
                    f"but outputs has {n} entries — they must be the same length."
                )
            quality = [
                (vc.grounding_pass_fraction, vc.mean_evidence_length)
                for vc in context.voter_contexts
            ]
        else:
            quality = [(_pass_frac(o), _mean_ev(o)) for o in outputs]

        # Exclude empty voters (0 findings) before matrix construction.
        # Including them would dilute non-empty voters' avg_sim scores.
        non_empty_idx = [i for i in range(n) if outputs[i].findings]
        empty_idx = [i for i in range(n) if not outputs[i].findings]

        if empty_idx:
            logger.debug(
                "SemanticAgreementScorer: excluding %d empty voter(s) at %s from matrix",
                len(empty_idx),
                empty_idx,
            )

        if len(non_empty_idx) < 2:
            logger.debug(
                "SemanticAgreementScorer: fewer than 2 non-empty voters (%d) — escalating",
                len(non_empty_idx),
            )
            return ScoreBundle(confidence=0.0, decision=ChunkDecision.ESCALATE)

        outputs_sub = [outputs[i] for i in non_empty_idx]
        quality_sub = [quality[i] for i in non_empty_idx]

        # Build similarity matrix over non-empty voters only (batched if supported).
        # Forward a filtered AgreementContext so EmbeddingSimilarityStrategy can
        # apply per-pair grounding penalties aligned with outputs_sub.
        pair_breakdowns: list[dict] = []
        ctx_sub: AgreementContext | None = None
        if context is not None:
            ctx_sub = AgreementContext(voter_contexts=[
                VoterContext(grounding_pass_fraction=pf, mean_evidence_length=me)
                for pf, me in quality_sub
            ])

        if hasattr(self._strategy, "compute_matrix_with_breakdown"):
            matrix, pair_breakdowns = self._strategy.compute_matrix_with_breakdown(
                outputs_sub, context=ctx_sub
            )
        elif hasattr(self._strategy, "compute_matrix"):
            matrix = self._strategy.compute_matrix(outputs_sub, context=ctx_sub)
        else:
            matrix = self._build_matrix_pairwise(outputs_sub)

        m = len(non_empty_idx)
        avg_sim = np.zeros(m)
        for i in range(m):
            other_cols = [matrix[i, j] for j in range(m) if j != i]
            avg_sim[i] = float(np.mean(other_cols))

        deferral_score = float(avg_sim.max())
        best_in_sub = self._select_best(avg_sim, quality_sub, outputs_sub)
        best_index = non_empty_idx[best_in_sub]

        logger.debug(
            "SemanticAgreementScorer: non_empty=%s avg_sim=%s deferral=%.3f best=%d",
            non_empty_idx,
            [f"{s:.3f}" for s in avg_sim],
            deferral_score,
            best_index,
        )

        # Build pairwise_upper: merge breakdown data when available.
        # pair_breakdowns uses sub-indices (0..m-1); remap to global voter indices.
        bd_by_subpair: dict[tuple[int, int], dict] = {
            (bd["voter_i"], bd["voter_j"]): bd for bd in pair_breakdowns
        }
        pairwise_upper = []
        for i in range(m):
            for j in range(i + 1, m):
                entry: dict = {
                    "voter_i": non_empty_idx[i],
                    "voter_j": non_empty_idx[j],
                    "score": round(float(matrix[i, j]), 6),
                }
                sub_bd = bd_by_subpair.get((i, j))
                if sub_bd is not None:
                    entry.update({
                        k: sub_bd[k] for k in (
                            "claim_count_a", "claim_count_b",
                            "coverage_a_to_b", "coverage_b_to_a", "base",
                            "count_factor", "reuse_factor",
                            "polarity_contradiction_ratio", "numeric_contradiction_ratio",
                            "contradiction_ratio", "contradiction_factor",
                            "pre_grounding_score", "grounding_factor",
                        ) if k in sub_bd
                    })
                pairwise_upper.append(entry)

        bundle = ScoreBundle(
            confidence=deferral_score,
            embedding_agreement=deferral_score,
            best_index=best_index,
            score_details={
                "eligible_voter_indices": non_empty_idx,
                "avg_sim": avg_sim.tolist(),
                "pairwise_upper": pairwise_upper,
            },
        )

        if self._theta is not None:
            bundle.decision = (
                ChunkDecision.KEEP if deferral_score >= self._theta else ChunkDecision.ESCALATE
            )

        return bundle

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _build_matrix_pairwise(
        self, outputs: list[AuditableSummary]
    ) -> np.ndarray:
        n = len(outputs)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                s = self._strategy.similarity(outputs[i], outputs[j])
                matrix[i, j] = s
                matrix[j, i] = s
        return matrix

    def _select_best(
        self,
        avg_sim: np.ndarray,
        quality: list[tuple[float, float]],
        outputs: list[AuditableSummary],
    ) -> int:
        """
        argmax(avg_sim) restricted to non-empty voters, tie-broken by quality.

        Empty voters (0 findings) are excluded from candidacy.  If all voters
        are empty (degenerate case), fall back to plain argmax.

        Tie-break order:
          1. grounding_pass_fraction (higher = better validated)
          2. mean_evidence_length    (more evidence = more grounded)
          3. len(findings)           (final fallback)
        """
        non_empty = [i for i in range(len(avg_sim)) if outputs[i].findings]

        if not non_empty:
            return int(np.argmax(avg_sim))

        max_score = max(float(avg_sim[i]) for i in non_empty)
        candidates = [i for i in non_empty if avg_sim[i] == max_score]

        if len(candidates) == 1:
            return candidates[0]

        def _key(i: int) -> tuple:
            pass_frac, mean_ev = quality[i]
            return (pass_frac, mean_ev, len(outputs[i].findings))

        return max(candidates, key=_key)
