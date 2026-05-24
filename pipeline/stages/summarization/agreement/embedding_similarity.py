"""EmbeddingSimilarityStrategy — batched claim embedding similarity."""
from __future__ import annotations

import numpy as np

from pipeline.stages.summarization.interfaces.scoring import AgreementContext
from pipeline.stages.summarization.models import AuditableSummary
from .embedding import _align, _align_precomputed_with_breakdown, _claims
from .providers import EmbedFn, OpenAIEmbedder


class EmbeddingSimilarityStrategy:
    """
    Pairwise soft-alignment cosine similarity on claim embeddings.

    Wraps ``_align()`` for single-pair use and adds ``compute_matrix()`` for
    batched N-voter evaluation that issues a single embedding API call regardless
    of N.  Both paths apply the same penalty set as EmbeddingScorer: weak-match
    threshold, count mismatch, reuse concentration, and contradiction.
    When an ``AgreementContext`` is passed to ``compute_matrix()``, a per-pair
    grounding factor is applied using ``min(pf_i, pf_j)`` for each matrix entry.

    Parameters
    ----------
    embed_fn:
        Callable that maps a list of strings to embedding vectors.
        Defaults to OpenAIEmbedder (text-embedding-3-small).
        The embedder is responsible for its own batching internally.
    tau:
        Weak-match threshold.  Default 0.15.
    count_alpha:
        Count-mismatch penalty exponent.  Default 0.25.
    reuse_weight:
        Reuse-concentration penalty weight.  Default 0.15.
    contradiction_weight:
        Contradiction penalty weight.  Default 0.20.
    grounding_floor:
        Minimum per-pair grounding factor when AgreementContext is provided.
        Default 0.50.
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        tau: float = 0.15,
        count_alpha: float = 0.25,
        reuse_weight: float = 0.15,
        contradiction_weight: float = 0.20,
        grounding_floor: float = 0.50,
    ) -> None:
        self._embed: EmbedFn = embed_fn or OpenAIEmbedder()
        self._tau = tau
        self._count_alpha = count_alpha
        self._reuse_weight = reuse_weight
        self._contradiction_weight = contradiction_weight
        self._grounding_floor = grounding_floor

    @classmethod
    def from_config(cls, agreement_cfg, embed_fn: EmbedFn | None = None) -> "EmbeddingSimilarityStrategy":
        """Build from an ``AgreementConfig`` (H-EMB-01). Single mapping site for
        the soft-alignment weights; ``grounding_floor`` keeps its class default."""
        return cls(
            embed_fn=embed_fn,
            tau=agreement_cfg.tau,
            count_alpha=agreement_cfg.count_alpha,
            reuse_weight=agreement_cfg.reuse_weight,
            contradiction_weight=agreement_cfg.contradiction_weight,
        )

    def similarity(self, a: AuditableSummary, b: AuditableSummary) -> float:
        """Single-pair similarity (for unit tests and single-call use)."""
        return _align(
            self._embed, _claims(a), _claims(b),
            tau=self._tau,
            count_alpha=self._count_alpha,
            reuse_weight=self._reuse_weight,
            contradiction_weight=self._contradiction_weight,
        )

    def compute_matrix(
        self,
        outputs: list[AuditableSummary],
        context: AgreementContext | None = None,
    ) -> np.ndarray:
        """
        Build the full N×N similarity matrix in a single embedding API call.

        All claims from all voters are concatenated, embedded in one request,
        then split back by voter.  ``_align_precomputed`` applies the full
        penalty set (tau, count, reuse, contradiction) using the pre-normalised
        embeddings and raw claim strings.

        If ``context`` is provided and ``len(context.voter_contexts) == N``,
        each off-diagonal entry ``M[i, j]`` is multiplied by a grounding factor
        derived from ``min(pf_i, pf_j)`` — the worst-grounded voter in that pair.

        Returns
        -------
        np.ndarray
            N×N float matrix where ``M[i, j]`` is the adjusted soft-alignment
            score between voter i and voter j.  Diagonal entries are 1.0.
        """
        n = len(outputs)
        claim_lists = [_claims(o) for o in outputs]
        all_claims = [c for claims in claim_lists for c in claims]

        if not all_claims:
            return np.eye(n)

        raw_embs = np.array(self._embed(all_claims), dtype=float)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        all_embs = raw_embs / norms

        voter_embs: list[np.ndarray] = []
        start = 0
        for claims in claim_lists:
            count = len(claims)
            voter_embs.append(all_embs[start : start + count])
            start += count

        matrix, _ = self._build_matrix_and_breakdowns(n, voter_embs, claim_lists, context)
        return matrix

    def compute_matrix_with_breakdown(
        self,
        outputs: list[AuditableSummary],
        context: AgreementContext | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Same as ``compute_matrix`` but also returns per-pair component breakdowns.

        Returns
        -------
        matrix:
            N×N float matrix (same as ``compute_matrix``).
        breakdowns:
            Upper-triangle list of dicts, one per (i, j) pair with i < j,
            in row-major order.  Each dict contains:

            ``voter_i``, ``voter_j``,
            ``claim_count_a``, ``claim_count_b``,
            ``coverage_a_to_b``, ``coverage_b_to_a``, ``base``,
            ``count_factor``, ``reuse_factor``,
            ``polarity_contradiction_ratio``, ``numeric_contradiction_ratio``,
            ``contradiction_ratio``, ``contradiction_factor``,
            ``pre_grounding_score``,
            ``grounding_factor``,   (1.0 when no context provided)
            ``score``               (final value stored in matrix[i,j])
        """
        n = len(outputs)
        claim_lists = [_claims(o) for o in outputs]
        all_claims = [c for claims in claim_lists for c in claims]

        if not all_claims:
            empty_bd = [
                {"voter_i": i, "voter_j": j, "score": 1.0, "grounding_factor": 1.0,
                 "pre_grounding_score": 1.0}
                for i in range(n) for j in range(i + 1, n)
            ]
            return np.eye(n), empty_bd

        raw_embs = np.array(self._embed(all_claims), dtype=float)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        all_embs = raw_embs / norms

        voter_embs: list[np.ndarray] = []
        start = 0
        for claims in claim_lists:
            voter_embs.append(all_embs[start : start + len(claims)])
            start += len(claims)

        matrix, breakdowns = self._build_matrix_and_breakdowns(
            n, voter_embs, claim_lists, context
        )
        return matrix, breakdowns

    # ── Shared internals ───────────────────────────────────────────────────────

    def _build_matrix_and_breakdowns(
        self,
        n: int,
        voter_embs: list[np.ndarray],
        claim_lists: list[list[str]],
        context: AgreementContext | None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Build N×N matrix and upper-triangle breakdown list in one pass."""
        matrix = np.eye(n)
        breakdowns: list[dict] = []

        for i in range(n):
            for j in range(i + 1, n):
                score, bd = _align_precomputed_with_breakdown(
                    voter_embs[i], voter_embs[j],
                    claim_lists[i], claim_lists[j],
                    tau=self._tau,
                    count_alpha=self._count_alpha,
                    reuse_weight=self._reuse_weight,
                    contradiction_weight=self._contradiction_weight,
                )

                grounding_factor = 1.0
                if context is not None and len(context.voter_contexts) == n:
                    g = min(
                        context.voter_contexts[i].grounding_pass_fraction,
                        context.voter_contexts[j].grounding_pass_fraction,
                    )
                    grounding_factor = self._grounding_floor + (1.0 - self._grounding_floor) * g
                    score *= grounding_factor

                matrix[i, j] = score
                matrix[j, i] = score

                bd["voter_i"] = i
                bd["voter_j"] = j
                bd["grounding_factor"] = round(grounding_factor, 6)
                bd["score"] = round(score, 6)
                breakdowns.append(bd)

        return matrix, breakdowns
