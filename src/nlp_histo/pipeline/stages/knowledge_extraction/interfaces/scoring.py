"""Scoring DTOs shared across the agreement interface and its implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChunkDecision(str, Enum):
    """Decision produced by a MapOutputScorer for a single MAP chunk."""

    KEEP = "keep"        # Accept best voter output; skip escalation LLM.
    REJECT = "reject"    # Voter outputs unusable; escalation LLM re-processes from scratch.
    ESCALATE = "escalate"  # Hand off to the escalation LLM.


@dataclass
class VoterContext:
    """
    Grounding quality signals for one voter, indexed parallel to ``outputs``.

    Passed to ``MapOutputScorer.compute()`` via ``AgreementContext`` so scorers
    can use externally-validated signals for candidate selection instead of
    relying on self-reported confidence.

    Current fields
    --------------
    grounding_pass_fraction:
        Fraction of findings that passed all provenance checks (0.0–1.0).
        Computed by ``MapOutputRouter._classify_voters()``.
    mean_evidence_length:
        Mean ``len(Finding.evidence)`` across all findings in this voter output.

    Extensibility
    -------------
    Add future tie-break signals as optional fields with ``None`` defaults so
    callers that do not yet supply them remain valid.  The ``extra_signals``
    dict is available for ad-hoc named signals that do not yet warrant a
    dedicated field.
    """

    grounding_pass_fraction: float
    mean_evidence_length: float
    # Reserved extension point — add future tie-break signals here or below.
    extra_signals: dict[str, float] | None = field(default=None)


@dataclass
class AgreementContext:
    """
    Optional grounding metadata passed alongside ``outputs`` to
    ``MapOutputScorer.compute()``.

    ``voter_contexts[i]`` describes ``outputs[i]``.  The router validates
    ``len(voter_contexts) == len(outputs)`` before constructing this object,
    so scorers can assume the two lists are aligned.
    """

    voter_contexts: list[VoterContext]


@dataclass
class ScoreBundle:
    """
    Aggregates all scorer outputs for a single MAP chunk.

    Individual scorers populate only their own field(s) and leave the rest as
    None.  A composite scorer (e.g. CascadedCompositeScorer) combines them and
    sets ``confidence`` and ``decision``.

    AgreementChecker applies a fallback theta decision when ``decision`` is
    still None after the scorer returns.
    """

    embedding_agreement: float | None = None
    judge_agreement: float | None = None
    entity_overlap: float | None = None
    confidence: float | None = None
    decision: ChunkDecision | None = None
    best_index: int | None = None  # index into the voter list of the selected candidate
    # Populated by SemanticAgreementScorer for the observability layer.
    # Keys: "eligible_voter_indices", "avg_sim", "pairwise_upper"
    # (list of {"voter_i", "voter_j", "score"} for the upper triangle).
    score_details: dict | None = field(default=None)
