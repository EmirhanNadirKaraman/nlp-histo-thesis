"""
Routing decision models for the grounding-first MAP-stage router.

These are internal audit/decision objects — not persisted to disk directly,
but designed to be easily serialisable for logging and debugging.

``ReasonCode`` and ``FindingValidation`` are shared with the provenance
validators and now live in the neutral leaf module
``knowledge_extraction/validation/models.py`` (so provenance need not import
from routing — breaking the routing<->provenance cycle). They are re-exported
here for backward compatibility with the call sites that import them from
``routing.models``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision, ScoreBundle
from pipeline.stages.knowledge_extraction.validation.models import (  # noqa: F401  (compat re-export)
    FindingValidation,
    ReasonCode,
)


class GateOrigin(str, Enum):
    """Which gate produced the final routing decision."""

    SCHEMA_GATE = "schema_gate"
    GROUNDING_GATE = "grounding_gate"
    AGREEMENT_GATE = "agreement_gate"


@dataclass
class RoutingDecision:
    """
    Final routing decision for one MAP chunk, with full audit trail.

    Attributes
    ----------
    decision:
        KEEP / REJECT / ESCALATE.
    gate_origin:
        Which gate produced this decision.
    reason_codes:
        All reason codes that contributed.
    explanation:
        Human-readable explanation of the decision.
    schema_details:
        Per-finding results from the schema gate (None if gate was not reached).
    grounding_details:
        Per-finding results from the grounding gate (None if gate was not reached).
    agreement_details:
        ScoreBundle from the agreement gate (None if gate was not reached).
    """

    decision: ChunkDecision
    gate_origin: GateOrigin
    reason_codes: list[ReasonCode]
    explanation: str
    schema_details: list[FindingValidation] | None = None
    grounding_details: list[FindingValidation] | None = None
    agreement_details: ScoreBundle | None = None
    # Indices into the original voter list that were classified ELIGIBLE.
    # Populated on KEEP decisions so _cascade() knows which voters to pass to best().
    valid_voter_indices: list[int] | None = None
    # Per-voter grounding contexts from ProvenanceValidator — one entry per voter,
    # indexed by original voter position.  Populated by the router for all voters
    # (eligible, weakly-grounded, and unusable) so traces can show real grounding
    # quality instead of the structural fallback proxy.
    voter_grounding_contexts: list | None = None  # list[VoterContext] | None
