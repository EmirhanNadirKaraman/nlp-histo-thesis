"""
Routing decision models for the grounding-first MAP-stage router.

These are internal audit/decision objects — not persisted to disk directly,
but designed to be easily serialisable for logging and debugging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision, ScoreBundle


class ReasonCode(str, Enum):
    """Reason codes produced by each gate in the routing pipeline."""

    # ── Schema gate ────────────────────────────────────────────────────────────
    INVALID_SCHEMA = "invalid_schema"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_CATEGORY = "invalid_category"
    EMPTY_CLAIM = "empty_claim"
    EMPTY_EVIDENCE_CHAIN = "empty_evidence_chain"
    INVALID_SENTENCE_ID = "invalid_sentence_id"
    INVALID_TEXT_ELEMENT_ID = "invalid_text_element_id"
    NO_FINDINGS_EXTRACTED = "no_findings_extracted"

    # ── Grounding gate — hard failures (→ REJECT when all voters fail) ─────────
    NONEXISTENT_SOURCE = "nonexistent_source"
    CROSS_DOCUMENT_SOURCE_ERROR = "cross_document_source_error"
    FABRICATED_VERBATIM_SUPPORT = "fabricated_verbatim_support"

    # ── Grounding gate — soft failures (→ ESCALATE) ────────────────────────────
    WEAK_GROUNDING = "weak_grounding"
    PARTIAL_SUPPORT = "partial_support"
    AMBIGUOUS_SUPPORT = "ambiguous_support"
    UNSUPPORTED_CLAIM = "unsupported_claim"

    # ── Agreement gate ─────────────────────────────────────────────────────────
    HIGH_AGREEMENT = "high_agreement"
    INSUFFICIENT_AGREEMENT = "insufficient_agreement"
    ESCALATED_DUE_TO_LOW_AGREEMENT = "escalated_due_to_low_agreement"
    # B-051: structural hard-fail — comparable findings with opposite polarities.
    # Emitted INSTEAD of the low-agreement codes when the embedding score was
    # high but a polarity contradiction was detected; never emitted alongside
    # them, so the audit trail unambiguously distinguishes "the voters agreed
    # numerically but contradicted structurally" from "the voters disagreed".
    POLARITY_CONFLICT = "polarity_conflict"


class GateOrigin(str, Enum):
    """Which gate produced the final routing decision."""

    SCHEMA_GATE = "schema_gate"
    GROUNDING_GATE = "grounding_gate"
    AGREEMENT_GATE = "agreement_gate"


@dataclass
class FindingValidation:
    """
    Per-finding validation result from either the schema or provenance gate.

    Attributes
    ----------
    finding_index:
        Zero-based position of the finding within its AuditableSummary.
    claim_preview:
        First 80 characters of the claim for log readability.
    passed:
        True if no validation codes were raised.
    reason_codes:
        All reason codes that apply to this finding.
    explanation:
        Human-readable summary of what failed (or "ok").
    """

    finding_index: int
    claim_preview: str
    passed: bool
    reason_codes: list[ReasonCode] = field(default_factory=list)
    explanation: str = "ok"


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
