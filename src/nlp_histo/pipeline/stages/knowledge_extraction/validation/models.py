"""Shared validation vocabulary for the knowledge-extraction gates.

Neutral, dependency-free DTOs used by BOTH the routing layer and the provenance
validators. Extracted here from ``routing/models.py`` so ``provenance/`` no
longer has to import from ``routing/`` — which breaks the routing <-> provenance
package dependency cycle.

Keep this module a **leaf**: it must not import from ``routing/`` or
``provenance/``. Routing-outcome types (``GateOrigin``, ``RoutingDecision``)
stay in ``routing/models.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
