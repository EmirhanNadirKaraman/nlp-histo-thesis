"""
MapOutputRouter — grounding-first routing layer for MAP-stage voter outputs.

Decision flow
-------------
1. Classify each voter as ELIGIBLE / WEAKLY_GROUNDED / UNUSABLE.
2. Derive a chunk-level decision from N_eligible count.
3. If N_eligible >= 2: pass ELIGIBLE voters to the agreement gate.

Voter tiers
-----------
ELIGIBLE       — passes all schema + provenance checks (hard and soft).
                 Enters the similarity matrix for consensus scoring.
WEAKLY_GROUNDED — no hard failures, but soft provenance failures present.
                 Excluded from matrix; triggers ESCALATE if N_eligible < 2.
UNUSABLE       — any hard schema or provenance failure.
                 Excluded from matrix; triggers REJECT if all voters are unusable.

Key design rules (v2)
---------------------
- REJECT comes from hard validator failures, not from score thresholds.
- Score-based REJECT from the agreement gate is opt-in (default: ESCALATE).
- Both REJECT and ESCALATE route to the escalation model in _cascade().
  REJECT is an audit label for "voter outputs are structurally unusable",
  not a terminal chunk drop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pipeline.stages.summarization.agreement.checker import AgreementChecker
from pipeline.stages.summarization.interfaces.scoring import (
    AgreementContext,
    ChunkDecision,
    VoterContext,
)
from pipeline.stages.summarization.models import AuditableSummary

from .models import (
    FindingValidation,
    GateOrigin,
    ReasonCode,
    RoutingDecision,
)
from .provenance_validator import ProvenanceValidator, SourceIndex
from .schema_validator import SchemaValidator

logger = logging.getLogger(__name__)

# Hard codes — any of these on a voter → UNUSABLE tier.
_HARD_CODES = frozenset([
    ReasonCode.INVALID_SCHEMA,
    ReasonCode.MISSING_REQUIRED_FIELD,
    ReasonCode.INVALID_CATEGORY,
    ReasonCode.EMPTY_CLAIM,
    ReasonCode.EMPTY_EVIDENCE_CHAIN,
    ReasonCode.INVALID_SENTENCE_ID,
    ReasonCode.INVALID_TEXT_ELEMENT_ID,
    ReasonCode.NO_FINDINGS_EXTRACTED,
    ReasonCode.NONEXISTENT_SOURCE,
    ReasonCode.CROSS_DOCUMENT_SOURCE_ERROR,
    ReasonCode.FABRICATED_VERBATIM_SUPPORT,
])

# Schema-specific hard codes — used to determine gate_origin on REJECT.
_SCHEMA_HARD_CODES = frozenset([
    ReasonCode.INVALID_SCHEMA,
    ReasonCode.MISSING_REQUIRED_FIELD,
    ReasonCode.INVALID_CATEGORY,
    ReasonCode.EMPTY_CLAIM,
    ReasonCode.EMPTY_EVIDENCE_CHAIN,
    ReasonCode.INVALID_SENTENCE_ID,
    ReasonCode.NO_FINDINGS_EXTRACTED,
])

# Soft codes — real-but-uncertain grounding → WEAKLY_GROUNDED tier.
_SOFT_CODES = frozenset([
    ReasonCode.WEAK_GROUNDING,
    ReasonCode.PARTIAL_SUPPORT,
    ReasonCode.AMBIGUOUS_SUPPORT,
    ReasonCode.UNSUPPORTED_CLAIM,
])


@dataclass
class VoterClassification:
    """Per-voter classification result from _classify_voters()."""

    voter_index: int
    status: Literal["eligible", "weakly_grounded", "unusable"]
    schema_validations: list[FindingValidation]
    grounding_validations: list[FindingValidation]
    grounding_pass_fraction: float    # fraction of findings with no failures
    mean_evidence_chain_length: float  # mean len(Finding.evidence) per finding


# ── Pure helpers ────────────────────────────────────────────────────────────────

def _has_hard(validations: list[FindingValidation]) -> bool:
    return any(c in _HARD_CODES for v in validations for c in v.reason_codes)


def _has_soft(validations: list[FindingValidation]) -> bool:
    """True if any finding has a soft code and no hard code on the same finding."""
    return any(
        c in _SOFT_CODES for v in validations for c in v.reason_codes
        if not any(hc in _HARD_CODES for hc in v.reason_codes)
    )


def _collect_codes(
    validations_list: list[list[FindingValidation]],
    code_set: frozenset[ReasonCode],
) -> list[ReasonCode]:
    seen: set[ReasonCode] = set()
    for vvs in validations_list:
        for v in vvs:
            for c in v.reason_codes:
                if c in code_set:
                    seen.add(c)
    return list(seen)


def _flat(voter_validations: list[list[FindingValidation]]) -> list[FindingValidation]:
    return [v for vvs in voter_validations for v in vvs]


def _grounding_pass_fraction(validations: list[FindingValidation]) -> float:
    if not validations:
        return 1.0
    return sum(1 for v in validations if v.passed) / len(validations)


def _mean_evidence_length(summary: AuditableSummary) -> float:
    if not summary.findings:
        return 0.0
    return sum(len(f.evidence) for f in summary.findings) / len(summary.findings)


# ── Router ──────────────────────────────────────────────────────────────────────

class MapOutputRouter:
    """
    Grounding-first routing layer for MAP-stage voter outputs.

    Parameters
    ----------
    agreement_checker:
        The AgreementChecker instance wired into MapStage.
    schema_validator:
        SchemaValidator instance.  Defaults to SchemaValidator().
    fabricated_threshold:
        Verbatim SequenceMatcher ratio below which verbatim is fabricated.
        Passed to ProvenanceValidator.
    weak_threshold:
        Verbatim SequenceMatcher ratio below which verbatim is weak grounding.
        Passed to ProvenanceValidator.
    reject_on_low_agreement:
        When False (default), low agreement always ESCALATEs.
        Set True to allow score-based REJECT from the agreement gate.
    single_voter_policy:
        What to do when only one ELIGIBLE voter passes validation.
        "escalate" (default): escalate to the strong model.
        "keep": accept the single voter without peer comparison.
    """

    def __init__(
        self,
        agreement_checker: AgreementChecker,
        schema_validator: SchemaValidator | None = None,
        fabricated_threshold: float = 0.25,
        weak_threshold: float = 0.60,
        reject_on_low_agreement: bool = False,
        single_voter_policy: Literal["escalate", "keep"] = "escalate",
    ) -> None:
        self._agreement = agreement_checker
        self._schema = schema_validator or SchemaValidator()
        self._fabricated_threshold = fabricated_threshold
        self._weak_threshold = weak_threshold
        self._reject_on_low_agreement = reject_on_low_agreement
        self._single_voter_policy = single_voter_policy

    def route(
        self,
        outputs: list[AuditableSummary],
        chunk: list[dict],
        pmcid: str,
        source_text: str,
    ) -> RoutingDecision:
        """
        Classify voters, derive chunk-level decision, and optionally score agreement.

        Parameters
        ----------
        outputs:
            Voter AuditableSummary list (one per voter LLM).
        chunk:
            Raw sentence dicts for this chunk, used to build the SourceIndex.
        pmcid:
            Document PMCID for cross-document provenance check.
        source_text:
            Formatted source text string forwarded to the agreement scorer.
        """
        provenance_validator = ProvenanceValidator(
            source_index=SourceIndex(chunk),
            document_pmcid=pmcid,
            fabricated_threshold=self._fabricated_threshold,
            weak_threshold=self._weak_threshold,
        )

        classifications = self._classify_voters(outputs, provenance_validator)

        # Build per-voter grounding contexts (all voters, in original index order)
        # for trace enrichment in _cascade().  Indexed by voter_index.
        all_voter_contexts = [
            VoterContext(
                grounding_pass_fraction=c.grounding_pass_fraction,
                mean_evidence_length=c.mean_evidence_chain_length,
            )
            for c in sorted(classifications, key=lambda c: c.voter_index)
        ]

        early = self._chunk_decision_from_classifications(classifications)
        if early is not None:
            logger.debug(
                "Router pre-agreement: %s (gate=%s reasons=%s)",
                early.decision.value,
                early.gate_origin.value,
                [r.value for r in early.reason_codes],
            )
            early.voter_grounding_contexts = all_voter_contexts
            return early

        # N_eligible >= 2 — pass ELIGIBLE voters to the agreement gate.
        eligible = [c for c in classifications if c.status == "eligible"]
        valid_voter_indices = [c.voter_index for c in eligible]
        valid_outputs = [outputs[i] for i in valid_voter_indices]
        context = AgreementContext(
            voter_contexts=[
                VoterContext(
                    grounding_pass_fraction=c.grounding_pass_fraction,
                    mean_evidence_length=c.mean_evidence_chain_length,
                )
                for c in eligible
            ]
        )

        decision = self._agreement_gate(
            valid_outputs, source_text, valid_voter_indices, context
        )
        decision.voter_grounding_contexts = all_voter_contexts
        logger.debug(
            "Router agreement gate: %s (gate=%s reasons=%s)",
            decision.decision.value,
            decision.gate_origin.value,
            [r.value for r in decision.reason_codes],
        )
        return decision

    # ── Voter classification ────────────────────────────────────────────────────

    def _classify_voters(
        self,
        outputs: list[AuditableSummary],
        provenance_validator: ProvenanceValidator,
    ) -> list[VoterClassification]:
        """Classify each voter as ELIGIBLE / WEAKLY_GROUNDED / UNUSABLE."""
        classifications: list[VoterClassification] = []
        for i, output in enumerate(outputs):
            if not output.findings:
                classifications.append(VoterClassification(
                    voter_index=i,
                    status="unusable",
                    schema_validations=[FindingValidation(
                        finding_index=0,
                        claim_preview="(no findings)",
                        passed=False,
                        reason_codes=[ReasonCode.NO_FINDINGS_EXTRACTED],
                        explanation="Voter produced no findings.",
                    )],
                    grounding_validations=[],
                    grounding_pass_fraction=0.0,
                    mean_evidence_chain_length=0.0,
                ))
                continue

            schema_vals = self._schema.validate(output)
            grounding_vals = provenance_validator.validate(output)

            if _has_hard(schema_vals) or _has_hard(grounding_vals):
                status: Literal["eligible", "weakly_grounded", "unusable"] = "unusable"
            elif _has_soft(grounding_vals):
                status = "weakly_grounded"
            else:
                status = "eligible"

            classifications.append(VoterClassification(
                voter_index=i,
                status=status,
                schema_validations=schema_vals,
                grounding_validations=grounding_vals,
                grounding_pass_fraction=_grounding_pass_fraction(grounding_vals),
                mean_evidence_chain_length=_mean_evidence_length(output),
            ))

        logger.debug(
            "Voter classifications: %s",
            {c.voter_index: c.status for c in classifications},
        )
        return classifications

    # ── Chunk-level decision from N_eligible count ──────────────────────────────

    def _chunk_decision_from_classifications(
        self,
        classifications: list[VoterClassification],
    ) -> RoutingDecision | None:
        """
        Return an early RoutingDecision when N_eligible < 2, else None.

        Callers must proceed to the agreement gate when None is returned.
        """
        eligible = [c for c in classifications if c.status == "eligible"]
        unusable = [c for c in classifications if c.status == "unusable"]
        weak = [c for c in classifications if c.status == "weakly_grounded"]

        n_eligible = len(eligible)

        if n_eligible >= 2:
            return None

        # ── N_eligible == 0 ─────────────────────────────────────────────────────
        if n_eligible == 0:
            if unusable:
                # At least one hard failure — REJECT.
                # Gate origin: GROUNDING_GATE if any grounding-specific hard code;
                # otherwise SCHEMA_GATE (only schema hard codes present).
                # Schema failures take priority: if every unusable voter has at
                # least one schema hard code, the root cause is schema validation.
                # Only fall through to GROUNDING_GATE when grounding failures
                # exist without corresponding schema failures.
                all_have_schema_failure = all(
                    any(
                        c in _SCHEMA_HARD_CODES
                        for v in cls.schema_validations
                        for c in v.reason_codes
                    )
                    for cls in unusable
                )
                gate_origin = (
                    GateOrigin.SCHEMA_GATE
                    if all_have_schema_failure
                    else GateOrigin.GROUNDING_GATE
                )
                codes = _collect_codes(
                    [c.schema_validations + c.grounding_validations for c in unusable],
                    _HARD_CODES,
                )
                return RoutingDecision(
                    decision=ChunkDecision.REJECT,
                    gate_origin=gate_origin,
                    reason_codes=codes,
                    explanation=(
                        f"All {len(unusable)} voter(s) unusable (hard validation failures)."
                    ),
                    schema_details=_flat([c.schema_validations for c in classifications]),
                    grounding_details=_flat(
                        [c.grounding_validations for c in classifications]
                    ),
                )

            # All weak (N_unusable == 0) — ESCALATE.
            codes = _collect_codes([c.grounding_validations for c in weak], _SOFT_CODES)
            return RoutingDecision(
                decision=ChunkDecision.ESCALATE,
                gate_origin=GateOrigin.GROUNDING_GATE,
                reason_codes=codes,
                explanation=(
                    f"All {len(weak)} voter(s) weakly grounded; no eligible voters."
                ),
                grounding_details=_flat(
                    [c.grounding_validations for c in classifications]
                ),
            )

        # ── N_eligible == 1 ─────────────────────────────────────────────────────
        only = eligible[0]
        if self._single_voter_policy == "keep":
            return RoutingDecision(
                decision=ChunkDecision.KEEP,
                gate_origin=GateOrigin.AGREEMENT_GATE,
                reason_codes=[ReasonCode.HIGH_AGREEMENT],
                explanation="Single eligible voter accepted (single_voter_policy='keep').",
                valid_voter_indices=[only.voter_index],
            )

        return RoutingDecision(
            decision=ChunkDecision.ESCALATE,
            gate_origin=GateOrigin.GROUNDING_GATE,
            reason_codes=[ReasonCode.ESCALATED_DUE_TO_LOW_AGREEMENT],
            explanation=(
                f"Only 1 eligible voter (policy='escalate'); "
                f"{len(unusable)} unusable, {len(weak)} weakly grounded."
            ),
            valid_voter_indices=[only.voter_index],
        )

    # ── Agreement gate ──────────────────────────────────────────────────────────

    def _agreement_gate(
        self,
        valid_outputs: list[AuditableSummary],
        source_text: str,
        valid_voter_indices: list[int],
        context: AgreementContext,
    ) -> RoutingDecision:
        """
        Score agreement over ELIGIBLE voters only.

        Passes grounding quality metadata via AgreementContext so scorers
        (e.g. SemanticAgreementScorer) can use it for tie-breaking without
        requiring a scorer-specific method.
        """
        bundle = self._agreement.compute(
            valid_outputs, source_text=source_text, context=context
        )

        conf_str = f"{bundle.confidence:.2f}" if bundle.confidence is not None else "n/a"

        # B-051: polarity hard-fail overrides any theta-based decision. Emit
        # ONLY POLARITY_CONFLICT — never co-emit a low-agreement reason
        # because the embedding score was not actually low.
        if (
            bundle.score_details is not None
            and bundle.score_details.get("hard_fail_reason") == "polarity_conflict"
        ):
            conflict = bundle.score_details.get("polarity_conflict_details") or {}
            n_pairs = conflict.get("count", 0)
            return RoutingDecision(
                decision=ChunkDecision.ESCALATE,
                gate_origin=GateOrigin.AGREEMENT_GATE,
                reason_codes=[ReasonCode.POLARITY_CONFLICT],
                explanation=(
                    f"Polarity conflict on {n_pairs} comparable finding pair(s); "
                    f"embedding agreement={conf_str} overridden."
                ),
                agreement_details=bundle,
                valid_voter_indices=valid_voter_indices,
            )

        if bundle.decision == ChunkDecision.KEEP:
            return RoutingDecision(
                decision=ChunkDecision.KEEP,
                gate_origin=GateOrigin.AGREEMENT_GATE,
                reason_codes=[ReasonCode.HIGH_AGREEMENT],
                explanation=f"Eligible voters agree (confidence={conf_str}).",
                agreement_details=bundle,
                valid_voter_indices=valid_voter_indices,
            )

        if bundle.decision == ChunkDecision.REJECT:
            if self._reject_on_low_agreement:
                return RoutingDecision(
                    decision=ChunkDecision.REJECT,
                    gate_origin=GateOrigin.AGREEMENT_GATE,
                    reason_codes=[ReasonCode.INSUFFICIENT_AGREEMENT],
                    explanation=(
                        f"Eligible voter agreement too low (confidence={conf_str}) "
                        "and score-based rejection is enabled."
                    ),
                    agreement_details=bundle,
                    valid_voter_indices=valid_voter_indices,
                )
            # Default: downgrade REJECT → ESCALATE.
            return RoutingDecision(
                decision=ChunkDecision.ESCALATE,
                gate_origin=GateOrigin.AGREEMENT_GATE,
                reason_codes=[ReasonCode.ESCALATED_DUE_TO_LOW_AGREEMENT],
                explanation=(
                    f"Eligible voter agreement very low (confidence={conf_str}); "
                    "score-based rejection is disabled — escalating instead."
                ),
                agreement_details=bundle,
                valid_voter_indices=valid_voter_indices,
            )

        # ESCALATE from scorer.
        return RoutingDecision(
            decision=ChunkDecision.ESCALATE,
            gate_origin=GateOrigin.AGREEMENT_GATE,
            reason_codes=[ReasonCode.ESCALATED_DUE_TO_LOW_AGREEMENT],
            explanation=f"Insufficient eligible voter agreement (confidence={conf_str}).",
            agreement_details=bundle,
            valid_voter_indices=valid_voter_indices,
        )
