"""
SchemaValidator — structural integrity checks for AuditableSummary findings.

These checks catch hard failures that Pydantic does not enforce:
  - empty claim text
  - empty verbatim_support
  - empty evidence list
  - malformed citation ID format

Pydantic's Literal constraints (category, confidence) are enforced at parse
time by with_structured_output(), so they will rarely reach this validator in
production.  The category check is included for defence-in-depth.
"""
from __future__ import annotations

import re

from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary, Finding

from .models import FindingValidation, ReasonCode

# Citation IDs in Finding.evidence arrive without brackets: "S1|PMC123456|789".
# The PMC token tolerates word-chars + hyphens so document-id suffixes such as
# `PMC10100421_HIS-82-393` or `PMC7150310_main` (used as opaque doc keys in the
# DB) are accepted. Cross-document equality is enforced separately in
# provenance/validator.py against the expected pmcid.
_CITATION_RE = re.compile(r"^S\d+\|PMC[\w\-]+\|\d+$")

_VALID_CATEGORIES = frozenset([
    "morphology",
    "IHC",
    "molecular_genetics",
    "staging",
    "treatment",
    "prognosis",
    "demographic",
])


class SchemaValidator:
    """
    Validates the structural integrity of Finding objects inside an AuditableSummary.

    Returns one FindingValidation per finding.  Findings that pass all checks
    have passed=True and no reason_codes.

    This validator is stateless and requires no external resources.
    """

    def validate(self, summary: AuditableSummary) -> list[FindingValidation]:
        """Return one FindingValidation per finding in the summary."""
        return [
            self._validate_finding(idx, finding)
            for idx, finding in enumerate(summary.findings)
        ]

    def _validate_finding(self, idx: int, finding: Finding) -> FindingValidation:
        codes: list[ReasonCode] = []
        explanations: list[str] = []

        # Empty claim
        if not finding.claim or not finding.claim.strip():
            codes.append(ReasonCode.EMPTY_CLAIM)
            explanations.append("claim is empty or whitespace-only")

        # Invalid category (defence-in-depth; Pydantic usually catches this)
        if finding.category not in _VALID_CATEGORIES:
            codes.append(ReasonCode.INVALID_CATEGORY)
            explanations.append(f"unknown category: {finding.category!r}")

        # Empty evidence list
        if not finding.evidence:
            codes.append(ReasonCode.EMPTY_EVIDENCE_CHAIN)
            explanations.append("evidence list is empty")
        else:
            # Malformed citation IDs
            bad_cites = [c for c in finding.evidence if not _CITATION_RE.match(c)]
            if bad_cites:
                codes.append(ReasonCode.INVALID_SENTENCE_ID)
                examples = ", ".join(repr(c) for c in bad_cites[:3])
                explanations.append(f"malformed citation ID(s): {examples}")

        # Empty verbatim support
        if not finding.verbatim_support or not finding.verbatim_support.strip():
            codes.append(ReasonCode.MISSING_REQUIRED_FIELD)
            explanations.append("verbatim_support is empty or whitespace-only")

        passed = not codes
        return FindingValidation(
            finding_index=idx,
            claim_preview=finding.claim[:80] if finding.claim else "",
            passed=passed,
            reason_codes=codes,
            explanation="; ".join(explanations) if explanations else "ok",
        )
