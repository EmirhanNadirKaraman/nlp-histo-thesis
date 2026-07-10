"""GroundingChecker — protocol for NLI-based claim verification."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pipeline.stages.knowledge_extraction.models import AuditableSummary, ExtractedRules


@runtime_checkable
class GroundingChecker(Protocol):
    """
    Verifies that claims are entailed by their cited verbatim source text.

    Applied at two points in the pipeline:
      1. After MAP   — filter findings whose verbatim_support does not entail
                       the claim before they enter REDUCE.
      2. After RULES — filter/trim rules whose evidence_chain items do not
                       entail the condition+action.
    """

    def filter_findings(self, summary: AuditableSummary) -> AuditableSummary:
        """Return a copy of summary with only grounded findings."""
        ...

    def filter_rules(self, rules: ExtractedRules) -> ExtractedRules:
        """Return a copy of rules with ungrounded evidence trimmed/dropped."""
        ...
