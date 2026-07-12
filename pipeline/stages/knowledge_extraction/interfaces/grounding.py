"""GroundingChecker — protocol for NLI-based claim verification."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pipeline.stages.knowledge_extraction.models import AuditableSummary


@runtime_checkable
class GroundingChecker(Protocol):
    """
    Verifies that claims are entailed by their cited verbatim source text.

    Applied after MAP — filter findings whose verbatim_support does not entail
    the claim before they enter the post-MAP stages.
    """

    def filter_findings(self, summary: AuditableSummary) -> AuditableSummary:
        """Return a copy of summary with only grounded findings."""
        ...
