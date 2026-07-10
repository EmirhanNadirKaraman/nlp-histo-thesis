"""ContradictionChecker — protocol for rule contradiction detection."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pipeline.stages.knowledge_extraction.models import ContradictionReport, ExtractedRules


@runtime_checkable
class ContradictionChecker(Protocol):
    """
    Groups extracted rules into contradicting pairs and consistent rules.

    Applied after grounding filtering so only well-evidenced rules are
    evaluated.
    """

    def detect(self, rules: ExtractedRules) -> ContradictionReport:
        """Return a ContradictionReport for the given rule set."""
        ...
