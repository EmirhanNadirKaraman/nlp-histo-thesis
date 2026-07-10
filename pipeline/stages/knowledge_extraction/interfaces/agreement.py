"""MapOutputScorer — protocol for MAP-stage voter-agreement scoring."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pipeline.stages.knowledge_extraction.models import AuditableSummary
from pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext, ScoreBundle


@runtime_checkable
class MapOutputScorer(Protocol):
    """
    Scores agreement between N voter outputs for a single MAP chunk.

    Implementations receive the voter AuditableSummary objects and optionally
    the original formatted source text and grounding quality context, and
    return a populated ScoreBundle.

    Contract
    --------
    - Must populate at least one score field (embedding_agreement,
      judge_agreement, etc.).
    - May optionally set ScoreBundle.decision; if not set, AgreementChecker
      falls back to comparing the primary score against theta.
    - Must be stateless across calls (safe for concurrent use).
    - ``context`` carries per-voter grounding quality signals (see
      ``AgreementContext``).  Scorers that do not use it may ignore it.
    """

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
        context: AgreementContext | None = None,
    ) -> ScoreBundle: ...
