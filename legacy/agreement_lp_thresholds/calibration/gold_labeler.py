"""GoldLabeler — offline gold-label generation using a strong LLM judge."""
from __future__ import annotations

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision
from pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert medical informatics reviewer.  Multiple AI analysts have
independently extracted clinical findings from the same text chunk of a
histopathology paper.

Your task: decide the correct action for this chunk based on the quality and
agreement of the extracted findings.

Choose exactly one of:
- KEEP: The voters substantially agree on the key findings and the best output
  is trustworthy enough to use directly.  Minor wording differences are fine.
- ESCALATE: The voters meaningfully disagree, or the findings are incomplete or
  uncertain, and a more capable model should review this chunk.
- REJECT: All voters produced empty, hallucinated, clearly off-topic, or
  entirely useless outputs.  No trustworthy finding can be derived.

Evaluation criteria:
- Agreement on categories (morphology, IHC, molecular_genetics, staging,
  treatment, prognosis), clinical direction (positive vs. negative), and
  evidence weight all matter.
- Exact wording differences alone do NOT justify ESCALATE.
- REJECT only when every voter output is worthless for this chunk.\
"""

_USER = """\
Source text:
{source_text}

Voter analyses ({n_voters} voters):
{voter_summaries}

Return your decision, a one-sentence rationale, and your confidence.\
"""


def _format_voters(outputs: list[AuditableSummary]) -> str:
    lines = []
    for i, o in enumerate(outputs, 1):
        findings = [
            {"category": f.category, "claim": f.claim, "confidence": f.confidence}
            for f in o.findings
        ]
        lines.append(f"Voter {i}: {json.dumps(findings, ensure_ascii=False)}")
    return "\n".join(lines)


class GoldLabel(BaseModel):
    """Structured output from the gold label judge."""

    decision: ChunkDecision = Field(description="KEEP, ESCALATE, or REJECT")
    rationale: str = Field(description="One-sentence explanation of the decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Judge confidence in [0, 1]")


class GoldLabeler:
    """
    Generates offline gold labels for agreement dataset construction.

    Uses a strong LLM (recommended: claude-opus-4-6) to produce three-class
    KEEP / ESCALATE / REJECT decisions for (voter_outputs, source_text) pairs.
    Intended for OFFLINE use only — not called in the real-time pipeline.

    Parameters
    ----------
    llm:
        LangChain chat model.  Should be a strong, instruction-following model.
        Recommended: init with langchain_anthropic.ChatAnthropic(model="claude-opus-4-6").
    """

    def __init__(self, llm: object) -> None:
        prompt = ChatPromptTemplate([("system", _SYSTEM), ("user", _USER)])
        self._chain = prompt | llm.with_structured_output(  # type: ignore[union-attr]
            GoldLabel, strict=True
        ).with_retry(stop_after_attempt=3)

    def label(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
    ) -> GoldLabel:
        """
        Generate a gold label for a set of voter outputs.

        Parameters
        ----------
        outputs:
            Voter AuditableSummary objects for one chunk.
        source_text:
            Formatted source text chunk (optional but strongly recommended
            for accurate labeling).
        """
        result: GoldLabel = self._chain.invoke({
            "source_text": source_text or "(not provided)",
            "n_voters": len(outputs),
            "voter_summaries": _format_voters(outputs),
        })
        logger.debug(
            "GoldLabeler: %s (conf=%.2f) — %s",
            result.decision.value,
            result.confidence,
            result.rationale,
        )
        return result
