"""LLMJudgeScorer — single LLM call to rate overall voter agreement."""
from __future__ import annotations

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext, ScoreBundle
from pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are evaluating whether multiple medical analysts agree on the key findings
extracted from the same text chunk.

Rate the overall agreement between the provided analyses on a scale of 0.0 to 1.0:
- 1.0: All analysts extracted the same key findings.
- 0.5: Analysts partially agree (some shared findings, some unique or conflicting).
- 0.0: Analysts completely disagree on what findings are present.

Consider: which categories of evidence are present, the direction of clinical
claims (positive/negative), and the overall weight of evidence — not exact wording.\
"""

_USER = """\
Source text:
{source_text}

Voter analyses:
{voter_summaries}

Return a single agreement score.\
"""


class _JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Agreement score in [0, 1]")
    rationale: str = Field(description="One-sentence explanation of the score")


def _format_voters(outputs: list[AuditableSummary]) -> str:
    lines = [
        f"Voter {i}: {json.dumps([f.claim for f in o.findings])}"
        for i, o in enumerate(outputs, 1)
    ]
    return "\n".join(lines)


class LLMJudgeScorer:
    """
    Rates overall voter agreement with a single LLM call per chunk.

    Unlike pairwise comparison (O(n²) calls), this presents all voter outputs
    together and asks the LLM for one holistic score — efficient for 2–4 voters.
    Score is stored in ScoreBundle.judge_agreement.

    Parameters
    ----------
    llm:
        LangChain chat model.  A cheap model (e.g. gpt-4o-mini) is sufficient.
    """

    def __init__(self, llm: object) -> None:
        prompt = ChatPromptTemplate([("system", _SYSTEM), ("user", _USER)])
        self._chain = prompt | llm.with_structured_output(  # type: ignore[union-attr]
            _JudgeScore, strict=True
        ).with_retry(stop_after_attempt=3)

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
        context: AgreementContext | None = None,  # noqa: ARG002
    ) -> ScoreBundle:
        result: _JudgeScore = self._chain.invoke({
            "source_text": source_text or "(not provided)",
            "voter_summaries": _format_voters(outputs),
        })
        logger.debug("LLM judge: %.2f — %s", result.score, result.rationale)
        return ScoreBundle(judge_agreement=result.score)
