"""
Q3 — Recall gap check.

Opus reviews paragraphs and identifies generalizable medical findings
that the pipeline missed.

Samples both:
  • paragraphs where MAP extracted findings (with_extractions)
  • paragraphs with zero extractions (zero_extractions)

Zero-extraction paragraphs are section-filtered to exclude boilerplate
(acknowledgements, funding, etc.) so we don't waste judge calls.
"""
from __future__ import annotations

import logging
from typing import Any

from ..client import JudgeRequest
from ..cache import JudgeCache
from ..sampling import stable_seed, sample_paragraphs
from ..prompts import Q3_SYSTEM, Q3_USER, Q3_TOOL_NAME, Q3_SCHEMA
from .. import LABEL_SOURCE, MODEL, PROMPT_VERSION, SCHEMA_VERSION

logger = logging.getLogger(__name__)

TASK = "q3_recall"


def build_q3_requests(
    session: Any,
    pmcid: str,
    run_id: int,
    n_with_extraction: int,
    n_zero_extraction: int,
    global_seed: int,
) -> tuple[list[JudgeRequest], list[dict]]:
    """Build judge requests for Q3. Returns (requests, skipped)."""
    seed = stable_seed(global_seed, pmcid, str(run_id), TASK)

    paragraphs = sample_paragraphs(
        session, pmcid, run_id,
        n_with_extraction=n_with_extraction,
        n_zero_extraction=n_zero_extraction,
        seed=seed,
    )
    if not paragraphs:
        logger.warning("Q3 [%s] no paragraphs sampled", pmcid)
        return [], [{"pmcid": pmcid, "task": TASK, "reason": "no_paragraphs"}]

    requests: list[JudgeRequest] = []
    skipped: list[dict] = []

    for para in paragraphs:
        ck = JudgeCache.make_key(
            TASK,
            paragraph=para.text_content,
            claims=sorted(para.extracted_claims),
            stratum=para.stratum,
        )

        if para.extracted_claims:
            claims_text = "\n".join(f"- {c}" for c in para.extracted_claims)
        else:
            claims_text = "(none — no findings were extracted from this paragraph)"

        user = Q3_USER.format(
            paragraph=para.text_content,
            claims=claims_text,
        )

        requests.append(JudgeRequest(
            custom_id=f"{TASK}_{ck}",
            cache_key=ck,
            task=TASK,
            system=Q3_SYSTEM,
            user=user,
            tool_name=Q3_TOOL_NAME,
            schema=Q3_SCHEMA,
            metadata={
                "pmcid": pmcid,
                "pipeline_run_id": run_id,
                "text_element_id": para.text_element_id,
                "stratum": para.stratum,
                "paragraph": para.text_content,
                "extracted_claims": para.extracted_claims,
                "section_path": para.section_path,
            },
        ))

    return requests, skipped


def parse_q3_result(request: JudgeRequest, judgment: dict) -> dict:
    """Merge metadata + Opus judgment."""
    row = {
        **request.metadata,
        **judgment,
        "label_source": LABEL_SOURCE,
        "judge_model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
    }
    return row
