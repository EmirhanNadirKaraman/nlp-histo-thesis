"""
Q5 — Paragraph-level extraction F1.

Opus produces complete silver MAP findings for a paragraph, then
provides alignment decisions mapping silver findings ↔ pipeline findings.

TP / FP / FN / precision / recall / F1 are computed in Python from the
alignments returned by Opus.
"""
from __future__ import annotations

import logging
from typing import Any

from ..client import JudgeRequest
from ..cache import JudgeCache
from ..sampling import stable_seed, sample_paragraphs
from ..prompts import Q5_SYSTEM, Q5_USER, Q5_TOOL_NAME, Q5_SCHEMA
from .. import LABEL_SOURCE, MODEL, PROMPT_VERSION, SCHEMA_VERSION

logger = logging.getLogger(__name__)

TASK = "q5_f1"


def build_q5_requests(
    session: Any,
    pmcid: str,
    run_id: int,
    n_paragraphs: int,
    global_seed: int,
) -> tuple[list[JudgeRequest], list[dict]]:
    """Build judge requests for Q5. Returns (requests, skipped)."""
    seed = stable_seed(global_seed, pmcid, str(run_id), TASK)

    # Sample paragraphs — mix of with and zero extraction
    n_with = max(1, n_paragraphs * 2 // 3)
    n_zero = n_paragraphs - n_with
    paragraphs = sample_paragraphs(
        session, pmcid, run_id,
        n_with_extraction=n_with,
        n_zero_extraction=n_zero,
        seed=seed,
    )
    if not paragraphs:
        logger.warning("Q5 [%s] no paragraphs sampled", pmcid)
        return [], [{"pmcid": pmcid, "task": TASK, "reason": "no_paragraphs"}]

    requests: list[JudgeRequest] = []
    skipped: list[dict] = []

    for para in paragraphs:
        # Format pipeline findings for the prompt
        if para.extracted_claims:
            findings_text = "\n".join(
                f"[{i}] {c}" for i, c in enumerate(para.extracted_claims)
            )
        else:
            findings_text = "(none)"

        ck = JudgeCache.make_key(
            TASK,
            paragraph=para.text_content,
            pipeline_findings=sorted(para.extracted_claims),
        )

        user = Q5_USER.format(
            paragraph=para.text_content,
            pipeline_findings=findings_text,
        )

        requests.append(JudgeRequest(
            custom_id=f"{TASK}_{ck}",
            cache_key=ck,
            task=TASK,
            system=Q5_SYSTEM,
            user=user,
            tool_name=Q5_TOOL_NAME,
            schema=Q5_SCHEMA,
            metadata={
                "pmcid": pmcid,
                "pipeline_run_id": run_id,
                "text_element_id": para.text_element_id,
                "stratum": para.stratum,
                "paragraph": para.text_content,
                "pipeline_claims": para.extracted_claims,
                "n_pipeline": len(para.extracted_claims),
            },
        ))

    return requests, skipped


def _compute_f1_from_alignments(
    judgment: dict,
    n_pipeline: int,
) -> dict:
    """Compute TP/FP/FN/precision/recall/F1 from Opus alignment decisions.

    Opus returns:
      silver_findings: list of extracted findings
      alignments: list of {silver_index, pipeline_index, match_type}

    A silver finding with match_type="match" → TP
    A silver finding with match_type="partial" → counted separately
    A silver finding with match_type="unmatched" (pipeline_index=null) → FN
    A pipeline finding with no silver finding aligned to it → FP
    """
    alignments = judgment.get("alignments", [])
    silver_findings = judgment.get("silver_findings", [])
    n_silver = len(silver_findings)

    tp = sum(1 for a in alignments if a.get("match_type") == "match")
    partial = sum(1 for a in alignments if a.get("match_type") == "partial")
    fn = sum(1 for a in alignments if a.get("match_type") == "unmatched")

    # Pipeline findings not matched by any silver finding → FP
    matched_pipeline_indices = {
        a["pipeline_index"]
        for a in alignments
        if a.get("pipeline_index") is not None and a.get("match_type") in ("match", "partial")
    }
    fp = max(0, n_pipeline - len(matched_pipeline_indices))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_silver": n_silver,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "partial_match_count": partial,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def parse_q5_result(request: JudgeRequest, judgment: dict) -> dict:
    """Merge metadata + Opus judgment + computed F1 metrics."""
    f1_stats = _compute_f1_from_alignments(judgment, request.metadata["n_pipeline"])
    row = {
        **request.metadata,
        **judgment,
        **f1_stats,
        "label_source": LABEL_SOURCE,
        "judge_model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
    }
    return row
