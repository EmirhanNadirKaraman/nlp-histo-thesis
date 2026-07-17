"""
Q1 — MAP finding precision / field correctness.

Opus judges each MAP finding against its verbatim source text:
 • is the claim grounded?
 • are subject_entity, outcome_entity, relation_type, direction, category correct?

Opus returns corrected fields; fields_changed is computed in Python.
"""
from __future__ import annotations

import logging
import random
from typing import Any

from ..client import JudgeRequest
from ..cache import JudgeCache
from ..sampling import stable_seed
from ..prompts import Q1_SYSTEM, Q1_USER, Q1_TOOL_NAME, Q1_SCHEMA
from .. import LABEL_SOURCE, MODEL, PROMPT_VERSION, SCHEMA_VERSION

logger = logging.getLogger(__name__)

TASK = "q1_precision"


def _compute_fields_changed(pipeline: dict, judgment: dict) -> list[str]:
    """Compare pipeline values to Opus-corrected values, return changed field names."""
    changed: list[str] = []
    checks = [
        ("relation_type", "pipeline_relation_type", "correct_relation_type"),
        ("direction", "pipeline_direction", "correct_direction"),
        ("category", "pipeline_category", "correct_category"),
        ("subject_entity", "pipeline_subject_entity", "correct_subject_entity"),
        ("outcome_entity", "pipeline_outcome_entity", "correct_outcome_entity"),
    ]
    for field_name, pipe_key, correct_key in checks:
        pipe_val = pipeline.get(pipe_key)
        correct_val = judgment.get(correct_key)
        # Normalize None / "null" / empty string
        if pipe_val in (None, "null", "None", ""):
            pipe_val = None
        if correct_val in (None, "null", "None", ""):
            correct_val = None
        if field_name in ("subject_entity", "outcome_entity"):
            if pipe_val is not None:
                pipe_val = pipe_val.strip().lower()
            if correct_val is not None:
                correct_val = correct_val.strip().lower()
        if pipe_val != correct_val:
            changed.append(field_name)
    scope_errors = judgment.get("scope_errors", [])
    if scope_errors:
        changed.extend(f"scope_{s}" for s in scope_errors)
    return changed


def build_q1_requests(
    session: Any,
    pmcid: str,
    run_id: int,
    n_findings: int,
    global_seed: int,
) -> tuple[list[JudgeRequest], list[dict]]:
    """Build judge requests for Q1. Returns (requests, skipped)."""
    from nlp_histo.database import SumMapFinding  # noqa: PLC0415

    seed = stable_seed(global_seed, pmcid, str(run_id), TASK)

    findings = (
        session.query(SumMapFinding)
        .filter_by(pmcid=pmcid, pipeline_run_id=run_id)
        .all()
    )
    if not findings:
        logger.warning("Q1 [%s] no MAP findings", pmcid)
        return [], [{"pmcid": pmcid, "task": TASK, "reason": "no_map_findings"}]

    rng = random.Random(seed)
    sample = rng.sample(findings, min(n_findings, len(findings)))

    requests: list[JudgeRequest] = []
    skipped: list[dict] = []

    for f in sample:
        ck = JudgeCache.make_key(
            TASK,
            claim=f.claim,
            verbatim=f.verbatim_support,
            relation_type=f.relation_type,
            direction=f.direction,
            category=f.category,
            subject_entity=f.subject_entity,
            outcome_entity=f.outcome_entity,
        )

        user = Q1_USER.format(
            verbatim=f.verbatim_support,
            claim=f.claim,
            subject=f.subject_entity or "(none)",
            outcome=f.outcome_entity or "(none)",
            relation_type=f.relation_type,
            direction=f.direction or "null",
            category=f.category,
            scope_disease_subtype=getattr(f, "scope_disease_subtype", None) or "(none)",
            scope_assay_method=getattr(f, "scope_assay_method", None) or "(none)",
        )

        requests.append(JudgeRequest(
            custom_id=f"{TASK}_{ck}",
            cache_key=ck,
            task=TASK,
            system=Q1_SYSTEM,
            user=user,
            tool_name=Q1_TOOL_NAME,
            schema=Q1_SCHEMA,
            metadata={
                "pmcid": pmcid,
                "pipeline_run_id": run_id,
                "finding_id": f.id,
                "claim": f.claim,
                "verbatim_support": f.verbatim_support,
                "pipeline_relation_type": f.relation_type,
                "pipeline_direction": f.direction,
                "pipeline_category": f.category,
                "pipeline_subject_entity": f.subject_entity,
                "pipeline_outcome_entity": f.outcome_entity,
                "pipeline_grounding_score": f.grounding_score,
            },
        ))

    return requests, skipped


def parse_q1_result(request: JudgeRequest, judgment: dict) -> dict:
    """Merge pipeline metadata + Opus judgment + computed fields_changed."""
    row = {
        **request.metadata,
        **judgment,
        "fields_changed": _compute_fields_changed(request.metadata, judgment),
        "label_source": LABEL_SOURCE,
        "judge_model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
    }
    return row
