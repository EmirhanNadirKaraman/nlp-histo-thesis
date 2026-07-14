"""
Q2 — RELATE label accuracy.

Judges whether the pipeline's RELATE label (SUPPORT / CONTRADICT /
SCOPE_QUALIFY) is correct for persisted relation pairs.

NOTE (Phase 1): This test measures **precision** of persisted relations,
not relation recall.  The pipeline only persists non-UNRELATED pairs (those
that pass the comparability gate AND score above NLI thresholds).  Candidate
pairs that the pipeline labeled UNRELATED (or filtered before NLI) are not
in sum_relations, so this test cannot detect false-negative UNRELATED
labels.  Relation recall requires re-running NLI on all candidate pairs
with the judge — deferred to a future phase.

Blind judging by default: the pipeline label is NOT shown to Opus unless
--show-pipeline-label is set.
"""
from __future__ import annotations

import logging
import random
from typing import Any

from ..client import JudgeRequest
from ..cache import JudgeCache
from ..sampling import stable_seed
from ..prompts import (
    Q2_SYSTEM, Q2_USER_BLIND, Q2_USER_WITH_LABEL,
    Q2_TOOL_NAME, Q2_SCHEMA,
)
from .. import LABEL_SOURCE, MODEL, PROMPT_VERSION, SCHEMA_VERSION

logger = logging.getLogger(__name__)

TASK = "q2_relations"


def build_q2_requests(
    session: Any,
    pmcid: str,
    run_id: int,
    n_relations: int,
    global_seed: int,
    show_pipeline_label: bool = False,
) -> tuple[list[JudgeRequest], list[dict]]:
    """Build judge requests for Q2. Returns (requests, skipped)."""
    from nlp_histo.database import SumRelation, SumCanonicalRule  # noqa: PLC0415

    seed = stable_seed(global_seed, pmcid, str(run_id), TASK)

    relations = (
        session.query(SumRelation)
        .filter_by(pmcid=pmcid, pipeline_run_id=run_id)
        .all()
    )
    if not relations:
        logger.warning("Q2 [%s] no relations", pmcid)
        return [], [{"pmcid": pmcid, "task": TASK, "reason": "no_relations"}]

    cr_rows = (
        session.query(SumCanonicalRule)
        .filter_by(pmcid=pmcid, pipeline_run_id=run_id)
        .all()
    )
    cr_by_id: dict[str, Any] = {r.canonical_id: r for r in cr_rows}

    resolvable = [
        r for r in relations
        if r.rule_id_a in cr_by_id and r.rule_id_b in cr_by_id
    ]
    if not resolvable:
        logger.warning("Q2 [%s] no resolvable relations", pmcid)
        return [], [{"pmcid": pmcid, "task": TASK, "reason": "no_resolvable_relations"}]

    rng = random.Random(seed)
    sample = rng.sample(resolvable, min(n_relations, len(resolvable)))

    requests: list[JudgeRequest] = []
    skipped: list[dict] = []

    for rel in sample:
        a = cr_by_id[rel.rule_id_a]
        b = cr_by_id[rel.rule_id_b]

        ck = JudgeCache.make_key(
            TASK,
            predicate_a=a.predicate_text,
            predicate_b=b.predicate_text,
            subject_a=a.subject_entity,
            outcome_a=a.outcome_entity,
            subject_b=b.subject_entity,
            outcome_b=b.outcome_entity,
            show_label=show_pipeline_label,
        )

        fmt = dict(
            predicate_a=a.predicate_text,
            subject_a=a.subject_entity,
            outcome_a=a.outcome_entity,
            direction_a=a.direction or "null",
            category_a=a.category,
            predicate_b=b.predicate_text,
            subject_b=b.subject_entity,
            outcome_b=b.outcome_entity,
            direction_b=b.direction or "null",
            category_b=b.category,
        )
        if show_pipeline_label:
            fmt["pipeline_label"] = rel.relation_type
            user = Q2_USER_WITH_LABEL.format(**fmt)
        else:
            user = Q2_USER_BLIND.format(**fmt)

        requests.append(JudgeRequest(
            custom_id=f"{TASK}_{ck}",
            cache_key=ck,
            task=TASK,
            system=Q2_SYSTEM,
            user=user,
            tool_name=Q2_TOOL_NAME,
            schema=Q2_SCHEMA,
            metadata={
                "pmcid": pmcid,
                "pipeline_run_id": run_id,
                "relation_id": rel.id,
                "predicate_a": a.predicate_text,
                "predicate_b": b.predicate_text,
                "pipeline_label": rel.relation_type,
                "pipeline_nli_a_to_b": rel.nli_score_a_to_b,
                "pipeline_nli_b_to_a": rel.nli_score_b_to_a,
                "pipeline_label_shown": show_pipeline_label,
            },
        ))

    return requests, skipped


def parse_q2_result(request: JudgeRequest, judgment: dict) -> dict:
    """Merge pipeline metadata + Opus judgment + computed is_correct."""
    pipeline_label = request.metadata["pipeline_label"]
    correct_label = judgment.get("correct_label", "")
    row = {
        **request.metadata,
        **judgment,
        "is_correct": correct_label == pipeline_label,
        "label_source": LABEL_SOURCE,
        "judge_model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
    }
    return row
