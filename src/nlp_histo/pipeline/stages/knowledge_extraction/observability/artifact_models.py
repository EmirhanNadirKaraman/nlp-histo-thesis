"""Pydantic models for trace/observability artifacts.

Distinct from `models.py` (which holds the LLM I/O schemas bound via
`with_structured_output`). Anything here is pipeline-internal and never sent
to or returned from an LLM, so adding fields cannot affect OpenAI strict
schema or cached LLM outputs.
"""
from __future__ import annotations

from pydantic import BaseModel


class SkippedPair(BaseModel):
    """One pre-NLI gate rejection in `RelateStage.relate()`.

    Recorded for offline debugging when `raw_pairs.jsonl` is empty or
    smaller than expected. The gate runs before any NLI scoring, so this
    record carries no threshold values — threshold-based rejections
    (UNRELATED) already appear in `raw_pairs.jsonl` with full scores.

    TODO: enrich with corpus-relate skip reasons once that stage emits a
    parallel trace (see PERSISTENCE_TODOS #5).
    """
    rule_id_a: str
    rule_id_b: str
    reason: str
    stage: str = "gate_pre_nli"
    pmcid: str | None = None

    category_a: str | None = None
    category_b: str | None = None
    relation_type_a: str | None = None
    relation_type_b: str | None = None
    subject_entity_a: str | None = None
    subject_entity_b: str | None = None
    outcome_entity_a: str | None = None
    outcome_entity_b: str | None = None
