"""Pydantic schemas for silver-label evaluation."""
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field


# ── Source (Opus input) ────────────────────────────────────────────────────────

class SourceCase(BaseModel):
    """One TextElement paragraph sampled for silver-label generation."""
    case_id: str          # "{pmcid}|{te_id}"
    pmcid: str
    te_id: int
    text: str
    path_string: str


# ── Silver findings (Opus output) ─────────────────────────────────────────────

class SilverFindingScope(BaseModel):
    """Mirrors pipeline FindingScope exactly."""
    disease_subtype:   str | None = None
    cohort_n:          int | None = None
    assay_method:      str | None = None
    biomarker_cutoff:  str | None = None
    tissue_site:       str | None = None
    treatment_context: str | None = None
    endpoint:          str | None = None
    study_design:      str | None = None
    scope_parsed:      bool       = False


class SilverFinding(BaseModel):
    category: Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"]
    claim: str
    subject_entity: str | None = None
    outcome_entity: str | None = None
    relation_type: str = "unclear"
    direction: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    verbatim_support: str = ""
    scope: SilverFindingScope = Field(default_factory=SilverFindingScope)
    source_sentence_ids: List[str] = Field(default_factory=list)


class SilverCaseResult(BaseModel):
    """One Opus-annotated case, stored in silver_findings.jsonl."""
    case_id: str
    pmcid: str
    te_id: int
    prompt_version: str
    model: str
    findings: List[SilverFinding]


# ── Pipeline output ────────────────────────────────────────────────────────────

class PipelineFinding(BaseModel):
    """One row from sum_map_findings, exported for a given case."""
    pipeline_run_id: int
    run_id: str          # human-readable run key
    pmcid: str
    chunk_id: str
    category: str
    claim: str
    subject_entity: str | None = None
    outcome_entity: str | None = None
    relation_type: str = "unclear"
    direction: str | None = None
    confidence: str = "medium"
    verbatim_support: str = ""
    evidence: list[str] = []  # ["S{i}|{pmcid}|{te_id}", …] — carried so grounding can DB-substitute (E03)
    grounding_score: float | None = None
    scope_disease_subtype: str | None = None
    scope_cohort_n: int | None = None
    scope_assay_method: str | None = None
    scope_biomarker_cutoff: str | None = None
    scope_tissue_site: str | None = None
    scope_treatment_context: str | None = None
    scope_endpoint: str | None = None
    scope_study_design: str | None = None


class PipelineCaseOutput(BaseModel):
    """All pipeline findings for a single case, stored in pipeline_findings.jsonl."""
    case_id: str
    pmcid: str
    te_id: int
    run_id: str
    pipeline_run_id: int
    findings: List[PipelineFinding]


# ── Matching / evaluation ──────────────────────────────────────────────────────

class FieldMismatch(BaseModel):
    field: str
    silver: str | None
    pipeline: str | None


class MatchedPair(BaseModel):
    silver_claim: str
    pipeline_claim: str
    similarity: float
    field_mismatches: List[FieldMismatch] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Matching result for a single case."""
    case_id: str
    pmcid: str
    te_id: int
    matched: List[MatchedPair]
    unmatched_silver: List[str]    # silver claims with no pipeline match
    unmatched_pipeline: List[str]  # pipeline claims with no silver match


class EvalMetrics(BaseModel):
    """Aggregate metrics across all cases."""
    n_cases: int
    n_silver_findings: int
    n_pipeline_findings: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    avg_similarity: float
    # Strict F1: matched pairs with important field mismatches count as 0.5 TP
    strict_f1: float = 0.0
    n_field_mismatches: int = 0
    similarity_threshold: float = 0.55
    prompt_version: str
    model: str
    pipeline_run_ids: List[int]
    # Split provenance
    split: str = "all"
    split_seed: int = 42
    dev_fraction: float = 0.8
    evaluated_case_ids: List[str] = Field(default_factory=list)
