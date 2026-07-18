"""
Data models for asynchronous batch processing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from nlp_histo.pipeline.stages.knowledge_extraction.config import DEFAULT_MAX_TOKENS


class BatchPhase(str, Enum):
    """State machine phase for the cascading batch run."""
    L1_SUBMITTED = "l1_submitted"
    L2_SUBMITTED = "l2_submitted"
    L3_SUBMITTED = "l3_submitted"
    COMPLETE = "complete"


@dataclass
class VoterBatchConfig:
    """Config for one voter in batch mode."""
    model: str
    provider: str           # "azure" | "claude" | "gemini" | "vertex_gemini"
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    strip_thinking: bool = False  # strip <think>…</think> from response


@dataclass
class BatchRequest:
    """One LLM call packaged for a batch submission."""
    custom_id: str          # "{pmcid}__{chunk_id}__{level}__{voter_idx}"
    messages: list[dict]    # OpenAI-format role/content dicts
    model: str
    provider: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0


@dataclass
class BatchResult:
    """Raw result from one batch request."""
    custom_id: str
    content: str | None     # JSON string of the tool-call arguments
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProviderJob:
    """One submitted batch job at a specific provider."""
    provider: str
    job_id: str
    status: str             # "submitted" | "in_progress" | "completed" | "failed"
    model: str
    request_count: int
    output_location: str | None = None  # file_id (Azure), batch_id (Claude), GCS prefix (Vertex)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "job_id": self.job_id,
            "status": self.status,
            "model": self.model,
            "request_count": self.request_count,
            "output_location": self.output_location,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProviderJob:
        return cls(**d)


@dataclass
class BatchHandle:
    """
    JSON-serialisable state object persisted to disk between runs.

    The handle advances through BatchPhase values as batches complete.
    Call BatchKnowledgeExtractionRunner.advance(handle) to progress the pipeline.
    """
    pmcid: str
    phase: BatchPhase

    # Jobs belonging to the current phase (replaced on each advance)
    jobs: list[ProviderJob] = field(default_factory=list)

    # Original sentence provenance (needed for agreement re-computation)
    sentences: list[dict] = field(default_factory=list)

    # chunk_id → list of sentence dicts (built once on submit)
    chunk_map: dict[str, list[dict]] = field(default_factory=dict)

    # Voter index → strip_thinking flag, per level (needed when parsing results)
    l1_strip: list[bool] = field(default_factory=list)
    l2_strip: list[bool] = field(default_factory=list)
    l3_strip: bool = False

    # Raw batch content per custom_id (stored for debugging / re-parsing)
    l1_raw: dict[str, str] = field(default_factory=dict)
    l2_raw: dict[str, str] = field(default_factory=dict)
    l3_raw: dict[str, str] = field(default_factory=dict)

    # Which chunk_ids were escalated to each level
    l2_chunk_ids: list[str] = field(default_factory=list)
    l3_chunk_ids: list[str] = field(default_factory=list)

    # Final per-chunk AuditableSummary (serialised) — populated incrementally
    finalized: dict[str, dict] = field(default_factory=dict)

    # Actual token usage from API responses: level → model_id → {input, output}
    token_usage: dict[str, dict[str, dict[str, int]]] = field(default_factory=lambda: {
        "l1": {},
        "l2": {},
        "l3": {},
    })

    # In-run voter dedup: pre-satisfied next-level results carried over from an
    # earlier level. Generated when an L2 (or L3) voter has the same
    # (provider, model, temperature) as a voter that already ran at an earlier
    # level for the same chunk — instead of re-calling the API, we synthesise
    # a result here from the prior level's raw content. Merged into
    # raw_results during advance() so the agreement pass sees them as if they
    # had been retrieved from the batch provider.
    # Per entry: {"custom_id", "content", "input_tokens", "output_tokens", "error"}.
    synthetic_results: dict[str, list[dict]] = field(default_factory=lambda: {
        "l1": [],
        "l2": [],
        "l3": [],
    })

    # Run-artifact provenance — populated on submit() and persisted across resumes.
    schema_version:    str = ""
    prompt_version:    str = ""
    stage_name:        str = ""
    cascade_profile:   str = ""
    cascade_signature: str = ""

    # DB row id created at submit() so finalize() (possibly in a later process)
    # can FK persist + update status. None when db is unavailable or insert failed.
    pipeline_run_db_id: int | None = None

    # When True, submit() detected a valid on-disk result and short-circuited:
    # phase is COMPLETE but no batch jobs ran. finalize() returns the cached
    # JSON instead of re-running NORMALIZE/GROUP/... — protects against
    # re-spending L1/L2/L3 batch dollars on stale-cache invalidation flips.
    cached_result_only: bool = False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pmcid": self.pmcid,
            "phase": self.phase.value,
            "jobs": [j.to_dict() for j in self.jobs],
            "sentences": self.sentences,
            "chunk_map": self.chunk_map,
            "l1_strip": self.l1_strip,
            "l2_strip": self.l2_strip,
            "l3_strip": self.l3_strip,
            "l1_raw": self.l1_raw,
            "l2_raw": self.l2_raw,
            "l3_raw": self.l3_raw,
            "l2_chunk_ids": self.l2_chunk_ids,
            "l3_chunk_ids": self.l3_chunk_ids,
            "finalized": self.finalized,
            "token_usage": self.token_usage,
            "synthetic_results": self.synthetic_results,
            "schema_version":    self.schema_version,
            "prompt_version":    self.prompt_version,
            "stage_name":        self.stage_name,
            "cascade_profile":   self.cascade_profile,
            "cascade_signature": self.cascade_signature,
            "pipeline_run_db_id": self.pipeline_run_db_id,
            "cached_result_only": self.cached_result_only,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BatchHandle:
        data = json.loads(path.read_text(encoding="utf-8"))
        token_usage = data.get("token_usage", {"l1": {}, "l2": {}, "l3": {}})
        # Detect old flat schema: {"l1": {"input": N, "output": N}, ...}
        for level_val in token_usage.values():
            if isinstance(level_val, dict) and "input" in level_val:
                raise ValueError(
                    f"Handle at {path} has old token_usage schema (flat per-level). "
                    "Delete it and re-run to regenerate with per-model tracking."
                )
        return cls(
            pmcid=data["pmcid"],
            phase=BatchPhase(data["phase"]),
            jobs=[ProviderJob.from_dict(j) for j in data.get("jobs", [])],
            sentences=data.get("sentences", []),
            chunk_map=data.get("chunk_map", {}),
            l1_strip=data.get("l1_strip", []),
            l2_strip=data.get("l2_strip", []),
            l3_strip=data.get("l3_strip", False),
            l1_raw=data.get("l1_raw", {}),
            l2_raw=data.get("l2_raw", {}),
            l3_raw=data.get("l3_raw", {}),
            l2_chunk_ids=data.get("l2_chunk_ids", []),
            l3_chunk_ids=data.get("l3_chunk_ids", []),
            finalized=data.get("finalized", {}),
            token_usage=token_usage,
            synthetic_results=data.get(
                "synthetic_results", {"l1": [], "l2": [], "l3": []}
            ),
            schema_version=data.get("schema_version", ""),
            prompt_version=data.get("prompt_version", ""),
            stage_name=data.get("stage_name", ""),
            cascade_profile=data.get("cascade_profile", ""),
            cascade_signature=data.get("cascade_signature", ""),
            pipeline_run_db_id=data.get("pipeline_run_db_id"),
            cached_result_only=data.get("cached_result_only", False),
        )
