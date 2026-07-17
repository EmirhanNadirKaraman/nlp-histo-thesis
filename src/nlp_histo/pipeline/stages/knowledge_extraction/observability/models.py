"""
Structured trace models for the knowledge_extraction pipeline observability layer.

One RunTrace is produced per paper/run.
One ChunkTrace is produced per MAP chunk (whether cache hit or live).

Both are serializable to JSONL via ``to_dict()`` → ``dataclasses.asdict()``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ── Chunk-level models ──────────────────────────────────────────────────────────


@dataclass
class VoterTrace:
    """One entry per voter LLM for a single chunk."""

    voter_index: int
    finding_count: int
    grounding_pass_fraction: float  # fraction of findings with at least one evidence citation
    mean_evidence_length: float     # mean len(Finding.evidence) across findings
    latency_ms: float | None = None
    validation_warnings: list[str] = field(default_factory=list)
    # "validated" = from router ProvenanceValidator; "fallback" = structural proxy
    grounding_source: str = "fallback"


@dataclass
class PairwiseScore:
    """
    Pairwise similarity between two voters (upper triangle of N×N matrix).

    The core fields (voter_i, voter_j, score) are always present.
    The breakdown fields are populated when EmbeddingSimilarityStrategy is used
    via compute_matrix_with_breakdown(); they are None / absent otherwise.
    """

    voter_i: int
    voter_j: int
    score: float
    # Alignment component breakdown (present when SemanticAgreementScorer
    # is used with EmbeddingSimilarityStrategy):
    claim_count_a: int | None = None
    claim_count_b: int | None = None
    coverage_a_to_b: float | None = None
    coverage_b_to_a: float | None = None
    base: float | None = None
    count_factor: float | None = None
    reuse_factor: float | None = None
    polarity_contradiction_ratio: float | None = None
    numeric_contradiction_ratio: float | None = None
    contradiction_ratio: float | None = None
    contradiction_factor: float | None = None
    pre_grounding_score: float | None = None
    grounding_factor: float | None = None


@dataclass
class AgreementTrace:
    """
    Full agreement computation record for one chunk.

    Directly answers:
    - "Why was this chunk escalated?" → ``decision`` + ``reason``
    - "Which pairwise score caused low agreement?" → ``pairwise_scores``
    - "Which voter was selected, and why?" → ``selected_voter_index`` + ``avg_sim``
    - "Which threshold blocked acceptance?" → ``deferral_score`` vs ``theta`` / ``reject_theta``
    """

    eligible_voter_indices: list[int]  # non-empty voters included in the matrix
    avg_sim: list[float]               # per-eligible-voter mean off-diagonal similarity
    pairwise_scores: list[PairwiseScore]  # upper triangle of the similarity matrix
    deferral_score: float              # max(avg_sim) — the gate score
    theta: float                       # accept threshold
    reject_theta: float                # hard-reject threshold
    decision: str                      # "keep" / "escalate" / "reject"
    reason: str                        # human-readable one-liner
    selected_voter_index: int | None   # global voter index, None when escalated


@dataclass
class ChunkTrace:
    """
    Per-chunk observability record (written to chunks.jsonl).

    One record is written for every chunk, including cache hits (voters/agreement
    are empty lists / None for cache hits).
    """

    chunk_id: str
    run_id: str
    pmcid: str
    te_ids: list[int]        # text element IDs covered by this chunk
    sentence_count: int
    text_preview: str        # first 200 chars of formatted chunk text
    cache_hit: bool
    voters: list[VoterTrace]
    agreement: AgreementTrace | None  # None on cache hits
    selected_voter_index: int | None  # None when escalated or cache hit
    escalated: bool
    escalation_level: int = 1  # 1=kept by L1, 2=kept by L2, 3=sent to L3

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ── Run-level models ────────────────────────────────────────────────────────────


@dataclass
class IngestionTrace:
    sentence_count: int
    te_count: int              # unique text element IDs
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChunkingTrace:
    total_chunks: int
    chunk_size: int            # configured sentences-per-chunk
    chunk_overlap: int = 0     # sentences shared between adjacent chunks


@dataclass
class MapStageTrace:
    total_chunks: int
    cache_hits: int
    cache_misses: int
    escalations: int           # chunks sent to L3 (final escalation)
    l2_escalations: int        # chunks that reached L2 (may have been kept there)
    keeps: int                 # chunks accepted from voters (L1 or L2)
    rejects: int               # hard-rejected chunks (deferral ≤ reject_theta)
    total_findings_out: int    # sum of findings across all accepted chunk summaries


@dataclass
class GroundingFilterTrace:
    stage: str           # "map_findings" or "rules"
    items_before: int
    items_after: int
    dropped: int


@dataclass
class ExportTrace:
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunTrace:
    """
    Top-level run record written to runs.jsonl — one line per paper.

    All stage traces are nested; any stage can be None if the run
    failed before reaching it or if tracing was not enabled for that stage.
    """

    run_id: str
    pmcid: str
    started_at: str       # ISO 8601 UTC
    ended_at: str | None
    duration_s: float | None
    status: str           # "success" | "error" | "skipped"
    error: str | None
    warnings: list[str]
    config_snapshot: dict[str, Any]
    ingestion: IngestionTrace | None
    chunking: ChunkingTrace | None
    map_stage: MapStageTrace | None
    grounding_map: GroundingFilterTrace | None
    export: ExportTrace | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
