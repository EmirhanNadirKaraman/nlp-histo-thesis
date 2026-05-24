"""
SummarizationConfig — single source of truth for all numeric/boolean knobs
in the summarization pipeline.

Usage
-----
from pipeline.stages.summarization.config import SummarizationConfig, MapConfig, ResolveConfig
from dataclasses import replace

cfg = SummarizationConfig()                         # all defaults
cfg = replace(cfg, resolve=replace(cfg.resolve, grounding_weight=0.70))

# Or build from scratch:
cfg = SummarizationConfig(
    map=MapConfig(theta=0.65, reject_theta=0.15),
    resolve=ResolveConfig(contradict_pen_per_rel=0.20),
)
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_TOKENS: int = 16384
"""Default max_tokens for all voter / escalation LLM calls in the MAP stage."""


@dataclass
class MapConfig:
    """ABC cascade + chunking knobs for the MAP stage."""

    theta: float = 0.8
    """Deferral score >= theta → KEEP without escalation."""

    reject_theta: float = 0.2
    """Deferral score <= reject_theta → hard REJECT the chunk."""

    chunk_size: int = 10
    """Sentences per MAP chunk."""

    chunk_overlap: int = 2
    """Sentences shared between adjacent chunks. Must be < chunk_size."""

    chunk_workers: int = 5
    """Max parallel threads for chunk processing."""

    enable_router: bool = False
    """Cascade-path selector. False → legacy ``AgreementChecker`` (theta /
    reject_theta deferral). True → grounding-first ``MapOutputRouter`` (L1→L3
    skip path). Pinned here so the cascade path is reproducible from the config
    alone and the sync (`build_runner`) and batch (`build_batch_runner`) entry
    points cannot silently diverge — both read this field. The MAP calibration
    sweep replays the legacy path only (`CASCADE_PATH='legacy_agreement_checker'`
    in `eval/silver/map_theta_sweep.py`), so any sweep-chosen (theta, reject_theta,
    scorer) must be re-validated against the router before flipping this True."""

    router_single_voter_policy: str = "escalate"
    """Only consulted when ``enable_router=True``: how ``MapOutputRouter`` treats
    a chunk with a single eligible voter. ``"escalate"`` (default) → send to L3;
    ``"keep"`` → accept the lone voter. Ignored on the legacy path."""


@dataclass
class NormalizeConfig:
    """NORMALIZE-stage knobs."""

    extra_synonyms: dict[str, str] | None = None
    """Caller-supplied surface-form → canonical overrides merged on top of
    `synonyms.yaml` at NormalizeStage construction. Lower-cased keys win over
    the bundled dict; lets callers patch a misclassification without editing
    the shipped YAML. None → no overrides."""


@dataclass
class GroundingConfig:
    """NLI entailment filter applied after MAP."""

    threshold: float | None = None
    """Minimum entailment score to keep a claim. None disables the filter."""


@dataclass
class RelateConfig:
    """NLI thresholds for the RELATE stage pairwise relation detection."""

    entailment_threshold: float = 0.50
    """Score above which a rule pair is SUPPORT or SCOPE_QUALIFY."""

    contradiction_threshold: float = 0.50
    """Score (both directions) above which a rule pair is CONTRADICT."""


@dataclass
class ResolveConfig:
    """
    Weighted scoring formula constants for the RESOLVE stage.

    Two modes — relations-present (when RELATE produced output) and
    relations-absent (single-paper run or RELATE skipped).
    """

    # ── Relations-present mode ────────────────────────────────────────────────
    grounding_weight: float = 0.60
    """Grounding score multiplier; max 0.60 contribution to final score."""

    grounding_default: float = 0.50
    """Fallback grounding when mean_grounding_score is None."""

    finding_bonus_max: float = 0.10
    """Max bonus from finding count (reached at finding_bonus_scale findings)."""

    finding_bonus_scale: int = 5
    """Findings needed to reach finding_bonus_max."""

    support_boost_per_rel: float = 0.08
    """Score bonus added per SUPPORT relation."""

    support_boost_cap: float = 0.20
    """Maximum total support bonus."""

    single_study_pen: float = 0.10
    """Penalty when study_coverage == 'single_study'."""

    contradict_pen_per_rel: float = 0.15
    """Score penalty per CONTRADICT relation."""

    contradict_pen_cap: float = 0.30
    """Maximum total contradiction penalty."""

    # ── Relations-absent mode ─────────────────────────────────────────────────
    no_rel_grounding_weight: float = 0.80
    """Grounding weight when no RELATE output (scores spread across [0, 1])."""

    no_rel_finding_bonus_max: float = 0.15
    """Finding bonus max in relations-absent mode."""

    no_rel_single_study_pen: float = 0.05
    """Single-study penalty in relations-absent mode (halved)."""


@dataclass
class CostConfig:
    """Runtime cost / usage accounting knobs."""

    enable_cost_report: bool = True
    """Collect LLM usage records during the run and write a cost report at end."""

    write_usage_jsonl: bool = True
    """Write the canonical llm_usage_records.jsonl. Source of truth for reports."""

    model_prices_path: str | None = None
    """Override path to model_prices.json. None → configs/model_prices.json at repo root."""

    cost_report_output_dir: str | None = None
    """Override output directory. None → run artifact dir, or output_dir/cost/{run_id}/."""


@dataclass
class AgreementConfig:
    """Soft-alignment weights for embedding-based MAP agreement scoring (H-EMB-01).

    Both ``EmbeddingSimilarityStrategy`` (production) and
    ``HybridStructuredSimilarity`` (the scorer-choice experiment's alternative)
    route claim alignment through ``agreement.embedding._align`` — surfacing
    these here is the precondition for sweeping the scorer. Defaults match the
    historical hardcoded values in ``_align`` so production behaviour is
    unchanged.
    """

    tau: float = 0.15
    """Weak-match threshold; pairwise similarities below tau are zeroed before coverage."""

    count_alpha: float = 0.25
    """Exponent for the soft claim-count-mismatch penalty (0.25 → 4:1 ratio ≈ 0.71×)."""

    reuse_weight: float = 0.15
    """Reuse-concentration penalty weight; at most a 15% reduction by default."""

    contradiction_weight: float = 0.20
    """Polarity-flip + numeric contradiction penalty weight; up to 20% reduction."""


@dataclass
class SummarizationConfig:
    """
    All numeric/boolean knobs for SummarizationRunner in one place.

    Operational params (output_dir, trace_enabled, db, LLM instances) stay
    on SummarizationRunner itself since they cannot be expressed as plain data.
    """

    map: MapConfig = field(default_factory=MapConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    relate: RelateConfig = field(default_factory=RelateConfig)
    resolve: ResolveConfig = field(default_factory=ResolveConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    agreement: AgreementConfig = field(default_factory=AgreementConfig)

    contradiction_similarity_threshold: float | None = 0.7
    """Cosine similarity threshold for ContradictionDetector candidate pairs.
    None disables contradiction detection entirely."""
