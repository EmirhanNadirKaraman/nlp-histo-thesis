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

CONFIG_LAYOUT_VERSION: int = 2
"""Schema version for the SummarizationConfig dataclass *layout*.

Bumped when fields move between sub-dataclasses (vs. value changes, which are
captured by the per-field defaults + the pipeline_config_hash payload). Stamped
into every ``self._config_snapshot`` so analytical tooling that diffs
``manifest.json`` / ``PipelineRun.config_snapshot`` / TraceCollector JSONL
across runs can branch on the layout:

* **v1** — ``enable_router`` / ``router_single_voter_policy`` /
  ``legacy_single_voter_policy`` lived under ``map``.
* **v2** — those three fields moved to a dedicated ``RoutingConfig`` sub-config
  (2026-05-26).

Historical artifacts predate this stamp; readers should treat the absence of
``config_layout_version`` as v1."""


@dataclass
class MapConfig:
    """ABC cascade + chunking knobs for the MAP stage.

    Note: routing-policy fields (``enable_router``, ``router_single_voter_policy``,
    ``legacy_single_voter_policy``) moved to :class:`RoutingConfig` in
    layout v2 — they are routing/agreement-decision knobs, not MAP extraction
    parameters."""

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


@dataclass
class RoutingConfig:
    """Cascade-path + single-voter-policy knobs for the MAP stage.

    Separated from :class:`MapConfig` in layout v2 (2026-05-26) — these are
    routing / agreement-decision parameters, semantically distinct from MAP
    chunking / extraction. Production defaults preserve the v1 behaviour
    exactly: ``enable_router=False`` (legacy AgreementChecker path),
    ``router_single_voter_policy="escalate"``,
    ``legacy_single_voter_policy="keep"``."""

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

    legacy_single_voter_policy: str = "keep"
    """Only consulted when ``enable_router=False``: how ``AgreementChecker``
    treats a chunk where only one voter survived (e.g. the other voters' API
    calls failed). ``"keep"`` (default) preserves the prior implicit behaviour
    — the lone survivor is accepted with synthetic ``confidence=1.0`` and no
    peer comparison. ``"escalate"`` skips agreement and treats N=1 as
    low-evidence, escalating to the next cascade level (L1→L2, L2→L3). Mirror
    of ``router_single_voter_policy`` for the legacy path; the two are
    independent. Note the cause differs: the router's N=1 case follows
    schema/provenance gating (the survivor is *vetted*); the legacy N=1 case
    follows API survival (the survivor was not compared to anything). Default
    ``"keep"`` is the silent pre-existing behaviour in ``AgreementChecker``.
    Sweepable for the experiment: does the legacy path's silent ``"keep"``
    reward lucky surviving voters that should have escalated?"""


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

    scorer_kind: str = "embedding"
    """Similarity strategy selector for ``SemanticAgreementScorer``.

    * ``"embedding"`` (default) — ``EmbeddingSimilarityStrategy`` (production
      path; matches the historical hardcoded choice and the values pinned in
      ``configs/run.yaml`` 2026-05-26).
    * ``"hybrid"`` — ``HybridStructuredSimilarity`` (category + embedding +
      entity + evidence blend; defaults `0.25 / 0.40 / 0.25 / 0.10` baked into
      the constructor — *not* sweepable from YAML until ``AgreementConfig.hybrid``
      is promoted in a follow-up diff).

    The MAP-stage Stage 1 sweep (``run_summarization_sweeps --stage map_scorer``)
    already compares both strategies against silver F1; this field is what
    makes the Stage 1 winner pinnable in production YAML rather than only in
    the harness's ``BEST_SCORER`` constant. Any value outside
    ``{"embedding", "hybrid"}`` raises at runner construction time."""


@dataclass
class SummarizationConfig:
    """
    All numeric/boolean knobs for SummarizationRunner in one place.

    Operational params (output_dir, trace_enabled, db, LLM instances) stay
    on SummarizationRunner itself since they cannot be expressed as plain data.
    """

    map: MapConfig = field(default_factory=MapConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    relate: RelateConfig = field(default_factory=RelateConfig)
    resolve: ResolveConfig = field(default_factory=ResolveConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    agreement: AgreementConfig = field(default_factory=AgreementConfig)

    contradiction_similarity_threshold: float | None = 0.7
    """Cosine similarity threshold for ContradictionDetector candidate pairs.
    None disables contradiction detection entirely."""
