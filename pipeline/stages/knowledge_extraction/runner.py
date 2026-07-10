"""
KnowledgeExtractionRunner — orchestrates MAP → GROUNDING → NORMALIZE → GROUP
→ CANONICALIZE → RELATE → RESOLVE with ABC cascading.
REDUCE → RULES is an optional secondary block (disabled by default).

Usage example
-------------
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pipeline.stages.knowledge_extraction import KnowledgeExtractionRunner

from dataclasses import replace
from pipeline.stages.knowledge_extraction.config import KnowledgeExtractionConfig, MapConfig

cfg = KnowledgeExtractionConfig(map=MapConfig(theta=0.65))   # override just what you need

runner = KnowledgeExtractionRunner(
    voter_llms=[                                                   # Level 1: cheapest
        AzureChatOpenAI(model="DeepSeek-V3.2-Speciale",          temperature=0.0),
        VertexAI(model="gemini-2.5-flash-lite-preview-06-17",     temperature=0.0),
        AzureChatOpenAI(model="mistral-large-3",                  temperature=0.0),
    ],
    level2_voter_llms=[                                            # Level 2: mid-tier
        VertexAI(model="gemini-2.5-flash",                        temperature=0.0),
        AzureChatOpenAI(model="kimi-k2.5",                        temperature=0.0),
        ChatAnthropic(model="claude-haiku-4-5-20251001",          temperature=0.0),
    ],
    escalation_llm=ChatAnthropic(model="claude-sonnet-4-6", temperature=0),  # Level 3
    config=cfg,
    output_dir=Path("out/summaries"),
    trace_enabled=True,   # ← enable structured JSONL traces
)

# Single paper
file_data = KnowledgeExtractionRunner.load_paper_from_db("PMC10047158")
result = runner.process(file_data)

# Batch
pmcids = ["PMC10047158", "PMC10047213", "PMC10047408"]
results = runner.process_batch([KnowledgeExtractionRunner.load_paper_from_db(p) for p in pmcids])

# After a batch, export CSV summaries from the JSONL traces:
from pipeline.stages.knowledge_extraction.observability import export_all_csv
counts = export_all_csv(runner.trace_dir)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .cache import PipelineCache
from .config import KnowledgeExtractionConfig
from .stages.canonicalize_stage import CanonicalizeStage
from .helpers.contradiction_detector import ContradictionDetector
from .helpers.grounding_filter import GroundingFilter
from .stages.group_stage import GroupStage, is_groupable
from .stages.map_stage import MapStage
from .persistence import (
    RunArtifactWriter,
    RunManifest,
    build_rejection_summary as _build_rejection_summary,
    clear_normalized_run_data as _persistence_clear_normalized_run_data,
    create_pipeline_run as _persistence_create_pipeline_run,
    finish_pipeline_run as _persistence_finish_pipeline_run,
    persist_canonical_rules as _persistence_persist_canonical_rules,
    persist_canonicalize_artifacts,
    persist_final_rules as _persistence_persist_final_rules,
    persist_finding_groups as _persistence_persist_finding_groups,
    persist_group_artifacts,
    persist_map_artifacts,
    persist_map_findings as _persistence_persist_map_findings,
    persist_normal_findings as _persistence_persist_normal_findings,
    persist_normalize_artifacts,
    persist_rejection_summary as _persistence_persist_rejection_summary,
    persist_relate_artifacts,
    persist_relations as _persistence_persist_relations,
    persist_resolve_artifacts,
    replace_verbatim_from_db as _persistence_replace_verbatim_from_db,
)
from .models import (
    CanonicalRule,
    ConsolidatedSummary,
    ContradictionReport,
    ExtractedRules,
    FinalRule,
    Finding,
    FindingGroup,
    NormalFinding,
    Relation,
    RejectionSummary,
    compute_finding_id,
)
from .stages.normalize_stage import NormalizeStage
from .costing import PriceBook, UsageCollector, write_cost_reports
from .observability import TraceCollector, flush_collector
from .old_stages.reduce_stage import ReduceStage
from .stages.relate_stage import RelateStage
from .stages.resolve_stage import ResolveStage
from .old_stages.rule_stage import RuleStage
from pipeline.stages.knowledge_extraction.interfaces import (
    ContradictionChecker,
    GroundingChecker,
    MapOutputScorer,
)
from pipeline.utils.memory_logging import MemoryLogger

logger = logging.getLogger(__name__)


class KnowledgeExtractionRunner:
    """
    Full pipeline runner: MAP → GROUNDING → NORMALIZE → GROUP → CANONICALIZE
    → RELATE → RESOLVE.  REDUCE → RULES is an optional secondary block.

    Parameters
    ----------
    voter_llms:
        List of LangChain chat models used as Level-1 voters in the MAP stage.
        Use cheap models from different providers for genuine independence
        (e.g. [DeepSeek, Gemini-Flash-Lite, Mistral-Large]).
    level2_voter_llms:
        List of LangChain chat models used as Level-2 voters.  Called only when
        Level-1 voters disagree.  Use mid-tier models from different providers
        (e.g. [Gemini-Flash, kimi-k2.5, Haiku]).
    escalation_llm:
        LLM for MAP Level-3 (final) escalations, REDUCE, and RULES.
        Typically the most capable model (e.g. Sonnet 4.6).
    config:
        All numeric/boolean pipeline knobs.  Defaults to KnowledgeExtractionConfig()
        which provides calibrated defaults for all thresholds and scoring
        constants.  Use dataclasses.replace() to override specific fields.
    scorer:
        MapOutputScorer used to score voter agreement in the MAP stage.
        Defaults to ``SemanticAgreementScorer(EmbeddingSimilarityStrategy)``
        — Soiffer-style max-consensus with centrality-based best-output
        selection. Pass ``EmbeddingScorer`` for the legacy mean-pairwise
        behaviour, or ``CascadedCompositeScorer`` for the LP-thresholded
        embedding + NER cascade.
    output_dir:
        Where to write per-concept result JSON files.
    cache_path:
        Override for the cache file location.  Defaults to
        ``output_dir/pipeline_cache.json``.
    trace_enabled:
        When True, structured JSONL traces are written to ``trace_dir`` for
        every processed paper.
    trace_dir:
        Directory for JSONL trace files.  Defaults to ``output_dir/traces``.
    db:
        DatabaseConnection instance for persisting pipeline outputs.  All DB
        writes are no-ops when None.
    force_rerun:
        When True, ignores any cached result on disk and re-runs the full pipeline.
    run_ner:
        When True and db is not None, runs scispaCy NER + UMLS linking after the
        pipeline completes.
    run_reduce:
        When True, runs the optional REDUCE → RULES → contradiction-detection
        block after RESOLVE.  Disabled by default.
    """

    def __init__(
        self,
        voter_llms: list,
        level2_voter_llms: list,
        escalation_llm,
        config: KnowledgeExtractionConfig | None = None,
        scorer: MapOutputScorer | None = None,
        embed_fn=None,
        output_dir: Path = Path("langchain-knowledge_extraction/summarization_results"),
        cache_path: Path | None = None,
        trace_enabled: bool = False,
        trace_dir: Path | None = None,
        db=None,
        force_rerun: bool = False,
        run_ner: bool = True,
        run_reduce: bool = False,
        voter_specs:        list[tuple[str, str]] | None = None,
        level2_voter_specs: list[tuple[str, str]] | None = None,
        escalation_spec:    tuple[str, str] | None = None,
        cascade_profile:    str = "custom",
        artifact_root:      Path | None = None,
        artifact_run_id:    str | None = None,
        enable_router:      bool = False,
        router_single_voter_policy: str = "escalate",
        legacy_single_voter_policy: str = "keep",
    ) -> None:
        cfg = config or KnowledgeExtractionConfig()
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_path or (output_dir / "pipeline_cache.json")
        self._cache = PipelineCache(cache_file)

        # Per-paper cascade decision log directory. One JSONL file per pmcid,
        # appended to as L1/L2/L3 decisions are made. Surfaces cascade
        # behaviour for offline analysis without depending on trace_enabled.
        self._cascade_log_dir: Path = output_dir / "cascade_decisions"

        # Default scorer: SemanticAgreementScorer over the strategy selected by
        # cfg.agreement.scorer_kind (default "embedding" — preserves the historical
        # hardcoded choice). Centrality-based best-output selection (Soiffer 2025) —
        # the previous default was EmbeddingScorer, which left best_index unset and
        # forced AgreementChecker.best() to fall back to a
        # (mean_evidence_len, n_findings) heuristic. Pass embed_fn through so we
        # don't silently fall back to OpenAIEmbedder when the caller supplied a
        # different embedder. Invalid scorer_kind raises here.
        if scorer is None:
            from .agreement import SemanticAgreementScorer
            scorer = SemanticAgreementScorer.from_agreement_config(
                cfg.agreement, embed_fn=embed_fn,
            )

        # Optional routing layer: grounding-first MapOutputRouter. Drops voters
        # that fail schema or provenance validation before they enter the
        # agreement matrix, and forwards per-voter grounding quality into
        # AgreementContext so the scorer can use it for tie-breaking. Off by
        # default — the legacy 3-tier L1 → L2 → L3 cascade is the default
        # path. Enable with enable_router=True to switch to the L1 → L3 skip
        # path described in MapStage._cascade.
        router = None
        if enable_router:
            from .agreement import AgreementChecker
            from .routing import MapOutputRouter
            router_checker = AgreementChecker(
                scorer=scorer,
                theta=cfg.map.theta,
                reject_theta=cfg.map.reject_theta,
                force_escalate_on_polarity_conflict=(
                    cfg.agreement.force_escalate_on_polarity_conflict
                ),
            )
            router = MapOutputRouter(
                agreement_checker=router_checker,
                single_voter_policy=router_single_voter_policy,  # type: ignore[arg-type]
            )
        # Captured so _pipeline_config_hash can include router state — flipping
        # this knob changes the cascade behaviour (L1→L3 skip vs L1→L2→L3) and
        # must invalidate cached results.
        self._enable_router = enable_router
        self._router_single_voter_policy = router_single_voter_policy
        # Legacy-path counterpart: how AgreementChecker treats N=1 surviving
        # voters when the router is off. Default "keep" preserves prior
        # implicit behaviour; "escalate" sends single-survivor chunks up the
        # cascade. Hash-gated on `not enable_router` so flipping it doesn't
        # invalidate router-on runs.
        self._legacy_single_voter_policy = legacy_single_voter_policy

        self._map = MapStage(
            voter_llms, level2_voter_llms, escalation_llm,
            theta=cfg.map.theta,
            reject_theta=cfg.map.reject_theta,
            chunk_size=cfg.map.chunk_size,
            chunk_overlap=cfg.map.chunk_overlap,
            chunk_workers=cfg.map.chunk_workers,
            scorer=scorer,
            router=router,
            voter_specs=voter_specs,
            level2_voter_specs=level2_voter_specs,
            escalation_spec=escalation_spec,
            cascade_profile=cascade_profile,
            legacy_single_voter_policy=legacy_single_voter_policy,
            force_escalate_on_polarity_conflict=(
                cfg.agreement.force_escalate_on_polarity_conflict
            ),
            citation_config=cfg.citation,
        )
        self._normalize = NormalizeStage(extra_synonyms=cfg.normalize.extra_synonyms)
        self._group = GroupStage()
        self._canonicalize = CanonicalizeStage()
        self._relate = RelateStage(
            entailment_threshold=cfg.relate.entailment_threshold,
            contradiction_threshold=cfg.relate.contradiction_threshold,
            scope_aware_nli=cfg.relate.scope_aware_nli,
            use_verbatim_for_nli=cfg.relate.use_verbatim_for_nli,
        )
        self._resolve = ResolveStage(cfg.resolve)
        self._reduce = ReduceStage(escalation_llm)
        self._rules = RuleStage(escalation_llm)
        self._grounding: GroundingChecker | None = (
            GroundingFilter(cfg.grounding.threshold)
            if cfg.grounding.threshold is not None else None
        )
        self._contradiction: ContradictionChecker | None = (
            ContradictionDetector(
                escalation_llm,
                similarity_threshold=cfg.contradiction_similarity_threshold,
                embed_fn=embed_fn,
            )
            if cfg.contradiction_similarity_threshold is not None else None
        )

        self._cfg = cfg
        self._db = db  # DatabaseConnection | None — persistence is fully optional
        self._force_rerun = force_rerun
        self._run_ner = run_ner
        self._run_reduce = run_reduce

        self._trace_enabled = trace_enabled
        self.trace_dir: Path = trace_dir or (output_dir / "traces")

        # Filesystem artifact persistence — disabled when artifact_root is None.
        # When enabled, every stage output is mirrored to runs/{run_id}/.
        self._artifact_root: Path | None = (
            Path(artifact_root) if artifact_root is not None else None
        )
        self._artifact_run_id_override: str | None = artifact_run_id

        # Stores post-MAP findings (post-grounding when enabled, raw otherwise).
        # NORMALIZE reads this.
        self._scored_map_findings: dict[str, list[Finding]] = {}
        # Stores NORMALIZE output per pmcid. Input to GROUP.
        self._normal_findings: dict[str, list[NormalFinding]] = {}
        # Stores GROUP output per pmcid. Input to Phase 4 CANONICALIZE.
        self._finding_groups: dict[str, list[FindingGroup]] = {}
        # Stores CANONICALIZE output per pmcid. Input to Phase 5 RELATE.
        self._canonical_rules: dict[str, list[CanonicalRule]] = {}
        # Stores RELATE output per pmcid. Input to Phase 6 RESOLVE.
        self._relations: dict[str, list[Relation]] = {}
        # Stores raw NLI scores for all eligible pairs (including UNRELATED).
        self._relate_raw_pairs: dict[str, list] = {}
        # Stores pre-NLI gate skips per pmcid (SkippedPair[]).
        self._relate_skipped_pairs: dict[str, list] = {}
        # Stores RESOLVE output per pmcid. Final knowledge base.
        self._final_rules: dict[str, list[FinalRule]] = {}

        import dataclasses
        from .config import CONFIG_LAYOUT_VERSION
        # Snapshot of config for traces (model introspection is best-effort).
        # ``config_layout_version`` is stamped so manifest.json / PipelineRun.
        # config_snapshot / TraceCollector JSONL consumers can branch on the
        # dataclass shape (v1 had routing-policy fields under ``map``; v2 has
        # them under ``routing``). Absence ⇒ v1 by convention.
        self._config_snapshot = {
            **dataclasses.asdict(cfg),
            "config_layout_version": CONFIG_LAYOUT_VERSION,
            "scorer": type(scorer).__name__ if scorer else "SemanticAgreementScorer",
            "voter_model_count": len(voter_llms),
            "voter_models": [_model_name(m) for m in voter_llms],
            "level2_voter_model_count": len(level2_voter_llms),
            "level2_voter_models": [_model_name(m) for m in level2_voter_llms],
            "escalation_model": _model_name(escalation_llm),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def last_map_escalation_counts(self) -> dict[str, int]:
        """Escalation counts from the most recent process() call's MAP stage."""
        return self._map.last_escalation_counts

    @property
    def last_map_invocation_usage_records(self) -> list:
        """InvocationUsage records from the most recent process() call's MAP stage.

        One record per LLM invocation (voter calls + L3 escalations).  Cache
        hits add no record.  Reset at the start of every ``process()`` call.

        Used by ``scripts/run_paper.py`` to build the cost report — these
        records are the source of truth, replacing the LangChain callback
        path that ``with_structured_output`` strips.
        """
        return self._map.invocation_usage_records()

    @property
    def last_map_voter_output_records(self) -> list[dict]:
        """Per-voter AuditableSummary records from the most recent MAP run.

        One dict per (chunk, level, voter_index). Persisted into
        ``sum_map_voter_outputs`` by ``_persist_voter_outputs``. Cache hits
        contribute no records (the original voter outputs live with whichever
        earlier run populated the cache). Reset at the start of every
        ``process()`` call.
        """
        return self._map.voter_output_records()

    def process(
        self,
        file_data: dict,
        start_chunk: int = 0,
        limit_chunks: int | None = None,
    ) -> dict:
        """
        Run the full pipeline for one paper.

        ``file_data`` must have the shape produced by ``load_paper_from_db``:
        keys: pmcid, sentences_with_provenance (list of dicts).

        Parameters
        ----------
        start_chunk:
            Zero-based index of the first MAP chunk to process.  Useful for
            testing a single chunk without processing the full paper.
        limit_chunks:
            Maximum number of MAP chunks to process.  None = all chunks.
            Combine with start_chunk to process an arbitrary slice.

        Returns a result dict with keys:
            status, run_id, pmcid, summary, rules, contradiction_report,
            canonical_rules, relations, final_rules, audit_trail, rejection_summary
        or on failure:
            status='error', run_id, pmcid, error
        """
        pmcid = file_data["pmcid"]

        # Skip if already fully processed (bypass with force_rerun=True)
        if not self._force_rerun:
            existing = self._load_result(pmcid)
            if existing is not None:
                logger.info("[%s] skipped (cached result on disk)", pmcid)
                if self._trace_enabled:
                    collector = self._make_collector(pmcid)
                    flush_collector(collector, self.trace_dir, status="skipped")
                return existing

        run_id = self._artifact_run_id_override or self._make_run_id(pmcid)
        pipeline_run_db_id = self._create_pipeline_run(pmcid, run_id)

        collector: TraceCollector | None = (
            self._make_collector(pmcid, run_id) if self._trace_enabled else None
        )

        # Cost / usage accounting — opt-in via cfg.cost.enable_cost_report.
        usage_collector: UsageCollector | None = (
            UsageCollector(run_id=run_id, paper_id=pmcid)
            if self._cfg.cost.enable_cost_report else None
        )

        # Filesystem artifacts — opt-in via artifact_root. Returns None when disabled.
        writer = self._make_artifact_writer(run_id, pmcid)
        prev_log_dir = os.environ.get("NLP_HISTO_LOG_DIR")
        if writer is not None:
            os.environ["NLP_HISTO_LOG_DIR"] = str(writer.logs_dir())

        mem = MemoryLogger(pmcid=pmcid)
        mem.checkpoint("pipeline", "start")

        try:
            t_total = time.perf_counter()
            sentences = file_data["sentences_with_provenance"]

            # Record ingestion
            if collector is not None:
                te_ids = {s.get("text_element_id", 0) for s in sentences}
                collector.record_ingestion(
                    sentence_count=len(sentences),
                    te_count=len(te_ids),
                )

            # 1. MAP (ABC cascade per chunk)
            logger.info("[%s] MAP — %d sentences", pmcid, len(sentences))
            t0 = time.perf_counter()
            # Per-paper cascade decision log: one JSONL row per L1/L2/L3
            # decision. Always emitted (independent of trace_enabled) so
            # cascade behaviour is inspectable on every run.
            from .agreement import CascadeDecisionLog as _CascadeDecisionLog
            cascade_log = _CascadeDecisionLog(
                self._cascade_log_dir / f"{pmcid}.jsonl"
            )
            with mem.stage("MAP"):
                chunk_summaries = self._map.process(
                    sentences, pmcid, cache=self._cache, collector=collector,
                    start_chunk=start_chunk, limit_chunks=limit_chunks,
                    usage_collector=usage_collector,
                    cascade_decision_log=cascade_log,
                    run_id=run_id,
                )
            logger.info("[%s] MAP done [%.1fs] — %d chunks, %d raw findings",
                        pmcid, time.perf_counter() - t0,
                        len(chunk_summaries),
                        sum(len(cs.findings) for cs in chunk_summaries))

            # Record chunking info (chunk_size is on MapStage)
            if collector is not None:
                collector.record_chunking(
                    total_chunks=len(chunk_summaries),
                    chunk_size=self._map.chunk_size,
                    chunk_overlap=self._map.chunk_overlap,
                )

            # Track total MAP findings for rejection summary (before any filtering)
            map_findings_total = sum(len(cs.findings) for cs in chunk_summaries)

            # Assign stable finding_id BEFORE grounding so downstream stages
            # (and rejected-finding artifacts) share the same lineage key. The
            # id is deterministic over (pmcid, chunk_id, position, claim), so
            # grounding-induced drops do not shift surviving ids.
            for cs in chunk_summaries:
                for pos, f in enumerate(cs.findings):
                    f.set_finding_id(
                        compute_finding_id(pmcid, cs.chunk_id, pos, f.claim)
                    )

            # 1a-pre. Replace LLM-paraphrased verbatim_support with actual source text from DB.
            # Evidence strings carry the text_element_id; we batch-query TextElement.text_content
            # so NLI grounding scores real paragraphs rather than LLM paraphrases.
            self._replace_verbatim_from_db(chunk_summaries)

            # 1a. Grounding filter — score + drop ungrounded findings before NORMALIZE
            grounding_rejected: list[tuple[str, Finding]] = []  # (chunk_id, finding)
            if self._grounding is not None:
                findings_before = map_findings_total
                logger.info("[%s] GROUNDING (filter+score) — %d findings", pmcid, findings_before)
                t0 = time.perf_counter()
                with mem.stage("GROUNDING"):
                    new_summaries = []
                    for cs in chunk_summaries:
                        kept_cs, dropped = self._grounding.filter_findings_with_scores(cs)
                        new_summaries.append(kept_cs)
                        for f in dropped:
                            grounding_rejected.append((cs.chunk_id, f))
                chunk_summaries = new_summaries
                findings_after = sum(len(cs.findings) for cs in chunk_summaries)
                logger.info(
                    "[%s] GROUNDING done [%.1fs] — %d/%d findings kept (%d rejected)",
                    pmcid, time.perf_counter() - t0,
                    findings_after, findings_before, len(grounding_rejected),
                )
                if collector is not None:
                    collector.record_grounding(
                        stage="map_findings",
                        items_before=findings_before,
                        items_after=findings_after,
                    )

                all_findings = [f for cs in chunk_summaries for f in cs.findings]
                self._scored_map_findings[pmcid] = all_findings
                logger.info("[%s] chunk_ids before persist: %s", pmcid, [cs.chunk_id for cs in chunk_summaries])
                self._persist_map_findings(pipeline_run_db_id, pmcid, chunk_summaries)
            else:
                # Grounding disabled — still populate so NORMALIZE receives MAP findings.
                self._scored_map_findings[pmcid] = [
                    f for cs in chunk_summaries for f in cs.findings
                ]

            # Filesystem MAP artifacts (opt-in)
            self._persist_map_artifacts(
                writer, pmcid, chunk_summaries, grounding_rejected,
            )

            # Per-voter outputs → sum_map_voter_outputs (captures the
            # AuditableSummary each model produced, including non-winners).
            self._persist_voter_outputs(pipeline_run_db_id, pmcid)

            # 1b. NORMALIZE — entity normalization + conditional dedup
            n_scored = len(self._scored_map_findings.get(pmcid, []))
            logger.info("[%s] NORMALIZE — %d scored findings", pmcid, n_scored)
            t0 = time.perf_counter()
            with mem.stage("NORMALIZE"):
                self._normal_findings[pmcid] = self._normalize.normalize(
                    self._scored_map_findings.get(pmcid, []), pmcid
                )
            logger.info("[%s] NORMALIZE done [%.1fs] — %d NormalFindings",
                        pmcid, time.perf_counter() - t0, len(self._normal_findings[pmcid]))
            _nf_db_id_map = self._persist_normal_findings(
                pipeline_run_db_id, pmcid, self._normal_findings[pmcid]
            )
            self._persist_normalize_artifacts(writer, pmcid, self._normal_findings[pmcid])

            # 1c. GROUP — bucket groupable NormalFindings by (subject, outcome, relation_type, category)
            all_normal = self._normal_findings[pmcid]
            groupable = [nf for nf in all_normal if is_groupable(nf)]
            non_groupable_nfs = [nf for nf in all_normal if not is_groupable(nf)]
            logger.info(
                "[%s] GROUP — %d groupable, %d non-groupable",
                pmcid, len(groupable), len(non_groupable_nfs),
            )
            t0 = time.perf_counter()
            with mem.stage("GROUP"):
                self._finding_groups[pmcid] = self._group.group(groupable, pmcid)
            logger.info("[%s] GROUP done [%.1fs] — %d groups",
                        pmcid, time.perf_counter() - t0, len(self._finding_groups[pmcid]))
            _fg_db_id_map = self._persist_finding_groups(
                pipeline_run_db_id, pmcid, self._finding_groups[pmcid], _nf_db_id_map
            )
            self._persist_group_artifacts(
                writer, pmcid, self._finding_groups[pmcid], non_groupable_nfs,
            )

            # 1d. CANONICALIZE — FindingGroup[] → CanonicalRule[]
            logger.info("[%s] CANONICALIZE — %d groups", pmcid, len(self._finding_groups[pmcid]))
            t0 = time.perf_counter()
            with mem.stage("CANONICALIZE"):
                nf_by_id: dict[str, NormalFinding] = {
                    nf.normal_id: nf for nf in all_normal
                }
                self._canonical_rules[pmcid] = self._canonicalize.canonicalize(
                    self._finding_groups[pmcid], nf_by_id, pmcid
                )
            logger.info("[%s] CANONICALIZE done [%.1fs] — %d CanonicalRules",
                        pmcid, time.perf_counter() - t0, len(self._canonical_rules[pmcid]))

            # Enrich canonical rules with UMLS CUIs for cross-paper entity matching.
            # No-ops silently if scispacy is unavailable.
            from .helpers.entity_linker import enrich_rules_with_cuis  # noqa: PLC0415
            with mem.stage("UMLS_ENRICH"):
                enrich_rules_with_cuis(self._canonical_rules[pmcid])
            _cr_db_id_map = self._persist_canonical_rules(
                pipeline_run_db_id, pmcid, self._canonical_rules[pmcid], _fg_db_id_map
            )
            self._persist_canonicalize_artifacts(
                writer, pmcid, self._canonical_rules[pmcid],
            )
            self._corpus_relate_incremental(pmcid, self._canonical_rules[pmcid])

            # 1e. RELATE — CanonicalRule[] → Relation[]
            logger.info(
                "[%s] RELATE — %d canonical rules", pmcid, len(self._canonical_rules[pmcid])
            )
            t0 = time.perf_counter()
            with mem.stage("RELATE"):
                (
                    self._relations[pmcid],
                    self._relate_raw_pairs[pmcid],
                    relate_skipped,
                ) = self._relate.relate(self._canonical_rules[pmcid], pmcid)
            self._relate_skipped_pairs[pmcid] = relate_skipped
            logger.info(
                "[%s] RELATE done [%.1fs] — %d relations, %d raw pairs, %d gate-skipped",
                pmcid, time.perf_counter() - t0,
                len(self._relations[pmcid]),
                len(self._relate_raw_pairs[pmcid]),
                len(relate_skipped),
            )
            self._persist_relations(
                pipeline_run_db_id, pmcid, self._relations[pmcid], _cr_db_id_map
            )
            self._persist_relate_artifacts(
                writer, pmcid,
                self._relations[pmcid],
                self._relate_raw_pairs[pmcid],
                self._relate_skipped_pairs[pmcid],
            )

            # 1f. RESOLVE — CanonicalRule[] + Relation[] → FinalRule[]
            logger.info(
                "[%s] RESOLVE — %d canonical rules, %d relations",
                pmcid, len(self._canonical_rules[pmcid]), len(self._relations[pmcid]),
            )
            t0 = time.perf_counter()
            with mem.stage("RESOLVE"):
                self._final_rules[pmcid] = self._resolve.resolve(
                    self._canonical_rules[pmcid], self._relations[pmcid], pmcid
                )
            logger.info("[%s] RESOLVE done [%.1fs] — %d FinalRules",
                        pmcid, time.perf_counter() - t0, len(self._final_rules[pmcid]))
            self._persist_final_rules(
                pipeline_run_db_id, pmcid, self._final_rules[pmcid], _cr_db_id_map
            )
            self._persist_resolve_artifacts(
                writer, pmcid, self._final_rules[pmcid], self._relations[pmcid],
            )

            # 2. REDUCE + RULES (optional — disabled by default)
            master: ConsolidatedSummary | None = None
            rules: ExtractedRules | None = None
            contradiction_report: ContradictionReport | None = None

            if self._run_reduce:
                logger.info("[%s] REDUCE — %d chunks", pmcid, len(chunk_summaries))
                t0 = time.perf_counter()
                with mem.stage("REDUCE"):
                    master = self._reduce.reduce(
                        chunk_summaries, pmcid, cache=self._cache, collector=collector
                    )
                logger.info("[%s] REDUCE done [%.1fs]", pmcid, time.perf_counter() - t0)

                # 3. RULE EXTRACTION
                logger.info("[%s] RULES", pmcid)
                t0 = time.perf_counter()
                with mem.stage("RULES"):
                    rules = self._rules.extract(
                        master, pmcid, cache=self._cache, collector=collector
                    )
                logger.info("[%s] RULES done [%.1fs] — %d rules",
                            pmcid, time.perf_counter() - t0, len(rules.rules))

                # 3a. Grounding filter — drop ungrounded rules
                if self._grounding is not None:
                    rules_before = len(rules.rules)
                    logger.info("[%s] GROUNDING (rules) — %d rules", pmcid, rules_before)
                    t0 = time.perf_counter()
                    rules = self._grounding.filter_rules(rules)
                    rules_after = len(rules.rules)
                    logger.info("[%s] GROUNDING (rules) done [%.1fs] — %d/%d rules kept",
                                pmcid, time.perf_counter() - t0, rules_after, rules_before)
                    if collector is not None:
                        collector.record_grounding(
                            stage="rules",
                            items_before=rules_before,
                            items_after=rules_after,
                        )

                # 4. Contradiction detection
                if self._contradiction is not None:
                    logger.info("[%s] CONTRADICTION DETECTION", pmcid)
                    t0 = time.perf_counter()
                    contradiction_report = self._contradiction.detect(rules)
                    logger.info("[%s] CONTRADICTION DETECTION done [%.1fs]",
                                pmcid, time.perf_counter() - t0)
            else:
                logger.info("[%s] REDUCE/RULES skipped (run_reduce=False)", pmcid)

            # NER + UMLS linking (optional)
            if self._run_ner and self._db is not None:
                logger.info("[%s] NER — running entity extraction + UMLS linking", pmcid)
                t0 = time.perf_counter()
                mem.checkpoint("NER", "before")
                try:
                    from named_entity_recognition.ner import run_ner_on_db
                    run_ner_on_db(pmcid, save_to_db=True, force=False)
                    logger.info("[%s] NER done [%.1fs]", pmcid, time.perf_counter() - t0)
                    mem.checkpoint("NER", "after")
                except Exception as ner_exc:
                    logger.warning("[%s] NER failed (non-fatal): %s", pmcid, ner_exc)
                    mem.checkpoint("NER", "failed")

            mem.checkpoint("pipeline", "end")
            logger.info("[%s] Pipeline complete [%.1fs total]",
                        pmcid, time.perf_counter() - t_total)

            rejection_summary = _build_rejection_summary(
                pmcid=pmcid,
                grounding_threshold=(
                    self._grounding.threshold if self._grounding is not None else None
                ),
                map_findings_total=map_findings_total,
                grounding_rejected=grounding_rejected,
                normal_findings=self._normal_findings.get(pmcid, []),
                non_groupable_nfs=non_groupable_nfs,
            )
            self._persist_rejection_summary(pipeline_run_db_id, rejection_summary)

            result = {
                "status": "success",
                "run_id": run_id,
                "pmcid": pmcid,
                # Phase 4–6 knowledge base
                "canonical_rules": [
                    r.model_dump() for r in self._canonical_rules.get(pmcid, [])
                ],
                "relations": [
                    r.model_dump() for r in self._relations.get(pmcid, [])
                ],
                "relate_raw_pairs": [
                    p.model_dump() for p in self._relate_raw_pairs.get(pmcid, [])
                ],
                "final_rules": [
                    r.model_dump() for r in self._final_rules.get(pmcid, [])
                ],
                "audit_trail": {
                    "map_chunks": [cs.model_dump() for cs in chunk_summaries],
                },
                "map_run_metadata": self._map.run_metadata_summary(),
                "rejection_summary": rejection_summary.model_dump(),
            }
            # Legacy REDUCE / RULES output is only included when explicitly run.
            if self._run_reduce:
                result["summary"] = master.narrative_summary if master else None
                result["rules"] = [r.model_dump() for r in rules.rules] if rules else []
                result["contradiction_report"] = (
                    contradiction_report.model_dump() if contradiction_report else None
                )
                result["audit_trail"]["master_summary"] = (
                    master.model_dump() if master else None
                )
                result["audit_trail"]["rules_provenance"] = (
                    rules.model_dump() if rules else None
                )
            self._finish_pipeline_run(
                pipeline_run_db_id, "success",
                narrative_summary=master.narrative_summary if master else None,
            )
            self._save_result(result)
            self._cache.save()

            if collector is not None:
                result_path = str(self._result_path(pmcid))
                collector.add_artifact(result_path, "result_json")
                flush_collector(collector, self.trace_dir, status="success")

            self._write_cost_artifacts(usage_collector, writer, run_id, pmcid)

            # Issue G — post-run row-count audit. Catches the partial-
            # persistence failure mode (one _persist_* swallowed an
            # exception, others wrote rows → tables out of sync). Never
            # raises; emits WARNING if any table is below the expected
            # row count for this paper.
            if self._db is not None:
                try:
                    from .health_checks import assert_persistence_row_counts  # noqa: PLC0415
                    assert_persistence_row_counts(self._db, pmcid)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[%s] Post-run row-count audit raised — ignored: %s",
                        pmcid, exc,
                    )

            if writer is not None:
                writer.finalize("completed")

            return result

        except KeyboardInterrupt:
            logger.warning("[%s] Pipeline interrupted (KeyboardInterrupt)", pmcid)
            mem.checkpoint(mem.last_stage or "pipeline", "interrupted")
            self._finish_pipeline_run(pipeline_run_db_id, "interrupted", error="KeyboardInterrupt")
            if collector is not None:
                flush_collector(collector, self.trace_dir, status="interrupted", error="KeyboardInterrupt")
            if writer is not None:
                writer.finalize("failed", error="KeyboardInterrupt")
            raise
        except Exception as exc:
            mem.checkpoint(mem.last_stage or "pipeline", "failed")
            logger.exception("[%s] Pipeline failed: %s", pmcid, exc)
            self._finish_pipeline_run(pipeline_run_db_id, "failed", error=str(exc))
            if collector is not None:
                flush_collector(collector, self.trace_dir, status="error", error=str(exc))
            self._write_cost_artifacts(usage_collector, writer, run_id, pmcid)
            if writer is not None:
                writer.write_error(stage="pipeline", error=str(exc), pmcid=pmcid)
                writer.finalize("failed", error=str(exc))
            return {"status": "error", "run_id": run_id, "pmcid": pmcid, "error": str(exc)}
        finally:
            # Flush MAP/REDUCE/RULE in-memory cache to disk on every exit path
            # (success, error, or KeyboardInterrupt) so partially-completed
            # paper runs don't re-pay for chunks already scored. Safe to call
            # even on the success path — _save_result already returned and
            # this is a second write to a separate file.
            try:
                self._cache.save()
            except Exception as exc:
                logger.warning("[%s] cache flush in finally: failed (non-fatal): %s", pmcid, exc)
            # B-009 — drop per-paper state from the runner-level dicts after
            # the result has been materialised.  Without this, process_batch
            # accumulates every paper's NormalFindings, FindingGroups,
            # CanonicalRules, Relations, raw NLI pairs, skipped pairs, and
            # FinalRules on `self` for the lifetime of the runner — memory
            # grows O(papers × avg eligible pairs).  Pop instead of clear
            # so a re-entrant call for the same pmcid (force_rerun) starts
            # from a clean slate.
            for store in (
                self._scored_map_findings,
                self._normal_findings,
                self._finding_groups,
                self._canonical_rules,
                self._relations,
                self._relate_raw_pairs,
                self._relate_skipped_pairs,
                self._final_rules,
            ):
                store.pop(pmcid, None)
            # Restore NLP_HISTO_LOG_DIR — never leave global env state changed.
            if writer is not None:
                if prev_log_dir is None:
                    os.environ.pop("NLP_HISTO_LOG_DIR", None)
                else:
                    os.environ["NLP_HISTO_LOG_DIR"] = prev_log_dir

    def process_batch(self, file_data_list: list[dict]) -> list[dict]:
        """
        Run the pipeline over a list of papers and return all results.
        Logs a summary at the end.
        """
        results = []
        for i, fd in enumerate(file_data_list, 1):
            logger.info("--- [%d/%d] %s ---", i, len(file_data_list), fd.get("pmcid", "?"))
            results.append(self.process(fd))

        n_ok = sum(1 for r in results if r["status"] == "success")
        n_err = sum(1 for r in results if r["status"] == "error")
        n_skip = sum(1 for r in results if r["status"] == "skipped")
        logger.info(
            "Batch complete: %d ok / %d skipped (cached) / %d errors",
            n_ok, n_skip, n_err,
        )
        logger.info(self._cache.stats_str())
        return results

    def corpus_relate(
        self,
        source_dir: Path | None = None,
        output_path: Path | None = None,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
        run_selection: str = "latest_per_pmcid",
        manifest: list[Path] | None = None,
    ) -> list:
        """
        Run the corpus-level relation stage over all per-paper JSONs in source_dir.

        Pools canonical_rules from every per-paper JSON, runs the same NLI-based
        pairwise comparison used by the per-paper RELATE stage, and writes a
        single corpus_relations.json artifact.  Each relation is labeled as
        "intra_paper" or "cross_paper" based on whether both rules came from the
        same paper.

        This is a post-hoc analytical step.  It does NOT modify any per-paper
        output and does NOT currently affect ResolveStage scoring, which only
        receives per-paper relations (see DES-1a in KNOWN_ISSUES.md).

        Parameters
        ----------
        source_dir:
            Directory containing per-paper JSON files.
            Defaults to ``self._summaries_dir``.
        output_path:
            Destination for corpus_relations.json.
            Defaults to ``self._output_dir / "corpus_relations.json"``.
        entailment_threshold:
            NLI entailment threshold forwarded to RelateStage.
            Defaults to the threshold used by the per-paper RELATE stage.
        contradiction_threshold:
            NLI contradiction threshold.  Defaults similarly.
        run_selection:
            "latest_per_pmcid" (default) — one representative run per PMCID.
            "all" — every valid file (status=success, non-empty canonical_rules).
            Ignored when manifest is provided.
        manifest:
            Explicit list of JSON paths to load.  When provided, source_dir and
            run_selection are both ignored.

        Returns
        -------
        List of CorpusRelation objects written to output_path.
        """
        from .helpers.corpus_relate import CorpusRelateStage  # noqa: PLC0415 — lazy import

        src = source_dir or self._summaries_dir
        out = output_path or (self._output_dir / "corpus_relations.json")

        stage = CorpusRelateStage(
            entailment_threshold=(
                entailment_threshold
                if entailment_threshold is not None
                else self._relate._entailment_threshold
            ),
            contradiction_threshold=(
                contradiction_threshold
                if contradiction_threshold is not None
                else self._relate._contradiction_threshold
            ),
        )
        return stage.relate_from_dir(src, out, run_selection=run_selection, manifest=manifest)

    # ── Disk I/O ───────────────────────────────────────────────────────────────

    @staticmethod
    def load_paper_from_db(pmcid: str, db_url: str | None = None) -> dict:
        """
        Load all text elements for ``pmcid`` from the database and split them
        into sentence-level provenance dicts ready for ``process()``.

        Args:
            pmcid:   PubMed Central ID (e.g. "PMC10047158").
            db_url:  Optional SQLAlchemy database URL.  Defaults to the value
                     in the project .env file.

        Returns:
            dict with keys ``pmcid`` and ``sentences_with_provenance``.
        """
        from database import get_db_connection, Document, TextElement  # type: ignore
        from .umls_resources import get_small_nlp

        # B-038 fix: route through the process-wide small-model singleton so
        # every call (especially in batch mode where this runs once per paper)
        # reuses the same loaded pipeline. Direct `spacy.load(...)` calls
        # bypass the cache and re-deserialise the model from disk per paper.
        nlp = get_small_nlp("en_core_sci_sm")
        if nlp is None:
            raise RuntimeError(
                "en_core_sci_sm not available for sentence segmentation in "
                "load_paper_from_db. Install with: "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
            )
        db = get_db_connection(database_url=db_url)

        with db.session_scope() as session:
            doc = session.query(Document).filter_by(pmcid=pmcid).first()
            if doc is None:
                raise ValueError(f"PMCID {pmcid!r} not found in database")
            rows = (
                session.query(TextElement)
                .filter_by(document_id=doc.id)
                .order_by(TextElement.id)
                .all()
            )
            sentences = []
            for te in rows:
                for sent in nlp(te.text_content).sents:
                    text = sent.text.strip()
                    if text:
                        sentences.append({
                            "pmcid": pmcid,
                            "text_element_id": te.id,
                            "sentence": text,
                        })

        return {"pmcid": pmcid, "sentences_with_provenance": sentences}

    def _make_run_id(self, pmcid: str) -> str:
        """Generate a human-readable run identifier: {pmcid}_{YYYYMMDDTHHmmss}."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{pmcid}_{ts}"

    def _make_collector(self, pmcid: str, run_id: str | None = None) -> TraceCollector:
        if run_id is None:
            run_id = self._make_run_id(pmcid)
        return TraceCollector(
            run_id=run_id,
            pmcid=pmcid,
            config_snapshot=self._config_snapshot,
        )

    def _create_pipeline_run(self, pmcid: str, run_id: str) -> int | None:
        return _persistence_create_pipeline_run(
            self._db, pmcid, run_id, self._config_snapshot,
        )

    def _finish_pipeline_run(
        self,
        db_id: int | None,
        status: str,
        error: str | None = None,
        narrative_summary: str | None = None,
    ) -> None:
        _persistence_finish_pipeline_run(
            self._db, db_id, status, error=error, narrative_summary=narrative_summary,
        )

    # ── Phase 2: DB persistence ────────────────────────────────────────────────

    def _clear_normalized_run_data(self, db_id: int, pmcid: str) -> None:
        _persistence_clear_normalized_run_data(self._db, db_id, pmcid)

    def _replace_verbatim_from_db(self, chunk_summaries: list) -> None:
        _persistence_replace_verbatim_from_db(self._db, chunk_summaries)

    # ── Filesystem artifact persistence (opt-in) ───────────────────────────────

    def _make_artifact_writer(
        self, run_id: str, pmcid: str,
    ) -> RunArtifactWriter | None:
        """Build the per-run filesystem writer when ``artifact_root`` is set.

        Returns None when persistence is disabled — every artifact helper is a
        no-op against a None writer. Failure to create the writer is logged but
        not fatal: filesystem persistence is observational and must never block
        the pipeline.
        """
        if self._artifact_root is None:
            return None
        try:
            from .models import MAP_SCHEMA_VERSION, MAP_PROMPT_VERSION  # noqa: PLC0415
            cfg = self._cfg
            map_meta = self._map.run_metadata_summary() or {}
            cascade_signature = (
                map_meta.get("cascade_signature")
                or self._config_snapshot.get("cascade_signature")
            )
            # Seed from module constants so the manifest is populated even
            # when MAP hasn't run yet; richer values from run_metadata_summary
            # win when present.
            schema_version = map_meta.get("schema_version") or MAP_SCHEMA_VERSION
            prompt_version = map_meta.get("prompt_version") or MAP_PROMPT_VERSION

            thresholds = {
                "grounding_threshold": (
                    self._grounding.threshold if self._grounding is not None else None
                ),
                "entailment_threshold":   cfg.relate.entailment_threshold,
                "contradiction_threshold": cfg.relate.contradiction_threshold,
                "map_theta":              cfg.map.theta,
                "map_reject_theta":       cfg.map.reject_theta,
                "contradiction_similarity_threshold": cfg.contradiction_similarity_threshold,
            }
            models = {
                "voter_models":         self._config_snapshot.get("voter_models"),
                "level2_voter_models":  self._config_snapshot.get("level2_voter_models"),
                "escalation_model":     self._config_snapshot.get("escalation_model"),
                "scorer":               self._config_snapshot.get("scorer"),
            }
            manifest = RunManifest(
                run_id=run_id,
                artifact_root=self._artifact_root,
                timestamp_start=datetime.now(tz=timezone.utc).isoformat(),
                papers=[pmcid],
                schema_version=schema_version,
                prompt_version=prompt_version,
                cascade_signature=cascade_signature,
                config=self._config_snapshot,
                models=models,
                thresholds=thresholds,
                chunk_size=cfg.map.chunk_size,
            )
            from .persistence import _try_git_commit  # noqa: PLC0415
            manifest.git_commit = _try_git_commit()
            # Reuse the single source of truth so the manifest's hash matches
            # the cache invalidation hash (B-007).
            manifest.extra["pipeline_config_hash"] = self._pipeline_config_hash()
            writer = RunArtifactWriter(
                run_id=run_id, root_dir=self._artifact_root, manifest=manifest,
            )
            return writer
        except Exception as exc:
            logger.warning("[%s] artifact writer setup failed: %s", pmcid, exc)
            return None

    # The per-stage artifact persisters live in persistence.py so the batch
    # runner can reuse them. Sync runner just forwards.
    def _persist_map_artifacts(self, writer, pmcid, chunk_summaries, grounding_rejected) -> None:
        persist_map_artifacts(writer, pmcid, chunk_summaries, grounding_rejected)

    def _persist_normalize_artifacts(self, writer, pmcid, normal_findings) -> None:
        persist_normalize_artifacts(writer, pmcid, normal_findings)

    def _persist_group_artifacts(self, writer, pmcid, groups, non_groupable_nfs) -> None:
        persist_group_artifacts(writer, pmcid, groups, non_groupable_nfs)

    def _persist_canonicalize_artifacts(self, writer, pmcid, canonical_rules) -> None:
        persist_canonicalize_artifacts(writer, pmcid, canonical_rules)

    def _persist_relate_artifacts(self, writer, pmcid, relations, raw_pairs, skipped_pairs=None) -> None:
        persist_relate_artifacts(writer, pmcid, relations, raw_pairs, skipped_pairs)

    def _persist_resolve_artifacts(self, writer, pmcid, final_rules, relations) -> None:
        persist_resolve_artifacts(writer, pmcid, final_rules, relations)

    def _persist_voter_outputs(self, db_id: int | None, pmcid: str) -> None:
        """Persist per-voter AuditableSummary records into ``sum_map_voter_outputs``.

        Reads from ``MapStage.voter_output_records()`` (populated during the
        cascade) and inserts one row per (chunk, level, voter_index). Cache
        hits contribute no records — their voter outputs live with whichever
        earlier run populated the MAP cache and were persisted then.

        No-op when ``db_id`` is None (sync run with no DB attached).
        Failures are logged as a warning and do not raise; voter-output
        capture is an observability artifact, not load-bearing.
        """
        if db_id is None:
            return
        records = self.last_map_voter_output_records
        if not records:
            return
        try:
            from database.models import SumMapVoterOutput
            rows = [
                SumMapVoterOutput(
                    pipeline_run_id = db_id,
                    pmcid           = pmcid,
                    chunk_id        = r["chunk_id"],
                    level           = r["level"],
                    voter_index     = r["voter_index"],
                    provider        = r["provider"],
                    model           = r["model"],
                    is_selected     = bool(r["is_selected"]),
                    failed          = bool(r["failed"]),
                    error_message   = r.get("error_message"),
                    finding_count   = int(r["finding_count"]),
                    latency_ms      = r.get("latency_ms"),
                    raw_output      = r.get("raw_output"),
                )
                for r in records
            ]
            with self._db.session_scope() as session:
                session.bulk_save_objects(rows)
            logger.info(
                "[%s] DB: persisted %d per-voter outputs (run_id=%d)",
                pmcid, len(rows), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist voter outputs: %s", pmcid, exc)

    def _persist_map_findings(
        self,
        db_id: int | None,
        pmcid: str,
        chunk_summaries: list,
    ) -> None:
        _persistence_persist_map_findings(self._db, db_id, pmcid, chunk_summaries)

    def _persist_normal_findings(
        self,
        db_id: int | None,
        pmcid: str,
        normal_findings: list,
    ) -> dict[str, int]:
        return _persistence_persist_normal_findings(
            self._db, db_id, pmcid, normal_findings,
        )

    def _persist_finding_groups(
        self,
        db_id: int | None,
        pmcid: str,
        finding_groups: list,
        nf_db_id_map: dict[str, int],
    ) -> dict[str, int]:
        return _persistence_persist_finding_groups(
            self._db, db_id, pmcid, finding_groups, nf_db_id_map,
        )

    def _persist_canonical_rules(
        self,
        db_id: int | None,
        pmcid: str,
        canonical_rules: list,
        fg_db_id_map: dict[str, int],
    ) -> dict[str, int]:
        return _persistence_persist_canonical_rules(
            self._db, db_id, pmcid, canonical_rules, fg_db_id_map,
        )

    def _corpus_relate_incremental(
        self,
        pmcid: str,
        canonical_rules: list,
    ) -> None:
        """
        Run incremental cross-paper corpus relate for a newly processed paper.

        No-ops when DB is unavailable or when fewer than 2 PMCIDs exist in DB.
        Errors are logged as warnings and never propagate — corpus relate is
        analytical and must not block per-paper pipeline completion.
        """
        if self._db is None:
            return
        try:
            from .helpers.corpus_relate import CorpusRelateStage  # noqa: PLC0415
            stage = CorpusRelateStage(
                entailment_threshold=self._relate._entailment_threshold,
                contradiction_threshold=self._relate._contradiction_threshold,
            )
            relations = stage.relate_incremental(pmcid, canonical_rules, self._db)
            logger.info(
                "[%s] CORPUS RELATE incremental: %d cross-paper relations",
                pmcid, len(relations),
            )
        except Exception as exc:
            logger.warning(
                "[%s] CORPUS RELATE incremental failed (non-fatal): %s", pmcid, exc,
                exc_info=True,
            )

    def _persist_relations(
        self,
        db_id: int | None,
        pmcid: str,
        relations: list,
        cr_db_id_map: dict[str, int],
    ) -> None:
        _persistence_persist_relations(
            self._db, db_id, pmcid, relations, cr_db_id_map,
        )

    def _persist_rejection_summary(
        self,
        db_id: int | None,
        rejection_summary: "RejectionSummary",
    ) -> None:
        _persistence_persist_rejection_summary(self._db, db_id, rejection_summary)

    def _persist_final_rules(
        self,
        db_id: int | None,
        pmcid: str,
        final_rules: list,
        cr_db_id_map: dict[str, int],
    ) -> None:
        _persistence_persist_final_rules(
            self._db, db_id, pmcid, final_rules, cr_db_id_map,
        )

    def _cost_output_dir(
        self, writer: RunArtifactWriter | None, run_id: str,
    ) -> Path:
        """Where to write cost_report.* and llm_usage_records.jsonl.

        Precedence: explicit cfg override → run-artifact dir → output_dir/cost/{run_id}.
        """
        override = self._cfg.cost.cost_report_output_dir
        if override:
            return Path(override)
        if writer is not None:
            return writer.run_dir / "cost"
        return self._output_dir / "cost" / run_id

    def _write_cost_artifacts(
        self,
        usage_collector: UsageCollector | None,
        writer: RunArtifactWriter | None,
        run_id: str,
        pmcid: str,
    ) -> None:
        """Persist usage JSONL and cost report. Never raises."""
        if usage_collector is None or len(usage_collector) == 0:
            return
        try:
            out_dir = self._cost_output_dir(writer, run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self._cfg.cost.write_usage_jsonl:
                usage_collector.write_jsonl(out_dir / "llm_usage_records.jsonl")
            book = PriceBook.load(self._cfg.cost.model_prices_path)
            write_cost_reports(usage_collector.records(), book, out_dir)
            logger.info("[%s] cost report written to %s", pmcid, out_dir)
        except Exception as exc:
            logger.warning("[%s] cost report write failed (non-fatal): %s", pmcid, exc)

    def _result_path(self, pmcid: str) -> Path:
        return self._summaries_dir / f"{pmcid}.json"

    def _pipeline_config_hash(self) -> str:
        """Hash of all knobs that should invalidate the cached result.

        Mirrors ``BatchKnowledgeExtractionRunner._pipeline_config_hash`` so the two
        runners agree on what counts as a config change (cascade composition,
        thresholds, schema/prompt versions). Any drift here re-opens B-007.
        """
        from .models import (  # noqa: PLC0415
            CANONICALIZE_DIRECTION_POLICY_VERSION,
            MAP_AGREEMENT_POLICY_VERSION,
            MAP_PROMPT_VERSION,
            MAP_SCHEMA_VERSION,
        )
        from .persistence import compute_pipeline_config_hash  # noqa: PLC0415
        cfg = self._cfg
        map_meta = self._map.run_metadata_summary() or {}
        cascade_signature = (
            map_meta.get("cascade_signature")
            or self._config_snapshot.get("cascade_signature")
        )
        schema_version = map_meta.get("schema_version") or MAP_SCHEMA_VERSION
        prompt_version = map_meta.get("prompt_version") or MAP_PROMPT_VERSION
        thresholds = {
            "grounding_threshold": (
                self._grounding.threshold if self._grounding is not None else None
            ),
            "entailment_threshold":    cfg.relate.entailment_threshold,
            "contradiction_threshold": cfg.relate.contradiction_threshold,
            "map_theta":               cfg.map.theta,
            "map_reject_theta":        cfg.map.reject_theta,
            "contradiction_similarity_threshold": cfg.contradiction_similarity_threshold,
            "enable_router":           self._enable_router,
            "router_single_voter_policy": (
                self._router_single_voter_policy if self._enable_router else None
            ),
            "legacy_single_voter_policy": (
                self._legacy_single_voter_policy if not self._enable_router else None
            ),
            # B-049 behavioural stamp — bumps invalidate cached summaries when
            # canonicalization semantics change.
            "canonicalize_direction_policy_version": CANONICALIZE_DIRECTION_POLICY_VERSION,
            # B-051 behavioural stamp — bumps invalidate cached summaries when
            # the MAP agreement-gate hard-fail policy changes (e.g. polarity
            # set widened). Companion to MAP_SCHEMA_VERSION which invalidates
            # the chunk-level PipelineCache; both must move together.
            "map_agreement_policy_version": MAP_AGREEMENT_POLICY_VERSION,
            # B-080 — citation filter changes the post-MAP finding set, so it
            # must invalidate cached summaries. Kept in sync with the batch
            # runner's hash.
            "citation_enabled":        cfg.citation.enabled,
            "citation_check_verbatim": cfg.citation.check_verbatim if cfg.citation.enabled else None,
            "citation_fabricated_threshold": (
                cfg.citation.fabricated_threshold
                if cfg.citation.enabled and cfg.citation.check_verbatim else None
            ),
        }
        models = {
            "voter_models":        self._config_snapshot.get("voter_models"),
            "level2_voter_models": self._config_snapshot.get("level2_voter_models"),
            "escalation_model":    self._config_snapshot.get("escalation_model"),
            "scorer":              self._config_snapshot.get("scorer"),
        }
        return compute_pipeline_config_hash(
            config=self._config_snapshot,
            thresholds=thresholds,
            models=models,
            schema_version=schema_version,
            prompt_version=prompt_version,
            cascade_signature=cascade_signature,
        )

    def _load_result(self, pmcid: str) -> dict | None:
        p = self._result_path(pmcid)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[%s] cached result unreadable, ignoring: %s", pmcid, exc)
            return None
        stored_hash = data.get("pipeline_config_hash")
        try:
            current_hash = self._pipeline_config_hash()
        except Exception as exc:
            logger.warning(
                "[%s] could not compute current pipeline_config_hash (%s) — "
                "ignoring cache for safety", pmcid, exc,
            )
            return None
        if stored_hash != current_hash:
            logger.info(
                "[%s] cached result stale (config hash %s != %s) — re-running",
                pmcid, stored_hash, current_hash,
            )
            return None
        # Tag the returned dict so process_batch can distinguish cache hits
        # ("skipped") from fresh runs ("success").  The on-disk JSON keeps
        # whatever status was stamped at write time — this is an in-memory
        # marker on the caller's copy only.
        data["status"] = "skipped"
        return data

    def _save_result(self, result: dict) -> None:
        try:
            result.setdefault("pipeline_config_hash", self._pipeline_config_hash())
        except Exception as exc:
            logger.warning(
                "[%s] could not stamp pipeline_config_hash on result (%s) — "
                "writing without it", result.get("pmcid"), exc,
            )
        p = self._result_path(result["pmcid"])
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def _model_name(llm) -> str:
    """Best-effort extraction of model name for config snapshots."""
    for attr in ("model_name", "model", "model_id"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    return type(llm).__name__


