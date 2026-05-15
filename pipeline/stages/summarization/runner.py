"""
SummarizationRunner — orchestrates MAP → GROUNDING → NORMALIZE → GROUP
→ CANONICALIZE → RELATE → RESOLVE with ABC cascading.
REDUCE → RULES is an optional secondary block (disabled by default).

Usage example
-------------
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pipeline.stages.summarization import SummarizationRunner

from dataclasses import replace
from pipeline.stages.summarization.config import SummarizationConfig, MapConfig

cfg = SummarizationConfig(map=MapConfig(theta=0.65))   # override just what you need

runner = SummarizationRunner(
    voter_llms=[                                                   # Level 1: cheapest
        AzureChatOpenAI(model="DeepSeek-V3.2-Speciale",          temperature=0.1),
        VertexAI(model="gemini-2.5-flash-lite-preview-06-17",     temperature=0.1),
        AzureChatOpenAI(model="mistral-large-3",                  temperature=0.1),
    ],
    level2_voter_llms=[                                            # Level 2: mid-tier
        VertexAI(model="gemini-2.5-flash",                        temperature=0.1),
        AzureChatOpenAI(model="kimi-k2.5",                        temperature=0.1),
        ChatAnthropic(model="claude-haiku-4-5-20251001",          temperature=0.1),
    ],
    escalation_llm=ChatAnthropic(model="claude-sonnet-4-6", temperature=0),  # Level 3
    config=cfg,
    output_dir=Path("out/summaries"),
    trace_enabled=True,   # ← enable structured JSONL traces
)

# Single paper
file_data = SummarizationRunner.load_paper_from_db("PMC10047158")
result = runner.process(file_data)

# Batch
pmcids = ["PMC10047158", "PMC10047213", "PMC10047408"]
results = runner.process_batch([SummarizationRunner.load_paper_from_db(p) for p in pmcids])

# After a batch, export CSV summaries from the JSONL traces:
from pipeline.stages.summarization.observability import export_all_csv
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
from .config import SummarizationConfig
from .current_stages.canonicalize_stage import CanonicalizeStage
from .helpers.contradiction_detector import ContradictionDetector
from .helpers.grounding_filter import GroundingFilter
from .current_stages.group_stage import GroupStage, is_groupable
from .current_stages.map_stage import MapStage
from .persistence import (
    RunArtifactWriter,
    RunManifest,
    non_groupable_reason as _non_groupable_reason_helper,
    persist_canonicalize_artifacts,
    persist_group_artifacts,
    persist_map_artifacts,
    persist_normalize_artifacts,
    persist_relate_artifacts,
    persist_resolve_artifacts,
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
    RejectedFinding,
    RejectionSummary,
    RelationTypeEnum,
    compute_finding_id,
)
from .current_stages.normalize_stage import NormalizeStage
from .costing import PriceBook, UsageCollector, write_cost_reports
from .observability import TraceCollector, flush_collector
from .old_stages.reduce_stage import ReduceStage
from .current_stages.relate_stage import RelateStage
from .current_stages.resolve_stage import ResolveStage
from .old_stages.rule_stage import RuleStage
from pipeline.stages.summarization.interfaces import (
    ContradictionChecker,
    GroundingChecker,
    MapOutputScorer,
)
from pipeline.utils.memory_logging import MemoryLogger

logger = logging.getLogger(__name__)


class SummarizationRunner:
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
        All numeric/boolean pipeline knobs.  Defaults to SummarizationConfig()
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
        config: SummarizationConfig | None = None,
        scorer: MapOutputScorer | None = None,
        embed_fn=None,
        output_dir: Path = Path("langchain-summarization/summarization_results"),
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
    ) -> None:
        cfg = config or SummarizationConfig()
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_path or (output_dir / "pipeline_cache.json")
        self._cache = PipelineCache(cache_file)

        # Per-paper cascade decision log directory. One JSONL file per pmcid,
        # appended to as L1/L2/L3 decisions are made. Surfaces cascade
        # behaviour for offline analysis without depending on trace_enabled.
        self._cascade_log_dir: Path = output_dir / "cascade_decisions"

        # Default scorer: SemanticAgreementScorer over EmbeddingSimilarityStrategy.
        # Centrality-based best-output selection (Soiffer 2025) — the previous
        # default was EmbeddingScorer, which left best_index unset and forced
        # AgreementChecker.best() to fall back to a (mean_evidence_len, n_findings)
        # heuristic. Pass embed_fn through so we don't silently fall back to
        # OpenAIEmbedder when the caller supplied a different embedder.
        if scorer is None:
            from .agreement import EmbeddingSimilarityStrategy, SemanticAgreementScorer
            scorer = SemanticAgreementScorer(
                strategy=EmbeddingSimilarityStrategy(embed_fn=embed_fn),
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
        )
        self._normalize = NormalizeStage(extra_synonyms=cfg.normalize.extra_synonyms)
        self._group = GroupStage()
        self._canonicalize = CanonicalizeStage()
        self._relate = RelateStage(
            entailment_threshold=cfg.relate.entailment_threshold,
            contradiction_threshold=cfg.relate.contradiction_threshold,
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
        # Snapshot of config for traces (model introspection is best-effort)
        self._config_snapshot = {
            **dataclasses.asdict(cfg),
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
        """
        Insert a pipeline_runs row with status='running'.

        Returns the integer row id on success, None if db=None or on any error.
        Persistence failures are logged as warnings and never crash the pipeline.
        """
        if self._db is None:
            return None
        try:
            from database import Document, PipelineRun  # noqa: PLC0415 — lazy import
            with self._db.session_scope() as session:
                doc = session.query(Document).filter_by(pmcid=pmcid).first()
                run = PipelineRun(
                    run_id=run_id,
                    pmcid=pmcid,
                    document_id=doc.id if doc else None,
                    status="running",
                    config_snapshot=self._config_snapshot,
                    started_at=datetime.now(tz=timezone.utc),
                )
                session.add(run)
                session.flush()
                return run.id
        except Exception as exc:
            logger.warning("[%s] DB: failed to create pipeline_run: %s", pmcid, exc)
            return None

    def _finish_pipeline_run(
        self,
        db_id: int | None,
        status: str,
        error: str | None = None,
        narrative_summary: str | None = None,
    ) -> None:
        """
        Update a pipeline_runs row to its final status.

        No-op when db_id is None (db=None or creation failed).
        Persistence failures are logged as warnings and never crash the pipeline.
        """
        if db_id is None:
            return
        try:
            from database import PipelineRun  # noqa: PLC0415 — lazy import
            with self._db.session_scope() as session:
                run = session.query(PipelineRun).filter_by(id=db_id).first()
                if run is not None:
                    run.status = status
                    run.finished_at = datetime.now(tz=timezone.utc)
                    run.error = error
                    if narrative_summary is not None:
                        run.narrative_summary = narrative_summary
        except Exception as exc:
            logger.warning("DB: failed to update pipeline_run id=%s: %s", db_id, exc)

    # ── Phase 2: DB persistence ────────────────────────────────────────────────

    def _clear_normalized_run_data(self, db_id: int, pmcid: str) -> None:
        """
        Delete all post-MAP rows for (pipeline_run_id, pmcid) in FK-safe order
        (children before parents).  Called before re-inserting NORMALIZE onwards
        so that re-runs don't hit unique-constraint or FK violations.

        MAP findings are handled separately in _persist_map_findings (which also
        does delete-before-insert), so they are not touched here.
        """
        try:
            from database.models import (
                SumFinalRule, SumRelation, SumCanonicalRule,
                SumGroupMember, SumFindingGroup,
                SumNormalFindingSpan, SumNormalFinding,
            )
            with self._db.engine.begin() as conn:
                t = SumFinalRule.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                t = SumRelation.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                t = SumCanonicalRule.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                # SumGroupMember has no pmcid column — join via finding_group_id
                fg_table = SumFindingGroup.__table__
                subq = fg_table.select().where(
                    (fg_table.c.pipeline_run_id == db_id) & (fg_table.c.pmcid == pmcid)
                ).with_only_columns(fg_table.c.id)
                t = SumGroupMember.__table__
                conn.execute(t.delete().where(t.c.finding_group_id.in_(subq)))
                t = SumFindingGroup.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                # SumNormalFindingSpan has no pmcid — join via normal_finding_id
                nf_table = SumNormalFinding.__table__
                subq2 = nf_table.select().where(
                    (nf_table.c.pipeline_run_id == db_id) & (nf_table.c.pmcid == pmcid)
                ).with_only_columns(nf_table.c.id)
                t = SumNormalFindingSpan.__table__
                conn.execute(t.delete().where(t.c.normal_finding_id.in_(subq2)))
                t = SumNormalFinding.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
        except Exception as exc:
            logger.warning(
                "[%s] DB: failed to clear normalized run data (run_id=%d): %s",
                pmcid, db_id, exc, exc_info=True,
            )

    def _replace_verbatim_from_db(self, chunk_summaries: list) -> None:
        """
        Replace verbatim_support on each Finding with the actual TextElement
        text_content from the database.

        Evidence strings have format "S{i}|{pmcid}|{te_id}".  We collect all
        cited te_ids, batch-query TextElement.text_content in one round-trip,
        and overwrite verbatim_support in-place.  Falls back to the LLM-supplied
        value when the DB is unavailable, the te_id has no match, or any error
        occurs.  No-ops when self._db is None.
        """
        if self._db is None:
            return

        te_ids: set[int] = set()
        for cs in chunk_summaries:
            for f in cs.findings:
                for ev in f.evidence:
                    parts = ev.split("|")
                    if len(parts) == 3:
                        try:
                            te_ids.add(int(parts[2]))
                        except ValueError:
                            pass

        if not te_ids:
            return

        from database import TextElement  # type: ignore  # optional dep

        te_map: dict[int, str] = {}
        try:
            with self._db.session_scope() as session:
                rows = (
                    session.query(TextElement.id, TextElement.text_content)
                    .filter(TextElement.id.in_(te_ids))
                    .all()
                )
                te_map = {row.id: row.text_content for row in rows}
        except Exception as exc:
            logger.warning("[verbatim] DB lookup failed — keeping LLM verbatim: %s", exc)
            return

        replaced = 0
        for cs in chunk_summaries:
            for f in cs.findings:
                if not f.evidence:
                    continue
                parts = f.evidence[0].split("|")
                if len(parts) == 3:
                    try:
                        te_id = int(parts[2])
                        text = te_map.get(te_id)
                        if text:
                            f.verbatim_support = text
                            replaced += 1
                    except ValueError:
                        pass

        logger.info("[verbatim] replaced %d/%d finding verbatims from DB",
                    replaced, sum(len(cs.findings) for cs in chunk_summaries))

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

    def _persist_map_findings(
        self,
        db_id: int | None,
        pmcid: str,
        chunk_summaries: list,
    ) -> None:
        """
        Persist scored MAP findings (post-grounding) to sum_map_findings.
        Entities are pre-normalization.  No-op when db_id is None.
        """
        if db_id is None:
            return
        try:
            from database.models import SumMapFinding
            from .models import FindingScope
            rows = []
            for cs in chunk_summaries:
                for pos, f in enumerate(cs.findings):
                    scope = f.scope or FindingScope()
                    rows.append({
                        "pipeline_run_id":       db_id,
                        "pmcid":                 pmcid,
                        "chunk_id":              cs.chunk_id,
                        "position_in_chunk":     pos,
                        "category":              f.category,
                        "claim":                 f.claim,
                        "confidence":            f.confidence,
                        "verbatim_support":      f.verbatim_support,
                        "subject_entity":        f.subject_entity,
                        "outcome_entity":        f.outcome_entity,
                        "relation_type":         f.relation_type.value,
                        "direction":             f.direction.value if f.direction else None,
                        "raw_relation_type":     f.raw_relation_type,
                        "raw_direction":         f.raw_direction,
                        "raw_category":          f.raw_category,
                        "grounding_score":       f.grounding_score,
                        "evidence_refs":         list(f.evidence) if f.evidence else [],
                        "scope_disease_subtype":   scope.disease_subtype,
                        "scope_cohort_n":          scope.cohort_n,
                        "scope_assay_method":      scope.assay_method,
                        "scope_biomarker_cutoff":  scope.biomarker_cutoff,
                        "scope_tissue_site":       scope.tissue_site,
                        "scope_treatment_context": scope.treatment_context,
                        "scope_endpoint":          scope.endpoint,
                        "scope_study_design":      scope.study_design,
                    })
            if not rows:
                return
            with self._db.engine.begin() as conn:
                conn.execute(
                    SumMapFinding.__table__.delete().where(
                        SumMapFinding.__table__.c.pipeline_run_id == db_id
                    )
                )
                conn.execute(SumMapFinding.__table__.insert(), rows)
            logger.info(
                "[%s] DB: persisted %d map findings (run_id=%d)",
                pmcid, len(rows), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist map findings: %s", pmcid, exc, exc_info=True)

    def _persist_normal_findings(
        self,
        db_id: int | None,
        pmcid: str,
        normal_findings: list,
    ) -> dict[str, int]:
        """
        Persist NormalFindings + evidence spans to sum_normal_findings /
        sum_normal_finding_spans.  Returns {normal_id: db_row_id} for later
        phases.  No-op (returns {}) when db_id is None.
        """
        if db_id is None:
            return {}
        self._clear_normalized_run_data(db_id, pmcid)
        nf_id_map: dict[str, int] = {}
        try:
            from database.models import SumNormalFinding, SumNormalFindingSpan
            from .models import FindingScope
            with self._db.session_scope() as session:
                for nf in normal_findings:
                    scope = nf.scope or FindingScope()
                    row = SumNormalFinding(
                        pipeline_run_id     = db_id,
                        pmcid               = pmcid,
                        normal_id           = nf.normal_id,
                        subject_entity      = nf.subject_entity,
                        outcome_entity      = nf.outcome_entity,
                        relation_type       = nf.relation_type.value,
                        direction           = nf.direction.value if nf.direction else None,
                        category            = nf.category,
                        predicate_text      = nf.predicate_text,
                        mean_grounding_score= nf.mean_grounding_score,
                        pmcids              = list(nf.pmcids) if nf.pmcids else [],
                        scope_disease_subtype   = scope.disease_subtype,
                        scope_cohort_n          = scope.cohort_n,
                        scope_assay_method      = scope.assay_method,
                        scope_biomarker_cutoff  = scope.biomarker_cutoff,
                        scope_tissue_site       = scope.tissue_site,
                        scope_treatment_context = scope.treatment_context,
                        scope_endpoint          = scope.endpoint,
                        scope_study_design      = scope.study_design,
                    )
                    session.add(row)
                    session.flush()           # get row.id before adding spans
                    nf_id_map[nf.normal_id] = row.id
                    for span in (nf.evidence or []):
                        session.add(SumNormalFindingSpan(
                            normal_finding_id = row.id,
                            sentence_id       = span.sentence_id,
                            pmcid             = span.pmcid,
                            text_element_id   = span.text_element_id or None,
                            verbatim          = span.verbatim,
                        ))
            logger.info(
                "[%s] DB: persisted %d normal findings + spans (run_id=%d)",
                pmcid, len(nf_id_map), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist normal findings: %s", pmcid, exc)
            return {}  # partial map is invalid after rollback — don't pass stale IDs downstream
        return nf_id_map

    def _persist_finding_groups(
        self,
        db_id: int | None,
        pmcid: str,
        finding_groups: list,
        nf_db_id_map: dict[str, int],
    ) -> dict[str, int]:
        """
        Persist FindingGroups + member links to sum_finding_groups /
        sum_group_members.

        nf_db_id_map maps normal_id -> DB integer id from _persist_normal_findings.
        Member rows are still created when nf_db_id_map is empty — normal_id
        is stored as a string backup; normal_finding_id FK will be NULL.

        Returns {group_id: db_row_id} for Phase 4 (CANONICALIZE) persistence.
        No-op (returns {}) when db_id is None.
        """
        if db_id is None:
            return {}
        group_id_map: dict[str, int] = {}
        try:
            from database.models import SumFindingGroup, SumGroupMember
            with self._db.session_scope() as session:
                for fg in finding_groups:
                    row = SumFindingGroup(
                        pipeline_run_id     = db_id,
                        pmcid               = pmcid,
                        group_id            = fg.group_id,
                        subject_entity      = fg.subject_entity,
                        outcome_entity      = fg.outcome_entity,
                        relation_type       = fg.relation_type.value,
                        category            = fg.category,
                        scope_heterogeneity = fg.scope_heterogeneity,
                        direction_counts    = dict(fg.direction_counts),
                    )
                    session.add(row)
                    session.flush()          # get row.id before adding members
                    group_id_map[fg.group_id] = row.id
                    missing = 0
                    for normal_id in fg.member_ids:
                        nf_db_id = nf_db_id_map.get(normal_id)
                        if nf_db_id is None:
                            missing += 1
                        session.add(SumGroupMember(
                            finding_group_id  = row.id,
                            normal_finding_id = nf_db_id,   # None when Phase 2 skipped
                            normal_id         = normal_id,
                        ))
                    if missing:
                        logger.debug(
                            "[%s] group %s: %d member(s) have no NF DB id (Phase 2 incomplete)",
                            pmcid, fg.group_id, missing,
                        )
            logger.info(
                "[%s] DB: persisted %d finding groups (run_id=%d)",
                pmcid, len(group_id_map), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist finding groups: %s", pmcid, exc)
            return {}
        return group_id_map

    def _persist_canonical_rules(
        self,
        db_id: int | None,
        pmcid: str,
        canonical_rules: list,
        fg_db_id_map: dict[str, int],
    ) -> dict[str, int]:
        """
        Persist CanonicalRules to sum_canonical_rules.
        Returns {canonical_id: db_row_id} for Phase 5/6 persistence.
        """
        if db_id is None:
            return {}
        cr_id_map: dict[str, int] = {}
        try:
            from database.models import SumCanonicalRule
            rows = []
            for cr in canonical_rules:
                fg_db_id = fg_db_id_map.get(cr.group_id)
                row = SumCanonicalRule(
                    pipeline_run_id      = db_id,
                    pmcid                = pmcid,
                    canonical_id         = cr.canonical_id,
                    finding_group_id     = fg_db_id,
                    group_id             = cr.group_id,
                    subject_entity       = cr.subject_entity,
                    outcome_entity       = cr.outcome_entity,
                    relation_type        = cr.relation_type.value,
                    direction            = cr.direction.value if cr.direction else None,
                    predicate_text       = cr.predicate_text,
                    is_conflicted        = cr.is_conflicted,
                    study_coverage       = cr.study_coverage,
                    category             = cr.category,
                    pmcids               = list(cr.supporting_pmcids) if cr.supporting_pmcids else [],
                    member_normal_ids    = list(cr.member_normal_ids) if cr.member_normal_ids else [],
                    mean_grounding_score = cr.mean_grounding_score,
                    finding_count        = cr.finding_count,
                    subject_cui          = cr.subject_cui,
                    outcome_cui          = cr.outcome_cui,
                )
                rows.append(row)
            
            with self._db.session_scope() as session:
                for row in rows:
                    session.add(row)
                session.flush()
                for row in rows:
                    cr_id_map[row.canonical_id] = row.id
            
            logger.info(
                "[%s] DB: persisted %d canonical rules (run_id=%d)",
                pmcid, len(cr_id_map), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist canonical rules: %s", pmcid, exc)
            return {}
        return cr_id_map

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
        """Persist Relations to sum_relations."""
        if db_id is None:
            return
        try:
            from database.models import SumRelation
            rows = []
            for rel in relations:
                rows.append(SumRelation(
                    pipeline_run_id     = db_id,
                    pmcid               = pmcid,
                    rule_id_a           = rel.rule_id_a,
                    rule_id_b           = rel.rule_id_b,
                    canonical_rule_a_id = cr_db_id_map.get(rel.rule_id_a),
                    canonical_rule_b_id = cr_db_id_map.get(rel.rule_id_b),
                    relation_type       = rel.relation_type.value,
                    nli_score_a_to_b    = rel.nli_score_a_to_b,
                    nli_score_b_to_a    = rel.nli_score_b_to_a,
                ))
            if rows:
                with self._db.session_scope() as session:
                    session.bulk_save_objects(rows)
                logger.info(
                    "[%s] DB: persisted %d relations (run_id=%d)",
                    pmcid, len(rows), db_id,
                )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist relations: %s", pmcid, exc)

    def _persist_rejection_summary(
        self,
        db_id: int | None,
        rejection_summary: "RejectionSummary",
    ) -> None:
        """
        Persist a RejectionSummary + its detail rows to sum_rejection_summaries /
        sum_rejected_findings.  No-op when db_id is None.
        """
        if db_id is None:
            return
        try:
            from database.models import SumRejectionSummary, SumRejectedFinding  # noqa: PLC0415
            with self._db.session_scope() as session:
                summary_row = SumRejectionSummary(
                    pipeline_run_id                = db_id,
                    pmcid                          = rejection_summary.pmcid,
                    grounding_threshold            = rejection_summary.grounding_threshold,
                    map_findings_total             = rejection_summary.map_findings_total,
                    map_grounding_rejected         = rejection_summary.map_grounding_rejected,
                    normal_findings_total          = rejection_summary.normal_findings_total,
                    non_groupable_total            = rejection_summary.non_groupable_total,
                    non_groupable_no_subject       = rejection_summary.non_groupable_no_subject,
                    non_groupable_no_outcome       = rejection_summary.non_groupable_no_outcome,
                    non_groupable_unclear_relation = rejection_summary.non_groupable_unclear_relation,
                    grounding_rejection_rate       = rejection_summary.grounding_rejection_rate,
                    non_groupable_rate             = rejection_summary.non_groupable_rate,
                    grounding_rejected_by_category = rejection_summary.grounding_rejected_by_category,
                )
                session.add(summary_row)
                session.flush()   # get summary_row.id before inserting detail rows

                for item in rejection_summary.rejected:
                    session.add(SumRejectedFinding(
                        pipeline_run_id      = db_id,
                        rejection_summary_id = summary_row.id,
                        pmcid                = rejection_summary.pmcid,
                        stage                = item.stage,
                        reason               = item.reason,
                        claim                = item.claim,
                        category             = item.category,
                        chunk_id             = item.chunk_id,
                        grounding_score      = item.grounding_score,
                        subject_entity       = item.subject_entity,
                        outcome_entity       = item.outcome_entity,
                        relation_type        = item.relation_type,
                    ))

            logger.info(
                "[%s] DB: persisted rejection summary (%d grounding, %d non-groupable, run_id=%d)",
                rejection_summary.pmcid,
                rejection_summary.map_grounding_rejected,
                rejection_summary.non_groupable_total,
                db_id,
            )
        except Exception as exc:
            logger.warning(
                "[%s] DB: failed to persist rejection summary: %s",
                rejection_summary.pmcid, exc, exc_info=True,
            )

    def _persist_final_rules(
        self,
        db_id: int | None,
        pmcid: str,
        final_rules: list,
        cr_db_id_map: dict[str, int],
    ) -> None:
        """Persist FinalRules (scored canonical rules) to sum_final_rules."""
        if db_id is None:
            return
        try:
            from database.models import SumFinalRule
            rows = []
            for fr in final_rules:
                rows.append(SumFinalRule(
                    pipeline_run_id      = db_id,
                    pmcid                = pmcid,
                    final_id             = fr.final_id,
                    canonical_rule_id    = cr_db_id_map.get(fr.canonical_id),
                    canonical_id         = fr.canonical_id,
                    subject_entity       = fr.subject_entity,
                    outcome_entity       = fr.outcome_entity,
                    relation_type        = fr.relation_type.value,
                    direction            = fr.direction.value if fr.direction else None,
                    predicate_text       = fr.predicate_text,
                    category             = fr.category,
                    final_score          = fr.final_score,
                    support_count        = fr.support_count,
                    contradict_count     = fr.contradict_count,
                    scope_qualify_count  = fr.scope_qualify_count,
                    is_contradicted      = fr.is_contradicted,
                    contradicted_by      = list(fr.contradicted_by) if fr.contradicted_by else [],
                ))
            if rows:
                with self._db.session_scope() as session:
                    session.bulk_save_objects(rows)
                logger.info(
                    "[%s] DB: persisted %d final rules (run_id=%d)",
                    pmcid, len(rows), db_id,
                )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist final rules: %s", pmcid, exc)

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

        Mirrors ``BatchSummarizationRunner._pipeline_config_hash`` so the two
        runners agree on what counts as a config change (cascade composition,
        thresholds, schema/prompt versions). Any drift here re-opens B-007.
        """
        from .models import MAP_SCHEMA_VERSION, MAP_PROMPT_VERSION  # noqa: PLC0415
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


def _non_groupable_reason(nf: NormalFinding) -> str:
    """Human-readable reason why a NormalFinding failed is_groupable()."""
    parts = []
    if nf.subject_entity is None:
        parts.append("subject_entity=None")
    if nf.outcome_entity is None:
        parts.append("outcome_entity=None")
    if nf.relation_type is RelationTypeEnum.unclear:
        parts.append("relation_type=unclear")
    return ", ".join(parts) if parts else "unknown"


def _build_rejection_summary(
    pmcid: str,
    grounding_threshold: float | None,
    map_findings_total: int,
    grounding_rejected: list[tuple[str, Finding]],
    normal_findings: list[NormalFinding],
    non_groupable_nfs: list[NormalFinding],
) -> RejectionSummary:
    """Build a RejectionSummary from the collected rejection data for one paper."""
    from collections import Counter  # noqa: PLC0415 — local import to avoid top-level cost

    rejected_items: list[RejectedFinding] = []

    # Grounding rejections
    for chunk_id, f in grounding_rejected:
        score = f.grounding_score
        reason = (
            f"grounding_score={score:.3f} < threshold={grounding_threshold}"
            if score is not None and grounding_threshold is not None
            else "grounding_score below threshold"
        )
        rejected_items.append(RejectedFinding(
            stage="grounding_map",
            reason=reason,
            claim=f.claim,
            category=f.category,
            chunk_id=chunk_id,
            grounding_score=score,
            subject_entity=f.subject_entity,
            outcome_entity=f.outcome_entity,
            relation_type=f.relation_type.value if f.relation_type else None,
        ))

    # Non-groupable exclusions
    for nf in non_groupable_nfs:
        rejected_items.append(RejectedFinding(
            stage="group_non_groupable",
            reason=_non_groupable_reason(nf),
            claim=nf.predicate_text,
            category=nf.category,
            grounding_score=nf.mean_grounding_score,
            subject_entity=nf.subject_entity,
            outcome_entity=nf.outcome_entity,
            relation_type=nf.relation_type.value if nf.relation_type else None,
        ))

    n_total = map_findings_total
    n_grounding = len(grounding_rejected)
    n_normal = len(normal_findings)
    n_non_groupable = len(non_groupable_nfs)

    cat_counts: Counter[str] = Counter(
        f.category for _, f in grounding_rejected
    )

    return RejectionSummary(
        pmcid=pmcid,
        grounding_threshold=grounding_threshold,
        map_findings_total=n_total,
        map_grounding_rejected=n_grounding,
        normal_findings_total=n_normal,
        non_groupable_total=n_non_groupable,
        non_groupable_no_subject=sum(
            1 for nf in non_groupable_nfs if nf.subject_entity is None
        ),
        non_groupable_no_outcome=sum(
            1 for nf in non_groupable_nfs if nf.outcome_entity is None
        ),
        non_groupable_unclear_relation=sum(
            1 for nf in non_groupable_nfs
            if nf.relation_type is RelationTypeEnum.unclear
        ),
        grounding_rejection_rate=round(n_grounding / n_total, 4) if n_total else 0.0,
        non_groupable_rate=round(n_non_groupable / n_normal, 4) if n_normal else 0.0,
        grounding_rejected_by_category=dict(cat_counts),
        rejected=rejected_items,
    )
