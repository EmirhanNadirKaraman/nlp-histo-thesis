"""
BatchSummarizationRunner — asynchronous batch variant of SummarizationRunner.

Workflow
--------
Each call to ``advance()`` checks whether the current batch phase is done
and, if so, runs agreement scoring and submits the next phase.
The handle is persisted to disk between calls so the process can exit freely.

    runner = BatchSummarizationRunner(l1_voters=[...], ...)

    handle = runner.submit(file_data)       # L1 jobs submitted; handle saved
    handle = runner.advance(handle)         # call again until COMPLETE
    result = runner.finalize(handle)        # REDUCE + RULES (sync)

Typical script usage::

    handle = runner.load_or_submit(file_data)
    while handle.phase != BatchPhase.COMPLETE:
        handle = runner.advance(handle)
        if handle.phase != BatchPhase.COMPLETE:
            print("Jobs still running — try again later.")
            break
    else:
        result = runner.finalize(handle)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from datetime import datetime, timezone

from ..agreement import (
    AgreementChecker,
    CascadeDecisionLog,
    EmbeddingSimilarityStrategy,
    SemanticAgreementScorer,
    evaluate_chunk,
    make_decision_record,
    producer_from_outcome,
)
from ..routing import MapOutputRouter
from ..config import SummarizationConfig
from ..current_stages.canonicalize_stage import CanonicalizeStage
from ..current_stages.group_stage import GroupStage, is_groupable
from ..current_stages.normalize_stage import NormalizeStage
from ..current_stages.relate_stage import RelateStage
from ..current_stages.resolve_stage import ResolveStage
from ..helpers.contradiction_detector import ContradictionDetector
from ..helpers.grounding_filter import GroundingFilter
from ..interfaces.scoring import ChunkDecision
from ..current_stages.map_stage import _format_sentences
from ..models import (
    AuditableSummary,
    CANONICALIZE_DIRECTION_POLICY_VERSION,
    Finding,
    MAP_AGREEMENT_POLICY_VERSION,
    MAP_PROMPT_VERSION,
    MAP_SCHEMA_VERSION,
    MAP_STAGE_NAME,
    NormalFinding,
    RejectedFinding,
    RejectionSummary,
    RelationTypeEnum,
    compute_cascade_signature,
    compute_finding_id,
)
from ..old_stages.reduce_stage import ReduceStage
from ..old_stages.rule_stage import RuleStage
from ..persistence import (
    RunArtifactWriter,
    RunManifest,
    compute_pipeline_config_hash,
    persist_canonicalize_artifacts,
    persist_group_artifacts,
    persist_map_artifacts,
    persist_normalize_artifacts,
    persist_relate_artifacts,
    persist_resolve_artifacts,
)
from ..costing import PriceBook, UsageCollector, write_cost_reports
from .dispatch import OPENAI_MAP_TOOL, build_providers, build_requests, parse_result, submit_level
from .models import BatchHandle, BatchPhase, BatchResult, ProviderJob, VoterBatchConfig

logger = logging.getLogger(__name__)


class BatchSummarizationRunner:
    """
    Asynchronous batch pipeline runner with 50 % cost discounts.

    Parameters
    ----------
    l1_voters, l2_voters:
        Ordered list of VoterBatchConfig for each level.  Provider must be
        "azure", "claude", or "vertex_gemini".
    l3_model:
        Single VoterBatchConfig for the Level-3 escalation model.
    escalation_llm:
        Synchronous LangChain LLM for REDUCE and RULE EXTRACTION (called once
        per paper at the end, after all MAP batches complete).
    config:
        All numeric/boolean pipeline knobs.  Defaults to SummarizationConfig().
        Only map and grounding sub-configs are used by the batch runner.
    output_dir:
        Directory for final ``{pmcid}.json`` result files.
    handle_dir:
        Directory for ``{pmcid}.batch.json`` state files.
        Defaults to ``output_dir / "batch_handles"``.
    """

    def __init__(
        self,
        l1_voters: list[VoterBatchConfig],
        l2_voters: list[VoterBatchConfig],
        l3_model: VoterBatchConfig,
        escalation_llm,
        config: SummarizationConfig | None = None,
        output_dir: Path = Path("out/summaries"),
        handle_dir: Path | None = None,
        embed_fn=None,
        cascade_profile: str = "custom",
        artifact_root: Path | None = None,
        artifact_run_id: str | None = None,
        run_modern_pipeline: bool = True,
        run_reduce: bool = False,
        db=None,
        force_rerun: bool = False,
        run_ner: bool = False,
        enable_router: bool = False,
        router_single_voter_policy: str = "escalate",
    ) -> None:
        from ..agreement.providers import OpenAIEmbedder
        cfg = config or SummarizationConfig()
        self._l1 = l1_voters
        self._l2 = l2_voters
        self._l3 = l3_model
        self._chunk_size = cfg.map.chunk_size
        self._embed_fn = embed_fn or OpenAIEmbedder()
        # Default scorer mirrors SummarizationRunner: SemanticAgreementScorer
        # (max-consensus + centrality-based best-output selection) over the
        # batched EmbeddingSimilarityStrategy. theta / reject_theta stay on
        # AgreementChecker so the scorer keeps theta=None and defers the
        # decision boundary to the checker.
        self._agreement = AgreementChecker(
            scorer=SemanticAgreementScorer(
                strategy=EmbeddingSimilarityStrategy(embed_fn=self._embed_fn),
            ),
            theta=cfg.map.theta,
            reject_theta=cfg.map.reject_theta,
        )
        # Grounding-first router config. When enabled, _process_level runs
        # schema + provenance validation before the agreement gate, drops
        # unusable voters, and (at L1) escalates straight to L3 — mirroring
        # the sync MapStage router path.
        self._enable_router = enable_router
        self._router_single_voter_policy = router_single_voter_policy
        # ReduceStage/RuleStage call `llm.with_structured_output(...)` in their
        # ctors, so we only construct them when the legacy REDUCE/RULES block
        # is enabled. Otherwise the runner can be built with any (or no) LLM —
        # convenient for tests.
        self._escalation_llm = escalation_llm
        self._reduce: ReduceStage | None = (
            ReduceStage(escalation_llm) if run_reduce else None
        )
        self._rules: RuleStage | None = (
            RuleStage(escalation_llm) if run_reduce else None
        )
        self._grounding = (
            GroundingFilter(cfg.grounding.threshold)
            if cfg.grounding.threshold is not None else None
        )
        # ContradictionDetector is only useful with REDUCE/RULES output, so it's
        # also lazy. Avoids needing a structured-output-capable LLM in tests.
        self._contradiction = (
            ContradictionDetector(
                escalation_llm,
                similarity_threshold=cfg.contradiction_similarity_threshold,
            )
            if (run_reduce and cfg.contradiction_similarity_threshold is not None)
            else None
        )
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)
        self._handle_dir = handle_dir or (output_dir / "batch_handles")
        self._handle_dir.mkdir(parents=True, exist_ok=True)
        # Per-paper cascade decision log directory — mirrors SummarizationRunner.
        self._cascade_log_dir: Path = output_dir / "cascade_decisions"

        # Run-artifact provenance — kept on the runner so submit() and
        # finalize() emit consistent metadata regardless of who calls them.
        self._cascade_profile = cascade_profile
        self._cascade_signature = compute_cascade_signature(
            [(v.provider, v.model) for v in self._l1]
            + [(v.provider, v.model) for v in self._l2]
            + [(self._l3.provider, self._l3.model)]
        )

        # Modern post-MAP chain. Stateless stages — instantiate once.
        # When ``run_modern_pipeline`` is False, finalize() falls back to the
        # legacy REDUCE/RULES-only behaviour.
        self._cfg_full = cfg
        self._run_modern_pipeline = run_modern_pipeline
        self._run_reduce = run_reduce
        self._normalize = NormalizeStage(extra_synonyms=cfg.normalize.extra_synonyms)
        self._group = GroupStage()
        self._canonicalize = CanonicalizeStage()
        self._relate = RelateStage(
            entailment_threshold=cfg.relate.entailment_threshold,
            contradiction_threshold=cfg.relate.contradiction_threshold,
        )
        self._resolve = ResolveStage(cfg.resolve)

        # Filesystem artifact persistence — opt-in (mirror of sync runner).
        self._artifact_root: Path | None = (
            Path(artifact_root) if artifact_root is not None else None
        )
        self._artifact_run_id_override: str | None = artifact_run_id

        # DB persistence + caching + NER (parity with sync runner). Persistence
        # is fully optional and degrades to no-ops when ``db`` is None.
        self._db = db
        self._force_rerun = force_rerun
        self._run_ner = run_ner

        # Snapshot of config + cascade for the manifest, mirrors sync runner shape.
        import dataclasses
        self._config_snapshot = {
            **dataclasses.asdict(cfg),
            "cascade_profile":   cascade_profile,
            "cascade_signature": self._cascade_signature,
            "voter_models":         [v.model for v in self._l1],
            "level2_voter_models":  [v.model for v in self._l2],
            "escalation_model":     self._l3.model,
        }

    # ── Public API ──────────────────────────────────────────────────────────────

    def handle_path(self, pmcid: str) -> Path:
        return self._handle_dir / f"{pmcid}.batch.json"

    def load_or_submit(self, file_data: dict) -> BatchHandle:
        """Load existing handle from disk, or submit a new L1 batch.

        Stale-cache trap: a handle from a previous run may have been written
        with ``cached_result_only=True`` pointing at a result JSON that has
        since been invalidated (config hash mismatch) or deleted. Returning
        such a handle would let ``finalize()`` fall through to an empty-data
        re-run with no MAP findings. Detect that case and re-submit instead.
        """
        path = self.handle_path(file_data["pmcid"])
        if path.exists():
            handle = BatchHandle.load(path)
            if handle.cached_result_only and self._load_result(handle.pmcid) is None:
                logger.info(
                    "[%s] handle pointed at stale/missing cache — re-submitting",
                    handle.pmcid,
                )
                path.unlink()
            else:
                logger.info("[%s] Loaded existing handle (phase=%s)", handle.pmcid, handle.phase.value)
                return handle
        return self.submit(file_data)

    def submit(self, file_data: dict) -> BatchHandle:
        """Chunk the paper, submit L1 batch jobs, and persist the handle.

        Short-circuits when a valid cached result JSON exists for the same
        pipeline_config_hash and ``force_rerun`` is False — returns a COMPLETE
        handle marked ``cached_result_only`` so ``finalize()`` returns the
        cached dict without re-spending L1/L2/L3 batch dollars.
        """
        pmcid = file_data["pmcid"]

        # Cache short-circuit (parity with sync ``process()`` skip-on-cached).
        if not self._force_rerun and self._load_result(pmcid) is not None:
            logger.info("[%s] cached result on disk — skipping batch submit", pmcid)
            handle = BatchHandle(
                pmcid=pmcid,
                phase=BatchPhase.COMPLETE,
                cached_result_only=True,
                schema_version=MAP_SCHEMA_VERSION,
                prompt_version=MAP_PROMPT_VERSION,
                stage_name=MAP_STAGE_NAME,
                cascade_profile=self._cascade_profile,
                cascade_signature=self._cascade_signature,
            )
            handle.save(self.handle_path(pmcid))
            return handle

        sentences = file_data["sentences_with_provenance"]
        chunk_map = self._make_chunk_map(sentences)

        run_id = self._artifact_run_id_override or self._make_run_id(pmcid)
        pipeline_run_db_id = self._create_pipeline_run(pmcid, run_id)

        handle = BatchHandle(
            pmcid=pmcid,
            phase=BatchPhase.L1_SUBMITTED,
            sentences=sentences,
            chunk_map=chunk_map,
            l1_strip=[cfg.strip_thinking for cfg in self._l1],
            l2_strip=[cfg.strip_thinking for cfg in self._l2],
            l3_strip=self._l3.strip_thinking,
            schema_version=MAP_SCHEMA_VERSION,
            prompt_version=MAP_PROMPT_VERSION,
            stage_name=MAP_STAGE_NAME,
            cascade_profile=self._cascade_profile,
            cascade_signature=self._cascade_signature,
            pipeline_run_db_id=pipeline_run_db_id,
        )
        handle.jobs = submit_level(chunk_map, pmcid, self._l1, level="l1")
        handle.save(self.handle_path(pmcid))

        total_reqs = sum(j.request_count for j in handle.jobs)
        logger.info(
            "[%s] L1 batch submitted: %d chunks × %d voters = %d requests across %d job(s)",
            pmcid, len(chunk_map), len(self._l1), total_reqs, len(handle.jobs),
        )
        return handle

    def advance(self, handle: BatchHandle) -> BatchHandle:
        """
        Refresh job statuses.  If all jobs in the current phase are done,
        collect results, run agreement, and submit the next phase (or mark COMPLETE).

        Always persists the updated handle to disk.
        """
        if handle.phase == BatchPhase.COMPLETE:
            return handle

        providers = build_providers({j.provider for j in handle.jobs})

        # Refresh statuses
        for job in handle.jobs:
            p = providers.get(job.provider)
            if p and job.status not in ("completed", "failed"):
                p.check(job)

        pending = [j for j in handle.jobs if j.status not in ("completed", "failed")]
        if pending:
            logger.info(
                "[%s] %s: %d/%d job(s) still running",
                handle.pmcid, handle.phase.value, len(pending), len(handle.jobs),
            )
            handle.save(self.handle_path(handle.pmcid))
            return handle

        # Collect all results from completed jobs
        raw_results: list[BatchResult] = []
        failed_job_summaries: list[dict] = []
        for job in handle.jobs:
            if job.status == "failed":
                # Expand the failure log to include provider/model/request_count
                # so a silent multi-model rejection (B-020 class) is obvious in
                # stdout. Previously this said only "Job <id> failed" with no
                # provider context.
                logger.warning(
                    "[%s] Batch job FAILED — provider=%s model=%s request_count=%d job_id=%s — "
                    "skipping its %d result(s). This usually means a model mismatch (OpenAI), "
                    "schema rejection, or rate-limit/quota error. See provider dashboard for details.",
                    handle.pmcid, job.provider, job.model, job.request_count,
                    job.job_id, job.request_count,
                )
                failed_job_summaries.append({
                    "provider": job.provider, "model": job.model,
                    "request_count": job.request_count, "job_id": job.job_id,
                })
                continue
            p = providers[job.provider]
            raw_results.extend(p.retrieve(job))

        # All jobs failed → cascade cannot proceed. Raise rather than producing
        # an empty result that the runner would treat as "every chunk had zero
        # voters" and silently escalate the entire paper to L3. Better to fail
        # loudly than to bill an L3-only re-run. See B-020.
        if failed_job_summaries and len(failed_job_summaries) == len(handle.jobs):
            raise RuntimeError(
                f"[{handle.pmcid}] phase={handle.phase.value}: ALL "
                f"{len(handle.jobs)} batch jobs failed. Cannot proceed. "
                f"Failed jobs: {failed_job_summaries}"
            )
        if failed_job_summaries:
            logger.warning(
                "[%s] phase=%s — %d of %d batch jobs failed. Cascade will proceed "
                "with reduced voter coverage; expect voter_count<expected warnings "
                "downstream. Failed: %s",
                handle.pmcid, handle.phase.value,
                len(failed_job_summaries), len(handle.jobs),
                failed_job_summaries,
            )

        # Merge in any synthetic results produced by in-run voter dedup at the
        # prior phase. These are voters whose (provider, model, temperature)
        # matched an earlier level's voter for the same chunk — their result
        # was carried over instead of issuing a duplicate API call. token_usage
        # is reported as zero so the L1 (or L2) call isn't double-counted.
        level_being_processed = {
            BatchPhase.L1_SUBMITTED: "l1",
            BatchPhase.L2_SUBMITTED: "l2",
            BatchPhase.L3_SUBMITTED: "l3",
        }.get(handle.phase)
        if level_being_processed is not None:
            synthetic = handle.synthetic_results.get(level_being_processed, [])
            if synthetic:
                logger.info(
                    "[%s] Merging %d dedup-synthesised %s result(s) into raw_results",
                    handle.pmcid, len(synthetic), level_being_processed.upper(),
                )
                for d in synthetic:
                    raw_results.append(BatchResult(
                        custom_id=d["custom_id"],
                        content=d.get("content"),
                        error=d.get("error"),
                        input_tokens=d.get("input_tokens", 0),
                        output_tokens=d.get("output_tokens", 0),
                    ))

        if handle.phase == BatchPhase.L1_SUBMITTED:
            # Router path mirrors sync MapStage: L1 escalates straight to L3
            # (schema/grounding failures aren't fixed by a similar-tier voter
            # set). Legacy path keeps the L1 → L2 → L3 staircase.
            if self._enable_router:
                self._process_level(handle, raw_results, level="l1", strip_flags=handle.l1_strip,
                                    next_voters=[self._l3], next_level="l3",
                                    next_strip=[handle.l3_strip], next_phase=BatchPhase.L3_SUBMITTED,
                                    escalated_attr="l3_chunk_ids")
            else:
                self._process_level(handle, raw_results, level="l1", strip_flags=handle.l1_strip,
                                    next_voters=self._l2, next_level="l2",
                                    next_strip=handle.l2_strip, next_phase=BatchPhase.L2_SUBMITTED,
                                    escalated_attr="l2_chunk_ids")
        elif handle.phase == BatchPhase.L2_SUBMITTED:
            self._process_level(handle, raw_results, level="l2", strip_flags=handle.l2_strip,
                                next_voters=[self._l3], next_level="l3",
                                next_strip=[handle.l3_strip], next_phase=BatchPhase.L3_SUBMITTED,
                                escalated_attr="l3_chunk_ids",
                                chunk_ids_to_process=handle.l2_chunk_ids)
        elif handle.phase == BatchPhase.L3_SUBMITTED:
            self._collect_l3(handle, raw_results)

        handle.save(self.handle_path(handle.pmcid))
        return handle

    def finalize(self, handle: BatchHandle) -> dict:
        """
        Run the modern post-MAP chain (and optionally REDUCE → RULES) on the
        batch MAP output, persist filesystem artifacts when configured, and
        return the assembled result dict.

        Only call when ``handle.phase == BatchPhase.COMPLETE``.
        """
        if handle.phase != BatchPhase.COMPLETE:
            raise RuntimeError(
                f"Cannot finalize: handle phase is {handle.phase.value!r}, not 'complete'."
            )

        pmcid = handle.pmcid

        # ── Cache short-circuit (set by submit() when valid result on disk) ───
        if handle.cached_result_only:
            cached = self._load_result(pmcid)
            if cached is not None:
                logger.info("[%s] returning cached result (no work performed)", pmcid)
                return cached
            logger.warning(
                "[%s] cached_result_only set but cache miss — falling through "
                "to re-run, though no MAP chunks were collected. Result will be empty.",
                pmcid,
            )

        chunk_summaries = [
            AuditableSummary.model_validate(v) for v in handle.finalized.values()
        ]

        # ── Filesystem persistence setup (opt-in) ─────────────────────────────
        run_id = self._artifact_run_id_override or self._make_run_id(pmcid)
        writer = self._make_artifact_writer(run_id, pmcid, handle.cascade_signature)
        pipeline_run_db_id = handle.pipeline_run_db_id

        try:
            # Track total MAP findings before any filtering — used by
            # rejection_summary later.
            map_findings_total = sum(len(cs.findings) for cs in chunk_summaries)

            # Assign stable finding_id BEFORE grounding so downstream stages
            # (and rejected-finding artifacts) share the same lineage key. The
            # id is deterministic over (pmcid, chunk_id, position, claim).
            for cs in chunk_summaries:
                for pos, f in enumerate(cs.findings):
                    f.set_finding_id(
                        compute_finding_id(pmcid, cs.chunk_id, pos, f.claim)
                    )

            # ── Verbatim-from-DB (replace LLM paraphrases before grounding) ──
            self._replace_verbatim_from_db(chunk_summaries)

            # ── Grounding (matches sync ordering) ─────────────────────────────
            grounding_rejected: list = []
            if self._grounding is not None:
                new_summaries = []
                for cs in chunk_summaries:
                    kept_cs, dropped = self._grounding.filter_findings_with_scores(cs)
                    new_summaries.append(kept_cs)
                    for f in dropped:
                        grounding_rejected.append((cs.chunk_id, f))
                chunk_summaries = new_summaries

            # ── MAP artifact persistence (filesystem + DB) ────────────────────
            persist_map_artifacts(writer, pmcid, chunk_summaries, grounding_rejected)
            self._persist_map_findings(pipeline_run_db_id, pmcid, chunk_summaries)

            # ── Optional modern chain — produces canonical/relate/final ──────
            normal_findings: list = []
            finding_groups:  list = []
            canonical_rules: list = []
            relations:       list = []
            relate_raw_pairs: list = []
            relate_skipped: list = []
            final_rules:     list = []
            non_groupable: list = []

            if self._run_modern_pipeline:
                import time as _time  # noqa: PLC0415
                all_findings = [f for cs in chunk_summaries for f in cs.findings]
                logger.info("[%s] NORMALIZE — start (%d findings)", pmcid, len(all_findings))
                t0 = _time.perf_counter()
                normal_findings = self._normalize.normalize(all_findings, pmcid)
                logger.info("[%s] NORMALIZE — done [%.1fs] → %d normal findings",
                            pmcid, _time.perf_counter() - t0, len(normal_findings))
                persist_normalize_artifacts(writer, pmcid, normal_findings)
                nf_db_id_map = self._persist_normal_findings(
                    pipeline_run_db_id, pmcid, normal_findings,
                )

                groupable     = [nf for nf in normal_findings if is_groupable(nf)]
                non_groupable = [nf for nf in normal_findings if not is_groupable(nf)]
                logger.info("[%s] GROUP — start (%d groupable, %d non-groupable)",
                            pmcid, len(groupable), len(non_groupable))
                t0 = _time.perf_counter()
                finding_groups = self._group.group(groupable, pmcid)
                logger.info("[%s] GROUP — done [%.1fs] → %d groups",
                            pmcid, _time.perf_counter() - t0, len(finding_groups))
                persist_group_artifacts(writer, pmcid, finding_groups, non_groupable)
                fg_db_id_map = self._persist_finding_groups(
                    pipeline_run_db_id, pmcid, finding_groups, nf_db_id_map,
                )

                logger.info("[%s] CANONICALIZE — start (%d groups)", pmcid, len(finding_groups))
                t0 = _time.perf_counter()
                nf_by_id: dict[str, NormalFinding] = {nf.normal_id: nf for nf in normal_findings}
                canonical_rules = self._canonicalize.canonicalize(
                    finding_groups, nf_by_id, pmcid
                )
                logger.info("[%s] CANONICALIZE — done [%.1fs] → %d canonical rules",
                            pmcid, _time.perf_counter() - t0, len(canonical_rules))
                # UMLS enrichment is a no-op when scispacy is unavailable but
                # the import itself can take 5-30s on first use.
                logger.info("[%s] UMLS enrichment — start (loading scispacy may be slow)", pmcid)
                t0 = _time.perf_counter()
                from ..helpers.entity_linker import enrich_rules_with_cuis  # noqa: PLC0415
                enrich_rules_with_cuis(canonical_rules)
                logger.info("[%s] UMLS enrichment — done [%.1fs]",
                            pmcid, _time.perf_counter() - t0)
                persist_canonicalize_artifacts(writer, pmcid, canonical_rules)
                cr_db_id_map = self._persist_canonical_rules(
                    pipeline_run_db_id, pmcid, canonical_rules, fg_db_id_map,
                )
                # Cross-paper relate must run after canonical_rules are in DB.
                self._corpus_relate_incremental(pmcid, canonical_rules)

                logger.info("[%s] RELATE — start (%d canonical rules)",
                            pmcid, len(canonical_rules))
                t0 = _time.perf_counter()
                if len(canonical_rules) < 2:
                    logger.info("[%s] RELATE — skipped (need ≥2 rules to compare)", pmcid)
                relations, relate_raw_pairs, relate_skipped = self._relate.relate(
                    canonical_rules, pmcid,
                )
                logger.info(
                    "[%s] RELATE — done [%.1fs] → %d relations, %d raw pairs, %d gate-skipped",
                    pmcid, _time.perf_counter() - t0,
                    len(relations), len(relate_raw_pairs), len(relate_skipped),
                )
                persist_relate_artifacts(
                    writer, pmcid, relations, relate_raw_pairs, relate_skipped,
                )
                self._persist_relations(pipeline_run_db_id, pmcid, relations, cr_db_id_map)

                logger.info("[%s] RESOLVE — start (%d canonical rules, %d relations)",
                            pmcid, len(canonical_rules), len(relations))
                t0 = _time.perf_counter()
                final_rules = self._resolve.resolve(canonical_rules, relations, pmcid)
                logger.info("[%s] RESOLVE — done [%.1fs] → %d final rules",
                            pmcid, _time.perf_counter() - t0, len(final_rules))
                persist_resolve_artifacts(writer, pmcid, final_rules, relations)
                self._persist_final_rules(pipeline_run_db_id, pmcid, final_rules, cr_db_id_map)

            # ── Optional REDUCE → RULES (legacy) ─────────────────────────────
            master = None
            rules  = None
            contradiction_report = None
            if self._run_reduce:
                if self._reduce is None or self._rules is None:
                    raise RuntimeError(
                        "run_reduce=True but ReduceStage/RuleStage were not built. "
                        "Reconstruct the runner with run_reduce=True."
                    )
                logger.info("[%s] REDUCE — %d chunks", pmcid, len(chunk_summaries))
                master = self._reduce.reduce(chunk_summaries, pmcid)
                logger.info("[%s] RULES", pmcid)
                rules = self._rules.extract(master, pmcid)
                if self._grounding is not None:
                    rules = self._grounding.filter_rules(rules)
                if self._contradiction is not None:
                    contradiction_report = self._contradiction.detect(rules)

            # ── Rejection summary (sync parity) ───────────────────────────────
            rejection_summary = _build_rejection_summary(
                pmcid=pmcid,
                grounding_threshold=(
                    self._grounding.threshold if self._grounding is not None else None
                ),
                map_findings_total=map_findings_total,
                grounding_rejected=grounding_rejected,
                normal_findings=normal_findings,
                non_groupable_nfs=non_groupable,
            )
            self._persist_rejection_summary(pipeline_run_db_id, rejection_summary)

            # ── NER + UMLS linking (optional) ─────────────────────────────────
            if self._run_ner and self._db is not None:
                logger.info("[%s] NER — running entity extraction + UMLS linking", pmcid)
                try:
                    from named_entity_recognition.ner import run_ner_on_db  # noqa: PLC0415
                    run_ner_on_db(pmcid, save_to_db=True, force=False)
                except Exception as ner_exc:
                    logger.warning("[%s] NER failed (non-fatal): %s", pmcid, ner_exc)

            result = {
                "status": "success",
                "run_id": run_id,
                "pmcid":  pmcid,
                "canonical_rules":  [cr.model_dump() for cr in canonical_rules],
                "relations":        [r.model_dump()  for r in relations],
                "relate_raw_pairs": [p.model_dump()  for p in relate_raw_pairs],
                "final_rules":      [fr.model_dump() for fr in final_rules],
                "audit_trail": {
                    "map_chunks": [cs.model_dump() for cs in chunk_summaries],
                },
                "map_run_metadata": {
                    "schema_version":    handle.schema_version or MAP_SCHEMA_VERSION,
                    "prompt_version":    handle.prompt_version or MAP_PROMPT_VERSION,
                    "stage_name":        handle.stage_name or MAP_STAGE_NAME,
                    "cascade_profile":   handle.cascade_profile or self._cascade_profile,
                    "cascade_signature": handle.cascade_signature or self._cascade_signature,
                    "l1_voters": [{"provider": v.provider, "model": v.model} for v in self._l1],
                    "l2_voters": [{"provider": v.provider, "model": v.model} for v in self._l2],
                    "l3_voter":  {"provider": self._l3.provider, "model": self._l3.model},
                },
                "rejection_summary": rejection_summary.model_dump(),
                "pipeline_config_hash": self._pipeline_config_hash(),
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
            self._save_result(result)
            logger.info("[%s] Result saved to %s", pmcid, self._result_path(pmcid))

            self._write_batch_cost_artifacts(handle, writer, run_id, pmcid)

            # Issue G — post-run row-count audit (batch parity with sync).
            if self._db is not None:
                try:
                    from ..health_checks import assert_persistence_row_counts  # noqa: PLC0415
                    assert_persistence_row_counts(self._db, pmcid)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[%s] Post-run row-count audit raised — ignored: %s",
                        pmcid, exc,
                    )

            self._finish_pipeline_run(
                pipeline_run_db_id, "success",
                narrative_summary=master.narrative_summary if master else None,
            )

            if writer is not None:
                writer.finalize("completed")
            return result

        except Exception as exc:
            self._finish_pipeline_run(pipeline_run_db_id, "failed", error=str(exc))
            if writer is not None:
                writer.write_error(stage="batch_finalize", error=str(exc), pmcid=pmcid)
                writer.finalize("failed", error=str(exc))
            raise

    # ── Filesystem artifact helpers ────────────────────────────────────────────

    def _make_run_id(self, pmcid: str) -> str:
        """Mirror SummarizationRunner._make_run_id format for consistency."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{pmcid}_{ts}"

    def _make_artifact_writer(
        self, run_id: str, pmcid: str, cascade_signature: str | None = None,
    ) -> RunArtifactWriter | None:
        """Build the per-run filesystem writer when ``artifact_root`` is set.

        Returns None when persistence is disabled. Failure is logged but never
        fatal — filesystem artifacts are observational.
        """
        if self._artifact_root is None:
            return None
        try:
            cfg = self._cfg_full
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
                "voter_models":        [v.model for v in self._l1],
                "level2_voter_models": [v.model for v in self._l2],
                "escalation_model":    self._l3.model,
            }
            cascade_sig = cascade_signature or self._cascade_signature
            manifest = RunManifest(
                run_id=run_id,
                artifact_root=self._artifact_root,
                timestamp_start=datetime.now(tz=timezone.utc).isoformat(),
                papers=[pmcid],
                schema_version=MAP_SCHEMA_VERSION,
                prompt_version=MAP_PROMPT_VERSION,
                cascade_signature=cascade_sig,
                config=self._config_snapshot,
                models=models,
                thresholds=thresholds,
                chunk_size=cfg.map.chunk_size,
            )
            from ..persistence import compute_pipeline_config_hash  # noqa: PLC0415
            manifest.extra["pipeline_config_hash"] = compute_pipeline_config_hash(
                config=self._config_snapshot,
                thresholds=thresholds,
                models=models,
                schema_version=MAP_SCHEMA_VERSION,
                prompt_version=MAP_PROMPT_VERSION,
                cascade_signature=cascade_sig,
            )
            return RunArtifactWriter(
                run_id=run_id, root_dir=self._artifact_root, manifest=manifest,
            )
        except Exception as exc:
            logger.warning("[%s] artifact writer setup failed: %s", pmcid, exc)
            return None

    def _write_batch_cost_artifacts(
        self,
        handle: BatchHandle,
        writer: RunArtifactWriter | None,
        run_id: str,
        pmcid: str,
    ) -> None:
        """Emit per-(level, model) usage records from handle.token_usage and write reports.

        Per-chunk granularity is not preserved across batch resumes (only the
        per-level/per-model totals on the handle survive); we synthesize one
        ``LLMUsageRecord`` per (level, model) instead.  This loses item_id
        attribution but keeps the cost report reproducible from durable state.
        """
        cost_cfg = self._cfg_full.cost
        if not cost_cfg.enable_cost_report:
            return
        try:
            collector = UsageCollector(run_id=run_id, paper_id=pmcid)
            level_to_cascade = {"l1": 1, "l2": 2, "l3": 3}
            level_to_role = {"l1": "voter", "l2": "voter", "l3": "escalator"}

            def _spec_for_model(level: str, model_id: str) -> tuple[str, str]:
                voters = (
                    self._l1 if level == "l1"
                    else self._l2 if level == "l2"
                    else [self._l3]
                )
                for v in voters:
                    if v.model == model_id:
                        return v.provider, v.model
                return "unknown", model_id

            for level, per_model in handle.token_usage.items():
                cascade_level = level_to_cascade.get(level)
                role = level_to_role.get(level, "voter")
                for model_id, tokens in per_model.items():
                    in_t = int(tokens.get("input", 0) or 0)
                    out_t = int(tokens.get("output", 0) or 0)
                    if in_t == 0 and out_t == 0:
                        continue
                    provider, model = _spec_for_model(level, model_id)
                    collector.record_batch(
                        stage="MAP", role=role,
                        provider=provider, model=model,
                        input_tokens=in_t, output_tokens=out_t,
                        cascade_level=cascade_level,
                        substage=level,
                    )

            if len(collector) == 0:
                return

            override = cost_cfg.cost_report_output_dir
            if override:
                out_dir = Path(override)
            elif writer is not None:
                out_dir = writer.run_dir / "cost"
            else:
                out_dir = self._output_dir / "cost" / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            if cost_cfg.write_usage_jsonl:
                collector.write_jsonl(out_dir / "llm_usage_records.jsonl")
            book = PriceBook.load(cost_cfg.model_prices_path)
            write_cost_reports(collector.records(), book, out_dir)
            logger.info("[%s] batch cost report written to %s", pmcid, out_dir)
        except Exception as exc:
            logger.warning("[%s] batch cost report write failed (non-fatal): %s", pmcid, exc)

    # ── DB / cache / verbatim helpers (parity with sync runner) ────────────────
    # TODO: deduplicate with SummarizationRunner — these are line-for-line copies.
    # Lift into pipeline/stages/summarization/persistence.py as module-level fns
    # taking ``db`` as a parameter so both runners share one implementation.

    def _pipeline_config_hash(self) -> str:
        cfg = self._cfg_full
        thresholds = {
            "grounding_threshold": (
                self._grounding.threshold if self._grounding is not None else None
            ),
            "entailment_threshold":   cfg.relate.entailment_threshold,
            "contradiction_threshold": cfg.relate.contradiction_threshold,
            "map_theta":              cfg.map.theta,
            "map_reject_theta":       cfg.map.reject_theta,
            "contradiction_similarity_threshold": cfg.contradiction_similarity_threshold,
            "enable_router":          self._enable_router,
            "router_single_voter_policy": (
                self._router_single_voter_policy if self._enable_router else None
            ),
            # B-049 behavioural stamp — bumps invalidate cached summaries when
            # canonicalization semantics change.
            "canonicalize_direction_policy_version": CANONICALIZE_DIRECTION_POLICY_VERSION,
            # B-051 behavioural stamp — bumps invalidate cached summaries when
            # the MAP agreement-gate hard-fail policy changes. Companion to
            # MAP_SCHEMA_VERSION (chunk-level PipelineCache); both must move
            # together to fully invalidate stale KEEP decisions.
            "map_agreement_policy_version": MAP_AGREEMENT_POLICY_VERSION,
        }
        models = {
            "voter_models":        [v.model for v in self._l1],
            "level2_voter_models": [v.model for v in self._l2],
            "escalation_model":    self._l3.model,
        }
        return compute_pipeline_config_hash(
            config=self._config_snapshot,
            thresholds=thresholds,
            models=models,
            schema_version=MAP_SCHEMA_VERSION,
            prompt_version=MAP_PROMPT_VERSION,
            cascade_signature=self._cascade_signature,
        )

    def _result_path(self, pmcid: str) -> Path:
        return self._summaries_dir / f"{pmcid}.json"

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
        current_hash = self._pipeline_config_hash()
        if stored_hash != current_hash:
            logger.info(
                "[%s] cached result stale (config hash %s != %s) — re-running",
                pmcid, stored_hash, current_hash,
            )
            return None
        return data

    def _save_result(self, result: dict) -> None:
        # Stamp the hash so future runs can decide whether the cache is valid.
        result.setdefault("pipeline_config_hash", self._pipeline_config_hash())
        p = self._result_path(result["pmcid"])
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def _create_pipeline_run(self, pmcid: str, run_id: str) -> int | None:
        if self._db is None:
            return None
        try:
            from database import Document, PipelineRun  # noqa: PLC0415
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
        if db_id is None:
            return
        try:
            from database import PipelineRun  # noqa: PLC0415
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

    def _clear_normalized_run_data(self, db_id: int, pmcid: str) -> None:
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
                fg_table = SumFindingGroup.__table__
                subq = fg_table.select().where(
                    (fg_table.c.pipeline_run_id == db_id) & (fg_table.c.pmcid == pmcid)
                ).with_only_columns(fg_table.c.id)
                t = SumGroupMember.__table__
                conn.execute(t.delete().where(t.c.finding_group_id.in_(subq)))
                t = SumFindingGroup.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
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
        from database import TextElement  # type: ignore
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

    def _persist_map_findings(self, db_id, pmcid, chunk_summaries) -> None:
        if db_id is None:
            return
        try:
            from database.models import SumMapFinding
            from ..models import FindingScope
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

    def _persist_normal_findings(self, db_id, pmcid, normal_findings) -> dict[str, int]:
        if db_id is None:
            return {}
        self._clear_normalized_run_data(db_id, pmcid)
        nf_id_map: dict[str, int] = {}
        try:
            from database.models import SumNormalFinding, SumNormalFindingSpan
            from ..models import FindingScope
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
                    session.flush()
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
            return {}
        return nf_id_map

    def _persist_finding_groups(self, db_id, pmcid, finding_groups, nf_db_id_map) -> dict[str, int]:
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
                    session.flush()
                    group_id_map[fg.group_id] = row.id
                    for normal_id in fg.member_ids:
                        nf_db_id = nf_db_id_map.get(normal_id)
                        session.add(SumGroupMember(
                            finding_group_id  = row.id,
                            normal_finding_id = nf_db_id,
                            normal_id         = normal_id,
                        ))
            logger.info(
                "[%s] DB: persisted %d finding groups (run_id=%d)",
                pmcid, len(group_id_map), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist finding groups: %s", pmcid, exc)
            return {}
        return group_id_map

    def _persist_canonical_rules(self, db_id, pmcid, canonical_rules, fg_db_id_map) -> dict[str, int]:
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

    def _persist_relations(self, db_id, pmcid, relations, cr_db_id_map) -> None:
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

    def _persist_final_rules(self, db_id, pmcid, final_rules, cr_db_id_map) -> None:
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
                    score_mode           = fr.score_mode,
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

    def _persist_rejection_summary(self, db_id, rejection_summary) -> None:
        if db_id is None:
            return
        try:
            from database.models import SumRejectionSummary, SumRejectedFinding
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
                session.flush()
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

    def _corpus_relate_incremental(self, pmcid: str, canonical_rules: list) -> None:
        if self._db is None:
            return
        try:
            from ..helpers.corpus_relate import CorpusRelateStage  # noqa: PLC0415
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

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _make_chunk_map(self, sentences: list[dict]) -> dict[str, list[dict]]:
        chunk_map: dict[str, list[dict]] = {}
        for i in range(0, len(sentences), self._chunk_size):
            chunk_id = f"C{i // self._chunk_size + 1}"
            chunk_map[chunk_id] = sentences[i : i + self._chunk_size]
        return chunk_map

    def _process_level(
        self,
        handle: BatchHandle,
        raw_results: list[BatchResult],
        level: str,
        strip_flags: list[bool],
        next_voters: list[VoterBatchConfig],
        next_level: str,
        next_strip: list[bool],
        next_phase: BatchPhase,
        escalated_attr: str,
        chunk_ids_to_process: list[str] | None = None,
    ) -> None:
        """
        Parse batch results for one level, run agreement, and either:
        - Finalise kept chunks
        - Submit the next level for escalated chunks
        - Mark COMPLETE if nothing needs escalation
        """
        pmcid = handle.pmcid
        targets = chunk_ids_to_process or list(handle.chunk_map.keys())

        # Group raw results by chunk_id
        by_chunk: dict[str, list[BatchResult]] = {}
        for res in raw_results:
            parts = res.custom_id.split("__")
            chunk_id = parts[1] if len(parts) > 1 else "unknown"
            by_chunk.setdefault(chunk_id, []).append(res)

        # Store raw content for debugging and accumulate token usage per model
        raw_store = getattr(handle, f"{level}_raw")
        current_voters = self._l1 if level == "l1" else self._l2
        level_usage = handle.token_usage.setdefault(level, {})
        for res in raw_results:
            if res.content:
                raw_store[res.custom_id] = res.content
            parts = res.custom_id.split("__")
            voter_idx = int(parts[-1]) if len(parts) >= 4 and parts[-1].isdigit() else 0
            model_id = current_voters[voter_idx].model if voter_idx < len(current_voters) else "unknown"
            m = level_usage.setdefault(model_id, {"input": 0, "output": 0})
            m["input"]  += res.input_tokens
            m["output"] += res.output_tokens

        # ── Pass 1: parse all voter outputs for every chunk ───────────────────
        # chunk_voters[chunk_id] is a length-N list aligned to current_voters
        # (original voter-spec index): slot vi holds the parsed result for that
        # voter, or None when the batch job for that voter was missing/unparseable.
        # Preserving the index mapping is load-bearing for producer attribution
        # at evaluate_chunk time — see B-041.
        chunk_voters: dict[str, list[AuditableSummary | None]] = {}
        for chunk_id in targets:
            results = by_chunk.get(chunk_id, [])
            results_sorted = sorted(results, key=lambda r: r.custom_id)
            voters_full: list[AuditableSummary | None] = [None] * len(current_voters)
            for res in results_sorted:
                vi_part = res.custom_id.split("__")[-1]
                vi = int(vi_part) if vi_part.isdigit() else 0
                if not (0 <= vi < len(voters_full)):
                    continue
                strip = strip_flags[vi] if vi < len(strip_flags) else False
                parsed = parse_result(res, strip_thinking=strip)
                if parsed is not None:
                    # Chunk_id round-trip: voters occasionally rename the
                    # chunk (e.g. emit "C5" for an expected "C3"). Batch
                    # mode can't re-submit, so log + repair in place.
                    if parsed.chunk_id != chunk_id:
                        from ..enum_logging import log_bad_finding  # noqa: PLC0415
                        log_bad_finding(
                            raw={"returned_chunk_id": parsed.chunk_id,
                                 "expected_chunk_id": chunk_id},
                            error=f"chunk_id mismatch: voter returned "
                                  f"{parsed.chunk_id!r} for expected {chunk_id!r}",
                            context={
                                "stage": "AuditableSummary",
                                "pmcid": pmcid, "chunk_id": chunk_id,
                                "custom_id": res.custom_id,
                                "voter_index": vi, "level": level,
                                "source": "batch._process_level",
                            },
                        )
                        parsed = parsed.model_copy(update={"chunk_id": chunk_id})
                    voters_full[vi] = parsed
            chunk_voters[chunk_id] = voters_full

        # ── Pass 2: batch-embed all unique claims upfront ─────────────────────
        from ..agreement.embedding import _claims as _extract_claims
        all_texts: list[str] = list({
            c
            for voters in chunk_voters.values()
            for v in voters
            for c in _extract_claims(v)
        })
        embed_cache: dict[str, list[float]] = {}
        if all_texts:
            raw_embed = self._embed_fn
            logger.info(
                "[%s] Pre-embedding %d unique claim strings across %d chunks",
                pmcid, len(all_texts), len(targets),
            )
            embs = raw_embed(all_texts)
            embed_cache = dict(zip(all_texts, embs))

        def _cached_embed(texts: list[str]) -> list[list[float]]:
            result = []
            misses: list[tuple[int, str]] = []
            for idx, t in enumerate(texts):
                if t in embed_cache:
                    result.append(embed_cache[t])
                else:
                    misses.append((idx, t))
                    result.append(None)  # type: ignore[arg-type]
            if misses:
                new_embs = raw_embed([t for _, t in misses])
                for (idx, t), e in zip(misses, new_embs):
                    embed_cache[t] = e
                    result[idx] = e
            return result  # type: ignore[return-value]

        # Mirror the runner-level scorer so the per-level decision uses the
        # same Soiffer-style max-consensus + centrality path. The cached embed
        # function is plugged into EmbeddingSimilarityStrategy so the
        # pre-embedded claim cache is reused for every pair.
        agreement = AgreementChecker(
            scorer=SemanticAgreementScorer(
                strategy=EmbeddingSimilarityStrategy(embed_fn=_cached_embed),
            ),
            theta=self._agreement.theta,
            reject_theta=self._agreement.reject_theta,
        )
        # Grounding-first router: built fresh per call so it shares the same
        # cached AgreementChecker as the agreement gate. Disabled when
        # self._enable_router is False to keep parity with sync's legacy path.
        router: MapOutputRouter | None = (
            MapOutputRouter(
                agreement_checker=agreement,
                single_voter_policy=self._router_single_voter_policy,  # type: ignore[arg-type]
            )
            if self._enable_router else None
        )

        # Per-paper cascade decision log: one JSONL row per chunk decision at
        # this level. Surfaces cascade behaviour for offline analysis.
        cascade_log = CascadeDecisionLog(
            self._cascade_log_dir / f"{pmcid}.jsonl"
        )
        run_id = handle.cascade_signature or "unknown"
        # Voter specs for the current level — used by the decision log to map
        # best_index → (provider, model) of the selected voter.
        level_voter_specs: list[tuple[str, str]] = [
            (cfg.provider, cfg.model) for cfg in current_voters
        ]

        # ── Pass 3: agreement scoring from cache — no API calls ───────────────
        escalated: list[str] = []

        for chunk_id in targets:
            voters_full = chunk_voters[chunk_id]
            survivor_indices = [i for i, v in enumerate(voters_full) if v is not None]
            voters: list[AuditableSummary] = [voters_full[i] for i in survivor_indices]

            if not voters:
                logger.warning("[%s] Chunk %s: all %s voters failed — escalating", pmcid, chunk_id, level)
                escalated.append(chunk_id)
                continue

            source_text = _format_sentences(handle.chunk_map[chunk_id])

            # Shared evaluate_chunk path — same function the sync runner uses,
            # so KEEP/escalate semantics are identical across sync and batch.
            outcome = evaluate_chunk(
                voters,
                chunk=handle.chunk_map[chunk_id],
                pmcid=pmcid,
                source_text=source_text,
                agreement=agreement,
                router=router,
                voter_indices=survivor_indices,
            )

            # Voter-count regression guard — see B-019/B-020. A shortfall
            # almost always means the router classified voters as UNUSABLE
            # or a batch job failed silently. Emit a WARNING per occurrence
            # so the next regression surfaces immediately.
            expected_voter_count = (
                len(self._l1) if level == "l1"
                else len(self._l2) if level == "l2"
                else 1  # l3 single-voter by design
            )
            if len(voters) < expected_voter_count:
                logger.warning(
                    "[%s] cascade chunk %s %s ran with voter_count=%d expected=%d "
                    "— %d voter(s) missing (UNUSABLE? failed batch job? router strip?)",
                    pmcid, chunk_id, level.upper(),
                    len(voters), expected_voter_count,
                    expected_voter_count - len(voters),
                )
            try:
                cascade_log.record(make_decision_record(
                    outcome,
                    run_id=run_id,
                    pmcid=pmcid,
                    chunk_id=chunk_id,
                    level=level,
                    voter_count=len(voters),
                    cascade_signature=handle.cascade_signature,
                    cascade_profile=handle.cascade_profile,
                    voter_specs=level_voter_specs,
                ))
            except Exception as exc:  # noqa: BLE001 — log-and-continue
                logger.warning(
                    "Failed to record cascade decision for chunk %s level %s: %s",
                    chunk_id, level, exc,
                )

            if outcome.keep:
                assert outcome.best is not None  # invariant of ChunkOutcome
                handle.finalized[chunk_id] = outcome.best.model_dump()
            else:
                if outcome.routing_decision is not None:
                    logger.debug(
                        "[%s] Chunk %s router → %s (gate=%s reasons=%s)",
                        pmcid, chunk_id,
                        outcome.routing_decision.decision.value,
                        outcome.routing_decision.gate_origin.value,
                        [r.value for r in outcome.routing_decision.reason_codes],
                    )
                escalated.append(chunk_id)

        kept = len(targets) - len(escalated)
        logger.info(
            "[%s] %s complete: %d kept, %d escalated to %s",
            pmcid, level.upper(), kept, len(escalated), next_level.upper(),
        )

        if escalated:
            setattr(handle, escalated_attr, escalated)
            handle.phase = next_phase
            # In-run dedup: for each escalated chunk × next-level voter, if any
            # already-run voter at L1 (or L1+L2 when promoting to L3) has the
            # same (provider, model, temperature), reuse that voter's raw
            # content instead of issuing a duplicate API call.
            skip_voters, synthetic = self._compute_voter_dedup(
                handle, escalated, next_voters, next_level,
            )
            if synthetic:
                handle.synthetic_results.setdefault(next_level, []).extend(synthetic)
                logger.info(
                    "[%s] Voter dedup: reusing %d %s result(s) from earlier level(s); "
                    "saved equal number of API calls",
                    pmcid, len(synthetic), next_level.upper(),
                )
            handle.jobs = submit_level(
                handle.chunk_map, pmcid, next_voters,
                level=next_level, chunk_ids=escalated,
                skip_voters=skip_voters,
            )
        else:
            handle.phase = BatchPhase.COMPLETE
            logger.info("[%s] No escalations — batch MAP complete.", pmcid)

    @staticmethod
    def _voter_sig(cfg: VoterBatchConfig) -> tuple[str, str, float]:
        """Identity tuple used to decide whether two voter slots are duplicates."""
        return (cfg.provider, cfg.model, round(float(cfg.temperature), 3))

    def _compute_voter_dedup(
        self,
        handle: BatchHandle,
        escalated: list[str],
        next_voters: list[VoterBatchConfig],
        next_level: str,
    ) -> tuple[dict[str, set[int]], list[dict]]:
        """Decide which next-level voter slots are duplicates of earlier-level voters.

        Returns
        -------
        skip_voters:
            chunk_id → set of next-level voter indices to skip submitting.
        synthetic:
            BatchResult-equivalent dicts (custom_id, content, tokens=0) that
            will be merged into raw_results when the next level is processed.
            Token counts are zero so the duplicate call isn't double-counted —
            the original call was already charged at its own level.
        """
        pmcid = handle.pmcid

        # Build earlier-level signature → (level, voter_idx) map. L2 sees L1
        # only; L3 sees both L1 and L2. First match wins.
        earlier: list[tuple[str, list[VoterBatchConfig]]] = []
        if next_level in ("l2", "l3"):
            earlier.append(("l1", self._l1))
        if next_level == "l3":
            earlier.append(("l2", self._l2))

        sig_to_source: dict[tuple, tuple[str, int]] = {}
        for prior_level, voters in earlier:
            for vi, cfg in enumerate(voters):
                sig_to_source.setdefault(self._voter_sig(cfg), (prior_level, vi))

        skip_voters: dict[str, set[int]] = {}
        synthetic: list[dict] = []

        for chunk_id in escalated:
            for vi, cfg in enumerate(next_voters):
                src = sig_to_source.get(self._voter_sig(cfg))
                if src is None:
                    continue
                prior_level, prior_vi = src
                prior_custom_id = f"{pmcid}__{chunk_id}__{prior_level}__{prior_vi}"
                raw_store = getattr(handle, f"{prior_level}_raw", {})
                content = raw_store.get(prior_custom_id)
                if not content:
                    # Prior voter failed / produced no content — no point reusing.
                    continue
                next_custom_id = f"{pmcid}__{chunk_id}__{next_level}__{vi}"
                synthetic.append({
                    "custom_id": next_custom_id,
                    "content": content,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": None,
                })
                skip_voters.setdefault(chunk_id, set()).add(vi)

        return skip_voters, synthetic

    def _collect_l3(self, handle: BatchHandle, raw_results: list[BatchResult]) -> None:
        """Parse L3 (escalation model) results and finalise remaining chunks."""
        pmcid = handle.pmcid
        targets = handle.l3_chunk_ids

        by_chunk: dict[str, list[BatchResult]] = {}
        for res in raw_results:
            parts = res.custom_id.split("__")
            chunk_id = parts[1] if len(parts) > 1 else "unknown"
            by_chunk.setdefault(chunk_id, []).append(res)

        l3_model = self._l3.model
        l3_level = handle.token_usage.setdefault("l3", {})
        l3_m = l3_level.setdefault(l3_model, {"input": 0, "output": 0})
        for res in raw_results:
            if res.content:
                handle.l3_raw[res.custom_id] = res.content
            l3_m["input"]  += res.input_tokens
            l3_m["output"] += res.output_tokens

        for chunk_id in targets:
            results = by_chunk.get(chunk_id, [])
            if not results:
                logger.warning("[%s] Chunk %s: no L3 result — skipping", pmcid, chunk_id)
                continue
            parsed = parse_result(results[0], strip_thinking=handle.l3_strip)
            if parsed is not None:
                handle.finalized[chunk_id] = parsed.model_dump()
            else:
                logger.warning("[%s] Chunk %s: L3 parse failed — skipping", pmcid, chunk_id)

        logger.info(
            "[%s] L3 complete. Total finalised: %d/%d chunks.",
            pmcid, len(handle.finalized), len(handle.chunk_map),
        )
        handle.phase = BatchPhase.COMPLETE


# ── Rejection-summary helpers (sync parity) ─────────────────────────────────────
# TODO: deduplicate with SummarizationRunner — these are line-for-line copies of
# the module-level helpers at the bottom of ``pipeline/stages/summarization/runner.py``.
# Lift to a shared helper module when consolidating sync/batch persistence.

def _non_groupable_reason(nf: NormalFinding) -> str:
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
    from collections import Counter  # noqa: PLC0415

    rejected_items: list[RejectedFinding] = []
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

    cat_counts: Counter[str] = Counter(f.category for _, f in grounding_rejected)

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
