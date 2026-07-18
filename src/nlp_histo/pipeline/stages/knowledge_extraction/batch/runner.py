"""
BatchKnowledgeExtractionRunner — asynchronous batch variant of KnowledgeExtractionRunner.

Workflow
--------
Each call to ``advance()`` checks whether the current batch phase is done
and, if so, runs agreement scoring and submits the next phase.
The handle is persisted to disk between calls so the process can exit freely.

    runner = BatchKnowledgeExtractionRunner(l1_voters=[...], ...)

    handle = runner.submit(file_data)       # L1 jobs submitted; handle saved
    handle = runner.advance(handle)         # call again until COMPLETE
    result = runner.finalize(handle)        # run post-MAP chain + persist

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
    SemanticAgreementScorer,
    evaluate_chunk,
    make_decision_record,
)
from ..routing import MapOutputRouter
from ..config import KnowledgeExtractionConfig
from ..stages.canonicalize_stage import CanonicalizeStage
from ..stages.group_stage import GroupStage, is_groupable
from ..stages.normalize_stage import NormalizeStage
from ..stages.relate_stage import RelateStage
from ..stages.resolve_stage import ResolveStage
from ..grounding.grounding_filter import GroundingFilter
from ..stages.map_stage import _format_sentences
from ..models import (
    AuditableSummary,
    CANONICALIZE_DIRECTION_POLICY_VERSION,
    MAP_AGREEMENT_POLICY_VERSION,
    MAP_PROMPT_VERSION,
    MAP_SCHEMA_VERSION,
    MAP_STAGE_NAME,
    NormalFinding,
    compute_cascade_signature,
    compute_finding_id,
)
from ..persistence import (
    RunArtifactWriter,
    RunManifest,
    build_rejection_summary as _build_rejection_summary,
    clear_normalized_run_data as _persistence_clear_normalized_run_data,
    compute_pipeline_config_hash,
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
from ..costing import PriceBook, UsageCollector, write_cost_reports
from .dispatch import build_providers, parse_result, submit_level
from .models import BatchHandle, BatchPhase, BatchResult, VoterBatchConfig

logger = logging.getLogger(__name__)


class BatchKnowledgeExtractionRunner:
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
        Retained for constructor compatibility with the sync runner; the batch
        cascade uses the l1/l2/l3 voter configs and does not call it.
    config:
        All numeric/boolean pipeline knobs.  Defaults to KnowledgeExtractionConfig().
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
        config: KnowledgeExtractionConfig | None = None,
        output_dir: Path = Path("out/summaries"),
        handle_dir: Path | None = None,
        embed_fn=None,
        cascade_profile: str = "custom",
        artifact_root: Path | None = None,
        artifact_run_id: str | None = None,
        run_modern_pipeline: bool = True,
        db=None,
        force_rerun: bool = False,
        run_ner: bool = False,
        enable_router: bool = False,
        router_single_voter_policy: str = "escalate",
        legacy_single_voter_policy: str = "keep",
    ) -> None:
        from ..agreement.providers import OpenAIEmbedder
        cfg = config or KnowledgeExtractionConfig()
        self._l1 = l1_voters
        self._l2 = l2_voters
        self._l3 = l3_model
        self._chunk_size = cfg.map.chunk_size
        self._embed_fn = embed_fn or OpenAIEmbedder()
        # Default scorer mirrors KnowledgeExtractionRunner: SemanticAgreementScorer
        # (max-consensus + centrality-based best-output selection) over the
        # strategy selected by cfg.agreement.scorer_kind (default "embedding").
        # theta / reject_theta stay on AgreementChecker so the scorer keeps
        # theta=None and defers the decision boundary to the checker. Invalid
        # scorer_kind raises here, before any batch job is submitted.
        self._agreement = AgreementChecker(
            scorer=SemanticAgreementScorer.from_agreement_config(
                cfg.agreement, embed_fn=self._embed_fn,
            ),
            theta=cfg.map.theta,
            reject_theta=cfg.map.reject_theta,
            single_voter_policy=legacy_single_voter_policy,  # type: ignore[arg-type]
            force_escalate_on_polarity_conflict=(
                cfg.agreement.force_escalate_on_polarity_conflict
            ),
        )
        # Grounding-first router config. When enabled, _process_level runs
        # schema + provenance validation before the agreement gate, drops
        # unusable voters, and (at L1) escalates straight to L3 — mirroring
        # the sync MapStage router path.
        self._enable_router = enable_router
        self._router_single_voter_policy = router_single_voter_policy
        # Legacy-path N=1 policy — see KnowledgeExtractionRunner for full rationale.
        # Captured here so per-call AgreementCheckers built in _process_level
        # inherit the same policy as the runner-level one (otherwise a
        # `--legacy-single-voter-policy=escalate` flag would silently revert
        # to "keep" inside the per-level processing). Hash-gated on
        # `not enable_router` to mirror the router policy's gating.
        self._legacy_single_voter_policy = legacy_single_voter_policy
        # escalation_llm is retained on the runner for API compatibility with
        # the sync runner's constructor; the batch cascade uses l1/l2/l3_model.
        self._escalation_llm = escalation_llm
        self._grounding = (
            GroundingFilter(cfg.grounding.threshold)
            if cfg.grounding.threshold is not None else None
        )
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)
        self._handle_dir = handle_dir or (output_dir / "batch_handles")
        self._handle_dir.mkdir(parents=True, exist_ok=True)
        # Per-paper cascade decision log directory — mirrors KnowledgeExtractionRunner.
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
        # When ``run_modern_pipeline`` is False, the post-MAP stages are skipped
        # and finalize() persists MAP output only.
        self._cfg_full = cfg
        self._run_modern_pipeline = run_modern_pipeline
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
        # ``config_layout_version`` is stamped so historical-vs-current artifact
        # consumers can branch on the dataclass shape — see
        # KnowledgeExtractionConfig.CONFIG_LAYOUT_VERSION for the version log.
        import dataclasses
        from ..config import CONFIG_LAYOUT_VERSION
        self._config_snapshot = {
            **dataclasses.asdict(cfg),
            "config_layout_version": CONFIG_LAYOUT_VERSION,
            "cascade_profile":   cascade_profile,
            "cascade_signature": self._cascade_signature,
            "voter_models":         [v.model for v in self._l1],
            "level2_voter_models":  [v.model for v in self._l2],
            "escalation_model":     self._l3.model,
        }

    # Public API

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
                # Log provider/model/request_count so a silent multi-model
                # rejection (B-020 class) is obvious in stdout.
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
        Run the modern post-MAP chain on the batch MAP output, persist
        filesystem artifacts when configured, and return the assembled
        result dict.

        Only call when ``handle.phase == BatchPhase.COMPLETE``.
        """
        if handle.phase != BatchPhase.COMPLETE:
            raise RuntimeError(
                f"Cannot finalize: handle phase is {handle.phase.value!r}, not 'complete'."
            )

        pmcid = handle.pmcid

        # Cache short-circuit (set by submit() when valid result on disk)
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

        # Citation filter (B-080) — single chokepoint covering every cascade
        # level (L1/L2 KEEP at finalized[1259], L3 at finalized[1407]). Runs
        # before grounding / _replace_verbatim_from_db so the optional verbatim
        # check sees the cited sentence. Mirrors sync MapStage._cascade.
        chunk_summaries = []
        _cit = self._cfg_full.citation
        # B-083 — a handle rebuilt from cached MAP (rebuild_from_cached_map's
        # _build_handle_from_cached_json) sets `finalized` but never `chunk_map`,
        # because the cached audit_trail.map_chunks carry no per-chunk source
        # index. With no source index the citation filter has nothing to validate
        # against and would emit one "empty chunk" WARNING per chunk. Detect the
        # whole-handle case, skip the filter, and note it once at debug. Verified
        # no-op: under the shipped config the filter drops 0 findings anyway (see
        # B-083). The per-chunk warning stays live for the real pipeline, where a
        # single empty chunk among populated ones is suspicious.
        _citation_active = _cit.enabled and bool(handle.chunk_map)
        if _cit.enabled and not handle.chunk_map:
            logger.debug(
                "[%s] citation filter skipped: rebuild handle carries no per-chunk "
                "source index (%d chunks); see B-083", pmcid, len(handle.finalized),
            )
        for chunk_id, v in handle.finalized.items():
            summary = AuditableSummary.model_validate(v)
            if _citation_active and summary.findings:
                from nlp_histo.pipeline.stages.knowledge_extraction.provenance.citation_filter import filter_summary_by_citation  # noqa: PLC0415
                n_before = len(summary.findings)
                summary, dropped = filter_summary_by_citation(
                    summary, handle.chunk_map.get(chunk_id) or [], pmcid,
                    check_verbatim=_cit.check_verbatim,
                    fabricated_threshold=_cit.fabricated_threshold,
                )
                if dropped:
                    logger.info(
                        "[%s] chunk %s: citation filter dropped %d/%d finding(s).",
                        pmcid, chunk_id, len(dropped), n_before,
                    )
            chunk_summaries.append(summary)

        # Filesystem persistence setup (opt-in)
        run_id = self._artifact_run_id_override or self._make_run_id(pmcid)
        writer = self._make_artifact_writer(run_id, pmcid, handle.cascade_signature)
        pipeline_run_db_id = handle.pipeline_run_db_id

        try:
            # Track total MAP findings before any filtering — used by
            # rejection_summary later.
            map_findings_total = sum(len(cs.findings) for cs in chunk_summaries)

            # Assign stable finding_id before grounding so downstream stages
            # (and rejected-finding artifacts) share the same lineage key. The
            # id is deterministic over (pmcid, chunk_id, position, claim).
            for cs in chunk_summaries:
                for pos, f in enumerate(cs.findings):
                    f.set_finding_id(
                        compute_finding_id(pmcid, cs.chunk_id, pos, f.claim)
                    )

            # Verbatim-from-DB (replace LLM paraphrases before grounding)
            self._replace_verbatim_from_db(chunk_summaries)

            # Grounding (matches sync ordering)
            grounding_rejected: list = []
            if self._grounding is not None:
                new_summaries = []
                for cs in chunk_summaries:
                    kept_cs, dropped = self._grounding.filter_findings_with_scores(cs)
                    new_summaries.append(kept_cs)
                    for f in dropped:
                        grounding_rejected.append((cs.chunk_id, f))
                chunk_summaries = new_summaries

            # MAP artifact persistence (filesystem + DB)
            persist_map_artifacts(writer, pmcid, chunk_summaries, grounding_rejected)
            self._persist_map_findings(pipeline_run_db_id, pmcid, chunk_summaries)

            # Optional modern chain — produces canonical/relate/final
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
                from ..entities.entity_linker import enrich_rules_with_cuis  # noqa: PLC0415
                enrich_rules_with_cuis(canonical_rules)
                logger.info("[%s] UMLS enrichment — done [%.1fs]",
                            pmcid, _time.perf_counter() - t0)
                persist_canonicalize_artifacts(writer, pmcid, canonical_rules)
                cr_db_id_map = self._persist_canonical_rules(
                    pipeline_run_db_id, pmcid, canonical_rules, fg_db_id_map,
                )
                # Cross-paper relate must run after canonical_rules are in DB.
                # Gate on pipeline_run_db_id (B-084): with --no-db this run
                # persists nothing and the DB holds a *different* corpus (e.g. a
                # held-out split rebuilt against the related15 DB), so the
                # incremental compare would pool the wrong rules — C(N,2) gate
                # churn against the wrong corpus — and write 0-row deletes. The
                # isolated cross-paper result still comes from the final
                # file-based _run_corpus_relate pass over the split's own dir.
                if pipeline_run_db_id is not None:
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

            # Rejection summary (sync parity)
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

            # NER + UMLS linking (optional)
            if self._run_ner and self._db is not None:
                logger.info("[%s] NER — running entity extraction + UMLS linking", pmcid)
                try:
                    from nlp_histo.ner.ner import run_ner_on_db  # noqa: PLC0415
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
            self._save_result(result)
            logger.info("[%s] Result saved to %s", pmcid, self._result_path(pmcid))

            self._write_batch_cost_artifacts(handle, writer, run_id, pmcid)

            # Issue G — post-run row-count audit (batch parity with sync).
            # Gate on pipeline_run_db_id too (B-084): with --no-db nothing was
            # persisted (every per-stage persister no-ops), so the audit would
            # fail every table by construction. Only audit when this run is
            # actually writing to the DB.
            if self._db is not None and pipeline_run_db_id is not None:
                try:
                    from ..observability.health_checks import assert_persistence_row_counts  # noqa: PLC0415
                    assert_persistence_row_counts(self._db, pmcid)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[%s] Post-run row-count audit raised — ignored: %s",
                        pmcid, exc,
                    )

            self._finish_pipeline_run(
                pipeline_run_db_id, "success",
                narrative_summary=None,
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

    # Filesystem artifact helpers

    def _make_run_id(self, pmcid: str) -> str:
        """Mirror KnowledgeExtractionRunner._make_run_id format for consistency."""
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

    # DB / cache / verbatim helpers (parity with sync runner)
    # TODO: deduplicate with KnowledgeExtractionRunner — these are line-for-line copies.
    # Lift into pipeline/stages/knowledge_extraction/persistence.py as module-level fns
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
            "enable_router":          self._enable_router,
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
            # the MAP agreement-gate hard-fail policy changes. Companion to
            # MAP_SCHEMA_VERSION (chunk-level PipelineCache); both must move
            # together to fully invalidate stale KEEP decisions.
            "map_agreement_policy_version": MAP_AGREEMENT_POLICY_VERSION,
            # B-080 — citation filter; kept identical to the sync runner's hash.
            "citation_enabled":        cfg.citation.enabled,
            "citation_check_verbatim": cfg.citation.check_verbatim if cfg.citation.enabled else None,
            "citation_fabricated_threshold": (
                cfg.citation.fabricated_threshold
                if cfg.citation.enabled and cfg.citation.check_verbatim else None
            ),
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

    def _clear_normalized_run_data(self, db_id: int, pmcid: str) -> None:
        _persistence_clear_normalized_run_data(self._db, db_id, pmcid)

    def _replace_verbatim_from_db(self, chunk_summaries: list) -> None:
        _persistence_replace_verbatim_from_db(self._db, chunk_summaries)

    def _persist_map_findings(self, db_id, pmcid, chunk_summaries) -> None:
        _persistence_persist_map_findings(self._db, db_id, pmcid, chunk_summaries)

    def _persist_normal_findings(self, db_id, pmcid, normal_findings) -> dict[str, int]:
        return _persistence_persist_normal_findings(
            self._db, db_id, pmcid, normal_findings,
        )

    def _persist_finding_groups(self, db_id, pmcid, finding_groups, nf_db_id_map) -> dict[str, int]:
        return _persistence_persist_finding_groups(
            self._db, db_id, pmcid, finding_groups, nf_db_id_map,
        )

    def _persist_canonical_rules(self, db_id, pmcid, canonical_rules, fg_db_id_map) -> dict[str, int]:
        return _persistence_persist_canonical_rules(
            self._db, db_id, pmcid, canonical_rules, fg_db_id_map,
        )

    def _persist_relations(self, db_id, pmcid, relations, cr_db_id_map) -> None:
        _persistence_persist_relations(
            self._db, db_id, pmcid, relations, cr_db_id_map,
        )

    def _persist_final_rules(self, db_id, pmcid, final_rules, cr_db_id_map) -> None:
        _persistence_persist_final_rules(
            self._db, db_id, pmcid, final_rules, cr_db_id_map,
        )

    def _persist_rejection_summary(self, db_id, rejection_summary) -> None:
        _persistence_persist_rejection_summary(self._db, db_id, rejection_summary)

    def _corpus_relate_incremental(self, pmcid: str, canonical_rules: list) -> None:
        if self._db is None:
            return
        try:
            from ..stages.corpus_relate import CorpusRelateStage  # noqa: PLC0415
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

    # Internal helpers

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

        # Pass 1: parse all voter outputs for every chunk
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

        # Pass 2: batch-embed all unique claims upfront
        # Filter Nones — voters_full slots stay None when a voter's batch result
        # was missing or unparseable (B-065). Pass 3 below handles all-None
        # chunks via the survivor_indices filter; Pass 2 must do the same or
        # `_extract_claims(None)` crashes with AttributeError. Latent for the
        # ``real`` profile (3 L1 voters; rarely all 3 fail on one chunk) but
        # certain for ``haiku_only`` (N=1 → any parse failure → [None]).
        from ..agreement.embedding import _claims as _extract_claims
        all_texts: list[str] = list({
            c
            for voters in chunk_voters.values()
            for v in voters
            if v is not None
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
        # function is plugged into the chosen strategy so the pre-embedded
        # claim cache is reused for every pair. Strategy choice is read from
        # cfg.agreement.scorer_kind (default "embedding") — identical to the
        # runner-level construction above.
        agreement = AgreementChecker(
            scorer=SemanticAgreementScorer.from_agreement_config(
                self._cfg_full.agreement, embed_fn=_cached_embed,
            ),
            theta=self._agreement.theta,
            reject_theta=self._agreement.reject_theta,
            single_voter_policy=self._legacy_single_voter_policy,  # type: ignore[arg-type]
            force_escalate_on_polarity_conflict=(
                self._cfg_full.agreement.force_escalate_on_polarity_conflict
            ),
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

        # Pass 3: agreement scoring from cache — no API calls
        escalated: list[str] = []
        dropped_chunks: list[str] = []  # legacy drop-on-reject (reject_theta > 0)

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
            elif outcome.rejected and not self._enable_router:
                # Legacy drop-on-reject: deferral <= reject_theta, so the chunk is
                # neither finalized nor escalated — it emits nothing. Mirrors the
                # sync MapStage legacy path. With reject_theta=0.0 (default) this
                # is unreachable for non-degenerate chunks.
                logger.info(
                    "[%s] Chunk %s rejected at %s (deferral <= reject_theta) — dropping",
                    pmcid, chunk_id, level.upper(),
                )
                dropped_chunks.append(chunk_id)
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

        kept = len(targets) - len(escalated) - len(dropped_chunks)
        logger.info(
            "[%s] %s complete: %d kept, %d escalated to %s, %d dropped",
            pmcid, level.upper(), kept, len(escalated), next_level.upper(),
            len(dropped_chunks),
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


